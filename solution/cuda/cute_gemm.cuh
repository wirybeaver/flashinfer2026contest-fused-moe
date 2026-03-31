/*
 * CuTe SM100 UMMA GEMM kernel — float32 inputs (step 1)
 * Uses cooperative_copy + SM100_MMA_TF32_SS
 */

#if defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)

#include <cutlass/arch/barrier.h>
#include <cute/arch/tmem_allocator_sm100.hpp>

// CuTe GEMM kernel: D[M,N] = A[M,K] * B[N,K]^T
// A is K-major (RowMajor), B is K-major (ColumnMajor), D is N-major (RowMajor)
// Both A and B use float32, MMA uses TF32 precision
// Grid: (ceil(M/128), ceil(N/128)), Block: 128 threads
template <int BK_TILES>  // Number of MMA K-steps per tile (BK = BK_TILES * MMA_K)
__global__ static void __launch_bounds__(128)
cute_gemm_tf32(
    const float* __restrict__ A_ptr, int A_stride,  // A[M, K] RowMajor, stride = K
    const float* __restrict__ B_ptr, int B_stride,  // B[N, K] ColMajor, stride = K
    float* __restrict__ D_ptr, int D_stride,         // D[M, N] RowMajor, stride = N
    int M, int N, int K
) {
    using namespace cute;

    // TF32 MMA atom: 128×128×8 (K=8 for 32-bit types)
    auto tiled_mma = make_tiled_mma(
        SM100_MMA_TF32_SS<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                          128, 128, UMMA::Major::K, UMMA::Major::K>{});

    constexpr int MMA_K_DIM = 8;  // TF32 MMA K dimension

    // MMA tiler: (128, 128, BK) where BK = BK_TILES * MMA_K_DIM
    auto bK = Int<MMA_K_DIM>{} * Int<BK_TILES>{};
    auto mma_tiler = make_shape(Int<128>{}, Int<128>{}, bK);

    // Partition shapes for SMEM
    auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(Int<128>{}, bK));
    auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(Int<128>{}, bK));

    // Swizzled SMEM layouts
    auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<float>{}, mma_shape_A);
    auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<float>{}, mma_shape_B);

    // SMEM allocation
    extern __shared__ char smem[];
    constexpr int sA_size = cosize(decltype(sA_layout){});
    constexpr int sB_size = cosize(decltype(sB_layout){});

    auto* sA_ptr = reinterpret_cast<float*>(smem);
    auto* sB_ptr = reinterpret_cast<float*>(smem + sizeof(float) * sA_size);
    auto* barrier_ptr = reinterpret_cast<uint64_t*>(smem + sizeof(float) * (sA_size + sB_size));
    auto* tmem_ptr_storage = reinterpret_cast<uint32_t*>(smem + sizeof(float) * (sA_size + sB_size) + 16);

    Tensor tCsA = make_tensor(make_smem_ptr(sA_ptr), sA_layout);
    Tensor tCsB = make_tensor(make_smem_ptr(sB_ptr), sB_layout);

    // GMEM tensors
    // A[M, K] RowMajor: stride = (K, 1)
    auto layout_A = make_layout(make_shape(M, K), make_stride(A_stride, Int<1>{}));
    auto layout_B = make_layout(make_shape(N, K), make_stride(B_stride, Int<1>{}));
    auto layout_D = make_layout(make_shape(M, N), make_stride(D_stride, Int<1>{}));

    Tensor mA = make_tensor(make_gmem_ptr(A_ptr), layout_A);
    Tensor mB = make_tensor(make_gmem_ptr(B_ptr), layout_B);
    Tensor mD = make_tensor(make_gmem_ptr(D_ptr), layout_D);

    // CTA coordinates
    auto mma_coord = make_coord(_0{}, (int)blockIdx.x, (int)blockIdx.y, _);
    Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X,_1>{});
    Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step< X,_1,_1>{});
    Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1,_1, X>{});

    // MMA partitioning
    ThrMMA cta_mma = tiled_mma.get_slice(_0{});
    Tensor tCgA = cta_mma.partition_A(gA);
    Tensor tCgB = cta_mma.partition_B(gB);
    Tensor tCgD = cta_mma.partition_C(gD);

    Tensor tCrA = cta_mma.make_fragment_A(tCsA);
    Tensor tCrB = cta_mma.make_fragment_B(tCsB);
    Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);

    // TMEM allocation
    uint32_t elect_one_thr = cute::elect_one_sync();
    uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

    using TmemAlloc = cute::TMEM::Allocator1Sm;
    TmemAlloc tmem_alloc{};
    if (elect_one_warp) {
        tmem_alloc.allocate(TmemAlloc::Sm100TmemCapacityColumns, tmem_ptr_storage);
    }
    __syncthreads();
    tCtAcc.data() = *tmem_ptr_storage;

    // Barrier
    if (elect_one_warp && elect_one_thr) {
        cute::initialize_barrier(*barrier_ptr, 1);
    }
    int phase = 0;
    __syncthreads();

    // Mainloop
    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

    for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
        cooperative_copy<128>(threadIdx.x, tCgA(_,_,_,k_tile), tCsA);
        cooperative_copy<128>(threadIdx.x, tCgB(_,_,_,k_tile), tCsB);
        __syncthreads();

        if (elect_one_warp) {
            for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(barrier_ptr);
        }
        cute::wait_barrier(*barrier_ptr, phase);
        phase ^= 1;
    }

    // Epilogue: TMEM → RMEM → GMEM
    TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
    ThrCopy thr_t2r = t2r.get_slice(threadIdx.x);

    Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
    Tensor tDgD = thr_t2r.partition_D(tCgD);
    Tensor tDrAcc = make_tensor<float>(shape(tDgD));

    copy(t2r, tDtAcc, tDrAcc);
    copy(tDrAcc, tDgD);

    __syncthreads();
    if (elect_one_warp) {
        tmem_alloc.release_allocation_lock();
        tmem_alloc.free(*tmem_ptr_storage, TmemAlloc::Sm100TmemCapacityColumns);
    }
}

// Calculate shared memory size for the CuTe GEMM
static int cute_gemm_smem_size() {
    using namespace cute;
    auto tiled_mma = make_tiled_mma(
        SM100_MMA_TF32_SS<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                          128, 128, UMMA::Major::K, UMMA::Major::K>{});
    constexpr int BK = 8 * 16;
    auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(Int<128>{}, Int<BK>{}));
    auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(Int<128>{}, Int<BK>{}));
    auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<float>{}, mma_shape_A);
    auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<float>{}, mma_shape_B);
    int sA_bytes = cosize(sA_layout) * sizeof(float);
    int sB_bytes = cosize(sB_layout) * sizeof(float);
    return sA_bytes + sB_bytes + 128;  // + barriers + tmem ptr
}

#endif
