/*
 * CuTe SM100 UMMA TF32 GEMM with inline FP8 dequant for weights
 *
 * D[M,N] = A[M,K](float32 activations) * B[N,K]^T(FP8 weights × float32 scale)
 *
 * A (activations): float32, loaded via cooperative_copy
 * B (weights): FP8 E4M3, converted to tfloat32_t inline with block scale multiplication
 * D (output): float32
 */

#include <cutlass/arch/barrier.h>
#include <cute/arch/tmem_allocator_sm100.hpp>

__device__ __forceinline__ float d_fp8f(uint8_t v) {
    __nv_fp8_e4m3 f; f.__x = v; return float(f);
}

// Standard float32×float32 CuTe GEMM (for activations that are already float32)
template <int BK_TILES>
__global__ static void __launch_bounds__(128)
cute_gemm_tf32(
    const float* __restrict__ A_ptr, int lda,
    const float* __restrict__ B_ptr, int ldb,
    float* __restrict__ D_ptr, int ldd,
    int M, int N, int K
) {
    using namespace cute;

    int tile_m = blockIdx.x * 128;
    int tile_n = blockIdx.y * 128;
    if (tile_m >= M || tile_n >= N) return;

    auto tiled_mma = make_tiled_mma(
        SM100_MMA_TF32_SS<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                          128, 128, UMMA::Major::K, UMMA::Major::K>{});
    constexpr int BK = BK_TILES * 8;

    auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(Int<128>{}, Int<BK>{}));
    auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(Int<128>{}, Int<BK>{}));
    auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<cutlass::tfloat32_t>{}, mma_shape_A);
    auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<cutlass::tfloat32_t>{}, mma_shape_B);

    extern __shared__ char smem[];
    constexpr int sA_sz = cosize(decltype(sA_layout){});
    constexpr int sB_sz = cosize(decltype(sB_layout){});
    auto* sA_p = reinterpret_cast<cutlass::tfloat32_t*>(smem);
    auto* sB_p = sA_p + sA_sz;
    auto* bar = reinterpret_cast<uint64_t*>(sB_p + sB_sz);
    auto* tptr = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(bar) + 16);

    Tensor sA = make_tensor(make_smem_ptr(sA_p), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(sB_p), sB_layout);

    Tensor gA = make_tensor(make_gmem_ptr(reinterpret_cast<const cutlass::tfloat32_t*>(A_ptr + tile_m * lda)),
                            make_layout(make_shape(Int<128>{}, K), make_stride(lda, Int<1>{})));
    Tensor gB = make_tensor(make_gmem_ptr(reinterpret_cast<const cutlass::tfloat32_t*>(B_ptr + tile_n * ldb)),
                            make_layout(make_shape(Int<128>{}, K), make_stride(ldb, Int<1>{})));
    Tensor gD = make_tensor(make_gmem_ptr(D_ptr + tile_m * ldd + tile_n),
                            make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(ldd, Int<1>{})));

    auto k_tiler = make_shape(Int<128>{}, Int<BK>{});
    Tensor gA_t = local_tile(gA, k_tiler, make_coord(0, _));
    Tensor gB_t = local_tile(gB, k_tiler, make_coord(0, _));

    ThrMMA cta_mma = tiled_mma.get_slice(_0{});
    Tensor tCgA = cta_mma.partition_A(gA_t);
    Tensor tCgB = cta_mma.partition_B(gB_t);
    Tensor tCgD = cta_mma.partition_C(gD);

    Tensor tCrA = cta_mma.make_fragment_A(sA);
    Tensor tCrB = cta_mma.make_fragment_B(sB);
    Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);

    uint32_t et = cute::elect_one_sync();
    uint32_t ew = (threadIdx.x / 32 == 0);
    using TA = cute::TMEM::Allocator1Sm;
    TA ta{};
    if (ew) ta.allocate(TA::Sm100TmemCapacityColumns, tptr);
    __syncthreads();
    tCtAcc.data() = *tptr;

    if (ew && et) cute::initialize_barrier(*bar, 1);
    int ph = 0;
    __syncthreads();

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    int num_k_tiles = K / BK;
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        cooperative_copy<128>(threadIdx.x, tCgA(_,_,_,kt), sA);
        cooperative_copy<128>(threadIdx.x, tCgB(_,_,_,kt), sB);
        __syncthreads();

        if (ew) {
            for (int kb = 0; kb < size<2>(tCrA); ++kb) {
                gemm(tiled_mma, tCrA(_,_,kb), tCrB(_,_,kb), tCtAcc);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(bar);
        }
        cute::wait_barrier(*bar, ph);
        ph ^= 1;
    }

    TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
    ThrCopy tc = t2r.get_slice(threadIdx.x);
    Tensor tDt = tc.partition_S(tCtAcc);
    Tensor tDg = tc.partition_D(tCgD);
    Tensor tDr = make_tensor<float>(shape(tDg));
    copy(t2r, tDt, tDr);
    copy(tDr, tDg);

    __syncthreads();
    if (ew) { ta.release_allocation_lock(); ta.free(*tptr, TA::Sm100TmemCapacityColumns); }
}

// FP8 weight dequant GEMM: A=float32 activations, B=FP8 weights with block scale
// D[M,N] = A[M,K] * (FP8_B[N,K] * scale[N/128, K/128])^T
template <int BK_TILES>
__global__ static void __launch_bounds__(128)
cute_gemm_fp8dq(
    const float* __restrict__ A_ptr, int lda,            // float32 activations [M, K]
    const uint8_t* __restrict__ B_fp8, int ldb,           // FP8 weights [N, K]
    const float* __restrict__ B_scale, int scale_stride,  // scales [N/128, K/128], stride = K/128
    float* __restrict__ D_ptr, int ldd,
    int M, int N, int K
) {
    using namespace cute;

    int tile_m = blockIdx.x * 128;
    int tile_n = blockIdx.y * 128;
    if (tile_m >= M || tile_n >= N) return;

    auto tiled_mma = make_tiled_mma(
        SM100_MMA_TF32_SS<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                          128, 128, UMMA::Major::K, UMMA::Major::K>{});
    constexpr int BK = BK_TILES * 8;
    constexpr int QB = 128;

    auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(Int<128>{}, Int<BK>{}));
    auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(Int<128>{}, Int<BK>{}));
    auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<cutlass::tfloat32_t>{}, mma_shape_A);
    auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<cutlass::tfloat32_t>{}, mma_shape_B);

    extern __shared__ char smem[];
    constexpr int sA_sz = cosize(decltype(sA_layout){});
    constexpr int sB_sz = cosize(decltype(sB_layout){});
    auto* sA_p = reinterpret_cast<cutlass::tfloat32_t*>(smem);
    auto* sB_p = sA_p + sA_sz;
    auto* bar = reinterpret_cast<uint64_t*>(sB_p + sB_sz);
    auto* tptr = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(bar) + 16);

    Tensor sA = make_tensor(make_smem_ptr(sA_p), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(sB_p), sB_layout);

    // A (activations): same as before — float32 from GMEM via cooperative_copy
    Tensor gA = make_tensor(make_gmem_ptr(reinterpret_cast<const cutlass::tfloat32_t*>(A_ptr + tile_m * lda)),
                            make_layout(make_shape(Int<128>{}, K), make_stride(lda, Int<1>{})));
    Tensor gD = make_tensor(make_gmem_ptr(D_ptr + tile_m * ldd + tile_n),
                            make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(ldd, Int<1>{})));

    auto k_tiler = make_shape(Int<128>{}, Int<BK>{});
    Tensor gA_t = local_tile(gA, k_tiler, make_coord(0, _));

    ThrMMA cta_mma = tiled_mma.get_slice(_0{});
    Tensor tCgA = cta_mma.partition_A(gA_t);
    Tensor tCgD = cta_mma.partition_C(gD);

    Tensor tCrA = cta_mma.make_fragment_A(sA);
    Tensor tCrB = cta_mma.make_fragment_B(sB);
    Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);

    uint32_t et = cute::elect_one_sync();
    uint32_t ew = (threadIdx.x / 32 == 0);
    using TmA = cute::TMEM::Allocator1Sm;
    TmA ta{};
    if (ew) ta.allocate(TmA::Sm100TmemCapacityColumns, tptr);
    __syncthreads();
    tCtAcc.data() = *tptr;

    if (ew && et) cute::initialize_barrier(*bar, 1);
    int ph = 0;
    __syncthreads();

    // B weight scale: one scale per 128×128 block
    // tile_n/QB = n_block for this N-tile
    int n_block = tile_n / QB;

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    int num_k_tiles = K / BK;
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k_off = kt * BK;

        // A: cooperative_copy from float32 GMEM → swizzled SMEM
        cooperative_copy<128>(threadIdx.x, tCgA(_,_,_,kt), sA);

        // B: MANUAL load FP8 from GMEM → convert to tfloat32_t × scale → write to swizzled SMEM
        // sB has shape ((_128, _8), _1, BK/8) with swizzle Sw<3,4,3>
        // We write to sB using coordinate indexing — CuTe handles swizzle
        float b_scale_val = B_scale[n_block * scale_stride + (k_off / QB)];
        for (int idx = threadIdx.x; idx < 128 * BK; idx += 128) {
            int n = idx / BK;     // which row (0-127) within this N-tile
            int k = idx % BK;     // which K element (0-BK)
            uint8_t fp8_byte = B_fp8[(tile_n + n) * ldb + k_off + k];
            float dequanted = d_fp8f(fp8_byte) * b_scale_val;
            // Write to swizzled SMEM using the MMA-compatible layout
            // Coordinate: (n, k) in the 128×BK tile
            // sB shape: ((_128, _8), _1, _BK/8) — first mode: (N_within_mma, K_within_mma)
            int k_mma = k % 8;   // K position within one MMA K-step
            int k_tile_idx = k / 8;  // Which MMA K-step
            sB(make_coord(make_coord(n, k_mma), Int<0>{}, k_tile_idx)) = cutlass::tfloat32_t(dequanted);
        }
        __syncthreads();

        if (ew) {
            for (int kb = 0; kb < size<2>(tCrA); ++kb) {
                gemm(tiled_mma, tCrA(_,_,kb), tCrB(_,_,kb), tCtAcc);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(bar);
        }
        cute::wait_barrier(*bar, ph);
        ph ^= 1;
    }

    TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
    ThrCopy tc = t2r.get_slice(threadIdx.x);
    Tensor tDt = tc.partition_S(tCtAcc);
    Tensor tDg = tc.partition_D(tCgD);
    Tensor tDr = make_tensor<float>(shape(tDg));
    copy(t2r, tDt, tDr);
    copy(tDr, tDg);

    __syncthreads();
    if (ew) { ta.release_allocation_lock(); ta.free(*tptr, TmA::Sm100TmemCapacityColumns); }
}

static int cute_gemm_smem_size() {
    using namespace cute;
    auto mma = make_tiled_mma(SM100_MMA_TF32_SS<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                              128, 128, UMMA::Major::K, UMMA::Major::K>{});
    constexpr int BK = 16 * 8;
    auto sa = partition_shape_A(mma, make_shape(Int<128>{}, Int<BK>{}));
    auto sb = partition_shape_B(mma, make_shape(Int<128>{}, Int<BK>{}));
    auto la = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<cutlass::tfloat32_t>{}, sa);
    auto lb = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<cutlass::tfloat32_t>{}, sb);
    return (cosize(la) + cosize(lb)) * sizeof(cutlass::tfloat32_t) + 128;
}
