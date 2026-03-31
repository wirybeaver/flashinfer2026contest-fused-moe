/*
 * CuTe SM100 UMMA TF32 GEMM kernel
 */

#include <cutlass/arch/barrier.h>
#include <cute/arch/tmem_allocator_sm100.hpp>

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

    // GMEM tiles: A[128, K], B[128, K], D[128, 128]
    Tensor gA = make_tensor(make_gmem_ptr(reinterpret_cast<const cutlass::tfloat32_t*>(A_ptr + tile_m * lda)),
                            make_layout(make_shape(Int<128>{}, K), make_stride(lda, Int<1>{})));
    Tensor gB = make_tensor(make_gmem_ptr(reinterpret_cast<const cutlass::tfloat32_t*>(B_ptr + tile_n * ldb)),
                            make_layout(make_shape(Int<128>{}, K), make_stride(ldb, Int<1>{})));
    Tensor gD = make_tensor(make_gmem_ptr(D_ptr + tile_m * ldd + tile_n),
                            make_layout(make_shape(Int<128>{}, Int<128>{}), make_stride(ldd, Int<1>{})));

    // K-tile: (128, K) → ((128, BK), K/BK) using local_tile
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
    for (int kt = 0; kt < size<3>(tCgA); ++kt) {
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

static int cute_gemm_smem_size() {
    using namespace cute;
    auto mma = make_tiled_mma(SM100_MMA_TF32_SS<cutlass::tfloat32_t, cutlass::tfloat32_t, float,
                              128, 128, UMMA::Major::K, UMMA::Major::K>{});
    auto sa = partition_shape_A(mma, make_shape(Int<128>{}, Int<128>{}));
    auto sb = partition_shape_B(mma, make_shape(Int<128>{}, Int<128>{}));
    auto la = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<cutlass::tfloat32_t>{}, sa);
    auto lb = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<cutlass::tfloat32_t>{}, sb);
    return (cosize(la) + cosize(lb)) * sizeof(cutlass::tfloat32_t) + 128;
}
