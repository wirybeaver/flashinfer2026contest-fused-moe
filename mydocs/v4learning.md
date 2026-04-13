# Optimization Learnings -- Fused MoE Triton Kernel on B200

## 1. FP8 Native Dot Product for GEMM1 (10-14% overall speedup)

**Discovery:** Replacing `cast-to-f32 + tf32 dot` with native FP8 `tl.dot()` and post-multiply scale gives a massive speedup on B200's FP8 tensor cores.

**Isolated benchmark (8192x4096x7168 GEMM):**
| Method | Time | TFLOPS | Speedup |
|--------|------|--------|---------|
| TF32 (cast FP8->f32, pre-scale, tf32 dot) | 2.673ms | 180 | 1.0x |
| FP8 (native dot, post-scale) BK=64 | 0.628ms | 766 | 4.3x |
| FP8 (native dot, post-scale) BK=128 | 0.360ms | 1338 | 7.4x |

**Key implementation detail:** With `BLOCK_K=128` matching `SCALE_BLOCK=128`, each K-loop iteration corresponds to exactly one scale block, making the post-scale application trivial:
```python
partial = tl.dot(a_fp8, w_fp8)  # native FP8 tensor cores
acc += partial * (h_scale[:, None] * w_scale_scalar)
```

**Precision:** Max abs error ~0.0002 vs tolerance of atol=1. No precision concerns at all.

---

## 2. num_stages=1 is Optimal for Scattered-Access Persistent Kernels

**Discovery:** Triton's software pipelining (`num_stages > 1`) dramatically *hurts* performance for kernels with non-contiguous memory access patterns.

**GEMM1 kernel sweep (FP8 dot, BK=128, warps=8):**
| num_stages | Time | Relative |
|------------|------|----------|
| 1 | 2.996ms | 1.0x (best) |
| 2 | 7.351ms | 2.5x slower |
| 3 | 7.456ms | 2.5x slower |
| 4 | 7.430ms | 2.5x slower |

**Root cause:** The GEMM1 kernel has three properties that break pipelining:

1. **Scattered token loading** -- Hidden states are loaded via indirect `token_ids` indexing (data-dependent, non-contiguous). The hardware prefetcher and Triton's pipelining can't predict these addresses.

2. **Three loads per K-iteration** -- Each iteration loads `a` (hidden states), `w_gate`, and `w_up`. With `num_stages=4`, Triton allocates 4 stages x 3 tiles = 12 shared memory buffers, causing severe register pressure and spilling.

3. **Post-scale compute between load and accumulate** -- Scale loads and element-wise multiplies between the FP8 dot and accumulation break the simple load-compute pipeline that multi-stage assumes.

**Contrast with GEMM2:** GEMM2's best config is `num_stages=3` because its input (c_buf) is a dense contiguous buffer with regular strides, which pipelines perfectly.

**Takeaway:** Always profile `num_stages` for each kernel independently. Scattered/indirect memory access patterns strongly favor `num_stages=1`.

---

## 3. BF16 and FP8 Intermediate Buffers Fail Precision

**Discovery:** Reducing the SwiGLU intermediate buffer (c_buf) from f32 to lower precision fails the contest's correctness checks.

| c_buf dtype | Workloads passing | Notes |
|------------|-------------------|-------|
| float32 | 19/19 | Current, works perfectly |
| bfloat16 | 9/19 | BF16 has ~7 mantissa bits, loses small values in wide-range SwiGLU output |
| float8_e4m3fn (per-row scale) | 0/19 | FP8 has ~3.5 mantissa bits, max abs error 12K-39K vs tolerance of 1 |

**Why:** SwiGLU output (`silu(up) * gate`) has wide dynamic range within each row. Even with per-row scaling, quantizing to fewer mantissa bits loses too much information in the subsequent GEMM2 multiplication.

---

## 4. CPU-GPU Sync Elimination via Upper-Bound Tiles is Counterproductive

**Discovery:** Replacing `.item()` syncs with upper-bound tile counts causes more harm than it solves.

The kernel has two `.item()` calls (~0.064ms each):
```python
total = int(expert_offsets[-1].item())        # sync 1
total_m_tiles = int(m_tile_offsets[-1].item()) # sync 2
```

**Attempted fix:** Use `total_m_tiles_ub = (total + BM - 1) // BM + NUM_LOCAL_EXPERTS` as upper bound.

**Result:** Small workloads went from ~0.6ms to ~1.3ms (2x regression). The extra empty tiles (up to 32 per workload) each still execute the binary search and loads with M=0, costing more than the 0.064ms sync they replaced.

---

## 5. `flatten=True` Hurts Complex Persistent Kernels

**Discovery:** Triton's `tl.range(..., flatten=True)` option (designed for Blackwell loop flattening) causes 10-15% regression on our kernels.

Our kernels have complex control flow inside the persistent loop (binary search for expert_id, scattered loads, conditional scale application). The loop flattening optimization conflicts with this complexity, producing worse code than the standard loop.

**Takeaway:** `flatten=True` is beneficial for simple dense GEMMs (as shown in the Triton tutorial) but harmful for kernels with data-dependent control flow.

---

## 6. Tile Ordering: Row-Major is Already Optimal for Grouped Expert GEMMs

**Discovery:** Column-major tile ordering and GROUP_SIZE_M swizzle patterns don't help (and slightly hurt) our grouped expert GEMM structure.

**Why:** In our kernel, the dominant memory traffic is the c_buf input (f32, [M, 2048]). Row-major ordering naturally processes all N-tiles for the same M-row consecutively, keeping c_buf data in L2. Column-major breaks this locality.

The GROUP_SIZE_M swizzle from CUTLASS/Triton tutorials is designed for dense GEMMs where both A and B matrices need L2 reuse. In our grouped-expert structure, weight data is per-expert and doesn't benefit from cross-expert sharing.

---

## 7. B200 Hardware Facts (Relevant to Optimization Decisions)

- **148 SMs** on B200
- **232 KB shared memory** per SM
- **~8 TB/s HBM bandwidth**
- **~4500 TFLOPS FP8**, ~2250 TFLOPS TF32
- **96 MB L2 cache**
- FP8 tensor cores have ~2x throughput vs TF32
- BF16 atomic_add is supported but has limited precision for multi-expert accumulation

---

## 8. Modal vs Local Benchmark Comparison

Local B200 and Modal B200 give very close numbers (within ~5%), but Modal is the single source of truth. Always validate on Modal before declaring an improvement.

| Metric | Local | Modal |
|--------|-------|-------|
| Average speedup | 14.0x | 14.33x |
| Total latency | 26.16ms | 25.39ms |
| Pass rate | 19/19 | 19/19 |
