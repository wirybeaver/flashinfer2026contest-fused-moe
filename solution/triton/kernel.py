"""
Track A (fused_moe) Triton implementation.

Entry point: kernel(...)
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# Fixed geometry/constants from the Track A definition.
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 256
NUM_LOCAL_EXPERTS = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLOCK = 128
USE_TORCH_ROUTING = True
USE_TORCH_DEQUANT = True
USE_TORCH_WEIGHT_DEQUANT = False
USE_TORCH_GEMM = False
USE_TORCH_COMPACTION = False
USE_TORCH_SWIGLU = False
USE_TORCH_ACCUMULATE = False
USE_TORCH_GEMM1 = False
USE_TORCH_GEMM2 = False
USE_FP32_GEMM = False
GEMM1_INPUT_PRECISION = "tf32"
GEMM2_INPUT_PRECISION = "tf32"
GEMM1_CORRECT_K_BLOCKS = 0
GEMM2_CORRECT_K_BLOCKS = 0
TORCH_MATMUL_PRECISION = "high"
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 128
KEEP_A_FP32 = True


def _block_dequant_matrix(
    matrix_fp8: torch.Tensor,
    scales: torch.Tensor,
    num_row_blocks: int,
    num_col_blocks: int,
    block_size: int,
) -> torch.Tensor:
    return (
        matrix_fp8.to(torch.float32)
        .view(num_row_blocks, block_size, num_col_blocks, block_size)
        .mul(scales.to(torch.float32).view(num_row_blocks, 1, num_col_blocks, 1))
        .reshape(num_row_blocks * block_size, num_col_blocks * block_size)
    )


@triton.jit
def routing_kernel(
    routing_logits_ptr,
    routing_bias_ptr,
    topk_idx_ptr,
    topk_weight_ptr,
    stride_logits_t,
    stride_topk_t,
    num_tokens,
    routed_scaling_factor,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    offs = tl.arange(0, NUM_EXPERTS)
    logits = tl.load(routing_logits_ptr + pid * stride_logits_t + offs, mask=offs < NUM_EXPERTS, other=0.0)
    bias = tl.load(routing_bias_ptr + offs, mask=offs < NUM_EXPERTS, other=0.0)

    s = 1.0 / (1.0 + tl.exp(-logits))
    s_with_bias = s + bias

    # Group scores: sum of top-2 within each group.
    group_scores = tl.zeros([N_GROUP], dtype=tl.float32)
    group_ids = tl.arange(0, N_GROUP)
    group_size = NUM_EXPERTS // N_GROUP
    group_id = offs // group_size
    for g in tl.static_range(0, N_GROUP):
        mask = group_id == g
        vals = tl.where(mask, s_with_bias, -float("inf"))
        max1 = tl.max(vals, axis=0)
        best_idx = tl.max(tl.where(vals == max1, offs, -1), axis=0)
        vals2 = tl.where(offs == best_idx, -float("inf"), vals)
        max2 = tl.max(vals2, axis=0)
        score = max1 + max2
        group_scores = tl.where(group_ids == g, score, group_scores)

    # Select topk_group groups.
    group_keep = tl.zeros([N_GROUP], dtype=tl.int1)
    for _ in tl.static_range(0, TOPK_GROUP):
        masked = tl.where(group_keep, -float("inf"), group_scores)
        best_val = tl.max(masked, axis=0)
        best_idx = tl.max(tl.where(masked == best_val, group_ids, -1), axis=0)
        group_keep = group_keep | (group_ids == best_idx)

    # Mask experts not in kept groups.
    group_id = offs // (NUM_EXPERTS // N_GROUP)
    group_keep_by_expert = tl.zeros([NUM_EXPERTS], dtype=tl.int1)
    for g in tl.static_range(0, N_GROUP):
        keep = tl.sum(tl.where(group_ids == g, group_keep, 0), axis=0) != 0
        mask = group_id == g
        group_keep_by_expert = tl.where(mask, keep, group_keep_by_expert)

    scores_pruned = tl.where(group_keep_by_expert, s_with_bias, -float("inf"))

    # Global top-k over masked scores.
    selected = tl.zeros([NUM_EXPERTS], dtype=tl.int1)
    topk_idx = tl.zeros([TOP_K], dtype=tl.int32)
    for k in tl.static_range(0, TOP_K):
        masked = tl.where(selected, -float("inf"), scores_pruned)
        best_val = tl.max(masked, axis=0)
        best_idx = tl.max(tl.where(masked == best_val, offs, -1), axis=0)
        topk_idx = tl.where(tl.arange(0, TOP_K) == k, best_idx, topk_idx)
        selected = selected | (offs == best_idx)

    weights = s * selected
    weights_sum = tl.sum(weights, axis=0) + 1e-20
    weights_norm = weights / weights_sum * routed_scaling_factor

    topk_weight = tl.zeros([TOP_K], dtype=tl.float32)
    topk_range = tl.arange(0, TOP_K)
    for k in tl.static_range(0, TOP_K):
        idx = tl.sum(tl.where(topk_range == k, topk_idx, 0), axis=0)
        w = tl.sum(tl.where(offs == idx, weights_norm, 0.0), axis=0)
        topk_weight = tl.where(topk_range == k, w, topk_weight)

    # Store topk indices and weights.
    tl.store(topk_idx_ptr + pid * stride_topk_t + tl.arange(0, TOP_K), topk_idx)
    tl.store(topk_weight_ptr + pid * stride_topk_t + tl.arange(0, TOP_K), topk_weight)


@triton.jit
def count_kernel(
    topk_idx_ptr,
    expert_counts_ptr,
    stride_topk_t,
    num_tokens,
    local_expert_offset,
    TOP_K: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    offs = tl.arange(0, TOP_K)
    idx = tl.load(topk_idx_ptr + pid * stride_topk_t + offs)
    le = idx - local_expert_offset
    mask = (le >= 0) & (le < NUM_LOCAL_EXPERTS)
    le_safe = tl.where(mask, le, 0)
    tl.atomic_add(expert_counts_ptr + le_safe, 1, mask=mask)


@triton.jit
def scatter_kernel(
    topk_idx_ptr,
    topk_weight_ptr,
    token_idx_ptr,
    token_weight_ptr,
    expert_offsets_ptr,
    expert_positions_ptr,
    stride_topk_t,
    num_tokens,
    local_expert_offset,
    TOP_K: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    offs = tl.arange(0, TOP_K)
    idx = tl.load(topk_idx_ptr + pid * stride_topk_t + offs)
    w = tl.load(topk_weight_ptr + pid * stride_topk_t + offs)
    le = idx - local_expert_offset
    mask = (le >= 0) & (le < NUM_LOCAL_EXPERTS)
    le_safe = tl.where(mask, le, 0)

    pos = tl.atomic_add(expert_positions_ptr + le_safe, 1, mask=mask)
    start = tl.load(expert_offsets_ptr + le_safe, mask=mask, other=0)
    out_idx = start + pos
    tl.store(token_idx_ptr + out_idx, pid, mask=mask)
    tl.store(token_weight_ptr + out_idx, w, mask=mask)


@triton.jit
def dequant_hidden_kernel(
    hidden_ptr,
    scale_ptr,
    out_ptr,
    stride_hidden_t,
    stride_hidden_h,
    stride_scale_b,
    stride_scale_t,
    stride_out_t,
    stride_out_h,
    num_tokens,
    BLOCK: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    if pid_t >= num_tokens:
        return

    offs = tl.arange(0, BLOCK)
    h = pid_b * BLOCK + offs
    mask = h < HIDDEN_SIZE
    x = tl.load(hidden_ptr + pid_t * stride_hidden_t + h * stride_hidden_h, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + pid_b * stride_scale_b + pid_t * stride_scale_t)
    y = tl.cast(x, tl.float32) * scale
    tl.store(out_ptr + pid_t * stride_out_t + h * stride_out_h, tl.cast(y, tl.bfloat16), mask=mask)


@triton.jit
def fp8_gemm_kernel(
    a_ptr,
    w_ptr,
    w_scale_ptr,
    token_idx_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    stride_scale_n,
    stride_scale_k,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    token_ids = tl.load(token_idx_ptr + offs_m, mask=offs_m < M, other=0)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        a_ptrs = a_ptr + token_ids[:, None] * stride_am + k_ids[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        a = tl.cast(a, tl.float32)

        w_ptrs = w_ptr + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w = tl.cast(w, tl.float32)

        n_block = (pid_n * BLOCK_N) // SCALE_BLOCK
        k_block = k // SCALE_BLOCK
        scale = tl.load(w_scale_ptr + n_block * stride_scale_n + k_block * stride_scale_k)
        w = w * scale

        acc += tl.dot(a, w, input_precision=INPUT_PRECISION)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def fp8_gemm_kernel_correction(
    a_ptr,
    w_ptr,
    w_scale_ptr,
    token_idx_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    stride_scale_n,
    stride_scale_k,
    INPUT_PRECISION: tl.constexpr,
    CORRECT_K_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    token_ids = tl.load(token_idx_ptr + offs_m, mask=offs_m < M, other=0)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    k_base = K - CORRECT_K_BLOCKS * BLOCK_K
    for kk in range(0, CORRECT_K_BLOCKS * BLOCK_K, BLOCK_K):
        k_ids = k_base + kk + offs_k
        a_ptrs = a_ptr + token_ids[:, None] * stride_am + k_ids[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        a = tl.cast(a, tl.float32)

        w_ptrs = w_ptr + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w = tl.cast(w, tl.float32)

        n_block = (pid_n * BLOCK_N) // SCALE_BLOCK
        k_block = (k_base + kk) // SCALE_BLOCK
        scale = tl.load(w_scale_ptr + n_block * stride_scale_n + k_block * stride_scale_k)
        w = w * scale

        corr = tl.dot(a, w, input_precision="ieee") - tl.dot(a, w, input_precision=INPUT_PRECISION)
        acc += corr

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    c += acc
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def bf16_gemm_kernel(
    a_ptr,
    w_ptr,
    token_idx_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    token_ids = tl.load(token_idx_ptr + offs_m, mask=offs_m < M, other=0)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        a_ptrs = a_ptr + token_ids[:, None] * stride_am + k_ids[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        a = tl.cast(a, tl.float32)

        w_ptrs = w_ptr + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w = tl.cast(w, tl.bfloat16)

        acc += tl.dot(a, w)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def fp32_gemm_kernel(
    a_ptr,
    w_ptr,
    token_idx_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    token_ids = tl.load(token_idx_ptr + offs_m, mask=offs_m < M, other=0)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        a_ptrs = a_ptr + token_ids[:, None] * stride_am + k_ids[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        a = tl.cast(a, tl.float32)

        w_ptrs = w_ptr + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w = tl.cast(w, tl.float32)

        acc += tl.dot(a, w, input_precision="ieee")

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def bf16_gemm_contig_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        a = tl.cast(a, tl.float32)

        w_ptrs = w_ptr + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w = tl.cast(w, tl.bfloat16)

        acc += tl.dot(a, w)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def fp32_gemm_contig_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        a = tl.cast(a, tl.float32)

        w_ptrs = w_ptr + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w = tl.cast(w, tl.float32)

        acc += tl.dot(a, w, input_precision="ieee")

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def fp8_gemm_contig_kernel(
    a_ptr,
    w_ptr,
    w_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    stride_scale_n,
    stride_scale_k,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        a = tl.cast(a, tl.float32)

        w_ptrs = w_ptr + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w = tl.cast(w, tl.float32)

        n_block = (pid_n * BLOCK_N) // SCALE_BLOCK
        k_block = k // SCALE_BLOCK
        scale = tl.load(w_scale_ptr + n_block * stride_scale_n + k_block * stride_scale_k)
        w = w * scale

        acc += tl.dot(a, w, input_precision=INPUT_PRECISION)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def fp8_gemm_contig_kernel_correction(
    a_ptr,
    w_ptr,
    w_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    stride_scale_n,
    stride_scale_k,
    INPUT_PRECISION: tl.constexpr,
    CORRECT_K_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    k_base = K - CORRECT_K_BLOCKS * BLOCK_K
    for kk in range(0, CORRECT_K_BLOCKS * BLOCK_K, BLOCK_K):
        k_ids = k_base + kk + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        a = tl.cast(a, tl.float32)

        w_ptrs = w_ptr + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w = tl.cast(w, tl.float32)

        n_block = (pid_n * BLOCK_N) // SCALE_BLOCK
        k_block = (k_base + kk) // SCALE_BLOCK
        scale = tl.load(w_scale_ptr + n_block * stride_scale_n + k_block * stride_scale_k)
        w = w * scale

        corr = tl.dot(a, w, input_precision="ieee") - tl.dot(a, w, input_precision=INPUT_PRECISION)
        acc += corr

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    c += acc
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def swiglu_kernel(
    g_ptr,
    c_ptr,
    M,
    stride_gm,
    stride_gn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x1_ptrs = g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    x2_ptrs = g_ptr + offs_m[:, None] * stride_gm + (offs_n[None, :] + INTERMEDIATE_SIZE) * stride_gn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < INTERMEDIATE_SIZE)
    x1 = tl.load(x1_ptrs, mask=mask, other=0.0)
    x2 = tl.load(x2_ptrs, mask=mask, other=0.0)
    silu = x2 / (1.0 + tl.exp(-x2))
    c = silu * x1
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c, mask=mask)


@triton.jit
def accumulate_kernel(
    o_ptr,
    token_idx_ptr,
    token_weight_ptr,
    out_ptr,
    M,
    N,
    stride_om,
    stride_on,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    token_ids = tl.load(token_idx_ptr + offs_m, mask=offs_m < M, other=0)
    weights = tl.load(token_weight_ptr + offs_m, mask=offs_m < M, other=0.0)

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    o = tl.load(o_ptrs, mask=mask, other=0.0)
    o = o * weights[:, None]

    out_ptrs = out_ptr + token_ids[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    # Each token appears at most once per expert and experts are processed sequentially.
    # Plain add+store is safe and avoids atomic accumulation issues.
    out = tl.load(out_ptrs, mask=mask, other=0.0)
    out += o
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def fused_gemm1_swiglu_kernel(
    a_ptr,
    w_ptr,
    w_scale_ptr,
    token_idx_ptr,
    c_ptr,
    M,
    N,  # INTERMEDIATE_SIZE
    K,  # HIDDEN_SIZE
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    stride_scale_n,
    stride_scale_k,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    """Fused GEMM1 + SwiGLU: computes both gate and up projections then applies silu(up) * gate."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    token_ids = tl.load(token_idx_ptr + offs_m, mask=offs_m < M, other=0)

    # Accumulate gate (x1) from W13 rows [0, N) and up (x2) from W13 rows [N, 2*N)
    acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        # Load activations
        a_ptrs = a_ptr + token_ids[:, None] * stride_am + k_ids[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        a = tl.cast(a, tl.float32)

        k_block = k // SCALE_BLOCK

        # Gate weights (rows [offs_n])
        n_block_gate = (pid_n * BLOCK_N) // SCALE_BLOCK
        w_gate_ptrs = w_ptr + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w_gate = tl.load(w_gate_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w_gate = tl.cast(w_gate, tl.float32)
        scale_gate = tl.load(w_scale_ptr + n_block_gate * stride_scale_n + k_block * stride_scale_k)
        w_gate = w_gate * scale_gate
        acc_gate += tl.dot(a, w_gate, input_precision=INPUT_PRECISION)

        # Up weights (rows [offs_n + N])
        n_block_up = ((pid_n * BLOCK_N) + N) // SCALE_BLOCK
        w_up_ptrs = w_ptr + k_ids[:, None] * stride_wk + (offs_n[None, :] + N) * stride_wm
        w_up = tl.load(w_up_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w_up = tl.cast(w_up, tl.float32)
        scale_up = tl.load(w_scale_ptr + n_block_up * stride_scale_n + k_block * stride_scale_k)
        w_up = w_up * scale_up
        acc_up += tl.dot(a, w_up, input_precision=INPUT_PRECISION)

    # SwiGLU: silu(up) * gate
    silu_up = acc_up / (1.0 + tl.exp(-acc_up))
    result = silu_up * acc_gate

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, result, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def fused_gemm2_accumulate_kernel(
    c_ptr,
    w_ptr,
    w_scale_ptr,
    token_idx_ptr,
    token_weight_ptr,
    out_ptr,
    M,
    N,  # HIDDEN_SIZE
    K,  # INTERMEDIATE_SIZE
    stride_cm,
    stride_ck,
    stride_wm,
    stride_wk,
    stride_out_m,
    stride_out_n,
    stride_scale_n,
    stride_scale_k,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    """Fused GEMM2 + weighted scatter-add to output."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    token_ids = tl.load(token_idx_ptr + offs_m, mask=offs_m < M, other=0)
    weights = tl.load(token_weight_ptr + offs_m, mask=offs_m < M, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + k_ids[None, :] * stride_ck
        c = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        c = tl.cast(c, tl.float32)

        w_ptrs = w_ptr + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w = tl.cast(w, tl.float32)

        n_block = (pid_n * BLOCK_N) // SCALE_BLOCK
        k_block = k // SCALE_BLOCK
        scale = tl.load(w_scale_ptr + n_block * stride_scale_n + k_block * stride_scale_k)
        w = w * scale

        acc += tl.dot(c, w, input_precision=INPUT_PRECISION)

    # Multiply by token weight and scatter-add to output
    acc = acc * weights[:, None]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + token_ids[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    out = tl.load(out_ptrs, mask=mask, other=0.0)
    out += acc
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def grouped_fused_dequant_gemm1_swiglu_kernel(
    hidden_ptr,
    hidden_scale_ptr,
    w_ptr,
    w_scale_ptr,
    token_idx_ptr,
    c_ptr,
    expert_offsets_ptr,
    m_tile_offsets_ptr,
    N,  # INTERMEDIATE_SIZE (output is N, not 2*N)
    K,  # HIDDEN_SIZE
    stride_hidden_m,
    stride_hidden_k,
    stride_hscale_b,  # hidden_states_scale stride for block dim
    stride_hscale_t,  # hidden_states_scale stride for token dim
    expert_stride_w,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    expert_stride_scale,
    stride_scale_n,
    stride_scale_k,
    NUM_EXPERTS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    """Fused: FP8 dequant + GEMM1 (gate & up) + SwiGLU. Writes INTERMEDIATE_SIZE output."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Binary search for expert_id from pid_m
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(0, 8):
        mid = (lo + hi) // 2
        mid_val = tl.load(m_tile_offsets_ptr + mid)
        lo = tl.where(mid_val <= pid_m, mid, lo)
        hi = tl.where(mid_val <= pid_m, hi, mid)
    expert_id = lo

    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
    M = expert_end - expert_start
    tile_start = tl.load(m_tile_offsets_ptr + expert_id)
    local_pid_m = pid_m - tile_start

    offs_m = local_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    token_ids = tl.load(token_idx_ptr + expert_start + offs_m, mask=offs_m < M, other=0)

    acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    w_base = w_ptr + expert_id * expert_stride_w
    wscale_base = w_scale_ptr + expert_id * expert_stride_scale

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        # Inline dequant: load FP8 hidden states and scale
        a_ptrs = hidden_ptr + token_ids[:, None] * stride_hidden_m + k_ids[None, :] * stride_hidden_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        a = tl.cast(a, tl.float32)
        # hidden_states_scale is [num_blocks, seq_len], k_block indexes the block dim
        h_scale_block = k // SCALE_BLOCK
        h_scale = tl.load(hidden_scale_ptr + h_scale_block * stride_hscale_b + token_ids * stride_hscale_t,
                          mask=offs_m < M, other=1.0)
        a = a * h_scale[:, None]

        k_block = k // SCALE_BLOCK

        # Gate weights (rows [offs_n])
        n_block_gate = (pid_n * BLOCK_N) // SCALE_BLOCK
        w_gate_ptrs = w_base + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w_gate = tl.load(w_gate_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w_gate = tl.cast(w_gate, tl.float32)
        scale_gate = tl.load(wscale_base + n_block_gate * stride_scale_n + k_block * stride_scale_k)
        w_gate = w_gate * scale_gate
        acc_gate += tl.dot(a, w_gate, input_precision=INPUT_PRECISION)

        # Up weights (rows [offs_n + N])
        n_block_up = ((pid_n * BLOCK_N) + N) // SCALE_BLOCK
        w_up_ptrs = w_base + k_ids[:, None] * stride_wk + (offs_n[None, :] + N) * stride_wm
        w_up = tl.load(w_up_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w_up = tl.cast(w_up, tl.float32)
        scale_up = tl.load(wscale_base + n_block_up * stride_scale_n + k_block * stride_scale_k)
        w_up = w_up * scale_up
        acc_up += tl.dot(a, w_up, input_precision=INPUT_PRECISION)

    # SwiGLU: silu(up) * gate
    silu_up = acc_up / (1.0 + tl.exp(-acc_up))
    result = silu_up * acc_gate

    c_ptrs = c_ptr + (expert_start + offs_m[:, None]) * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, result, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def grouped_fp8_gemm1_kernel(
    a_ptr,
    w_ptr,
    w_scale_ptr,
    token_idx_ptr,
    c_ptr,
    expert_offsets_ptr,
    m_tile_offsets_ptr,
    N,
    K,
    stride_am,
    stride_ak,
    expert_stride_w,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    expert_stride_scale,
    stride_scale_n,
    stride_scale_k,
    total_tokens,
    NUM_EXPERTS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Binary search for expert_id from pid_m
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(0, 8):  # log2(256) = 8, more than enough for 32
        mid = (lo + hi) // 2
        mid_val = tl.load(m_tile_offsets_ptr + mid)
        lo = tl.where(mid_val <= pid_m, mid, lo)
        hi = tl.where(mid_val <= pid_m, hi, mid)
    expert_id = lo

    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
    M = expert_end - expert_start
    tile_start = tl.load(m_tile_offsets_ptr + expert_id)
    local_pid_m = pid_m - tile_start

    offs_m = local_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    token_ids = tl.load(token_idx_ptr + expert_start + offs_m, mask=offs_m < M, other=0)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    w_base = w_ptr + expert_id * expert_stride_w
    scale_base = w_scale_ptr + expert_id * expert_stride_scale

    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        a_ptrs = a_ptr + token_ids[:, None] * stride_am + k_ids[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        a = tl.cast(a, tl.float32)

        w_ptrs = w_base + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w = tl.cast(w, tl.float32)

        n_block = (pid_n * BLOCK_N) // SCALE_BLOCK
        k_block = k // SCALE_BLOCK
        scale = tl.load(scale_base + n_block * stride_scale_n + k_block * stride_scale_k)
        w = w * scale

        acc += tl.dot(a, w, input_precision=INPUT_PRECISION)

    c_ptrs = c_ptr + (expert_start + offs_m[:, None]) * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def grouped_swiglu_kernel(
    g_ptr,
    c_ptr,
    expert_offsets_ptr,
    m_tile_offsets_ptr,
    stride_gm,
    stride_gn,
    stride_cm,
    stride_cn,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INTERMEDIATE_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Binary search for expert_id
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(0, 8):
        mid = (lo + hi) // 2
        mid_val = tl.load(m_tile_offsets_ptr + mid)
        lo = tl.where(mid_val <= pid_m, mid, lo)
        hi = tl.where(mid_val <= pid_m, hi, mid)
    expert_id = lo

    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
    M = expert_end - expert_start
    tile_start = tl.load(m_tile_offsets_ptr + expert_id)
    local_pid_m = pid_m - tile_start

    offs_m = local_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    global_m = expert_start + offs_m
    x1_ptrs = g_ptr + global_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    x2_ptrs = g_ptr + global_m[:, None] * stride_gm + (offs_n[None, :] + INTERMEDIATE_SIZE) * stride_gn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < INTERMEDIATE_SIZE)
    x1 = tl.load(x1_ptrs, mask=mask, other=0.0)
    x2 = tl.load(x2_ptrs, mask=mask, other=0.0)
    silu = x2 / (1.0 + tl.exp(-x2))
    c = silu * x1
    tl.store(c_ptr + global_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c, mask=mask)


@triton.jit
def grouped_fused_gemm2_acc_kernel(
    c_ptr,
    w_ptr,
    w_scale_ptr,
    token_idx_ptr,
    token_weight_ptr,
    out_ptr,
    expert_offsets_ptr,
    m_tile_offsets_ptr,
    N,  # HIDDEN_SIZE
    K,  # INTERMEDIATE_SIZE
    stride_cm,
    stride_ck,
    expert_stride_w,
    stride_wm,
    stride_wk,
    stride_out_m,
    stride_out_n,
    expert_stride_scale,
    stride_scale_n,
    stride_scale_k,
    NUM_EXPERTS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Binary search for expert_id
    lo = 0
    hi = NUM_EXPERTS
    for _ in tl.static_range(0, 8):
        mid = (lo + hi) // 2
        mid_val = tl.load(m_tile_offsets_ptr + mid)
        lo = tl.where(mid_val <= pid_m, mid, lo)
        hi = tl.where(mid_val <= pid_m, hi, mid)
    expert_id = lo

    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
    M = expert_end - expert_start
    tile_start = tl.load(m_tile_offsets_ptr + expert_id)
    local_pid_m = pid_m - tile_start

    offs_m = local_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    token_ids = tl.load(token_idx_ptr + expert_start + offs_m, mask=offs_m < M, other=0)
    weights = tl.load(token_weight_ptr + expert_start + offs_m, mask=offs_m < M, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    w_base = w_ptr + expert_id * expert_stride_w
    scale_base = w_scale_ptr + expert_id * expert_stride_scale

    global_m = expert_start + offs_m
    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        c_ptrs = c_ptr + global_m[:, None] * stride_cm + k_ids[None, :] * stride_ck
        c = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0)
        c = tl.cast(c, tl.float32)

        w_ptrs = w_base + k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wm
        w = tl.load(w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        w = tl.cast(w, tl.float32)

        n_block = (pid_n * BLOCK_N) // SCALE_BLOCK
        k_block = k // SCALE_BLOCK
        scale = tl.load(scale_base + n_block * stride_scale_n + k_block * stride_scale_k)
        w = w * scale

        acc += tl.dot(c, w, input_precision=INPUT_PRECISION)

    # Multiply by token weight and atomic scatter-add to output (multiple experts may write same token)
    acc = acc * weights[:, None]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + token_ids[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.atomic_add(out_ptrs, acc, mask=mask)


def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor,
) -> None:
    device = hidden_states.device
    seq_len = int(routing_logits.shape[0])
    num_hidden_blocks = HIDDEN_SIZE // BLOCK
    num_intermediate_blocks = INTERMEDIATE_SIZE // BLOCK
    num_gemm1_out_blocks = (2 * INTERMEDIATE_SIZE) // BLOCK
    prev_precision = None
    if USE_TORCH_GEMM and TORCH_MATMUL_PRECISION:
        prev_precision = torch.get_float32_matmul_precision()
        if prev_precision != TORCH_MATMUL_PRECISION:
            torch.set_float32_matmul_precision(TORCH_MATMUL_PRECISION)

    # Routing: compute topk indices and weights.
    weights_full = None
    if USE_TORCH_ROUTING:
        logits = routing_logits.to(torch.float32)
        bias = routing_bias.to(torch.float32).view(1, NUM_EXPERTS)
        s = torch.sigmoid(logits)
        s_with_bias = s + bias

        group_size = NUM_EXPERTS // N_GROUP
        s_grouped = s_with_bias.view(seq_len, N_GROUP, group_size)
        top2_vals, _ = torch.topk(s_grouped, k=2, dim=2, largest=True, sorted=False)
        group_scores = top2_vals.sum(dim=2)
        _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)

        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = group_mask.unsqueeze(2).expand(seq_len, N_GROUP, group_size).reshape(seq_len, NUM_EXPERTS)
        scores_pruned = s_with_bias.masked_fill(score_mask == 0, torch.finfo(torch.float32).min)
        _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

        m = torch.zeros_like(s)
        m.scatter_(1, topk_idx, 1.0)
        weights = s * m
        weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * float(routed_scaling_factor)
        weights_full = weights
        topk_weight = torch.gather(weights, 1, topk_idx)

        topk_idx = topk_idx.to(torch.int32)
        topk_weight = topk_weight.to(torch.float32)
    else:
        topk_idx = torch.empty((seq_len, TOP_K), device=device, dtype=torch.int32)
        topk_weight = torch.empty((seq_len, TOP_K), device=device, dtype=torch.float32)

        routing_grid = (seq_len,)
        routing_kernel[routing_grid](
            routing_logits,
            routing_bias,
            topk_idx,
            topk_weight,
            routing_logits.stride(0),
            topk_idx.stride(0),
            seq_len,
            float(routed_scaling_factor),
            NUM_EXPERTS=NUM_EXPERTS,
            TOP_K=TOP_K,
            N_GROUP=N_GROUP,
            TOPK_GROUP=TOPK_GROUP,
            num_warps=8,
            num_stages=2,
        )

    if not USE_TORCH_COMPACTION:
        # Expert counts and compaction.
        expert_counts = torch.zeros((NUM_LOCAL_EXPERTS,), device=device, dtype=torch.int32)
        count_kernel[(seq_len,)](
            topk_idx,
            expert_counts,
            topk_idx.stride(0),
            seq_len,
            int(local_expert_offset),
            TOP_K=TOP_K,
            NUM_LOCAL_EXPERTS=NUM_LOCAL_EXPERTS,
            num_warps=4,
            num_stages=1,
        )

        expert_offsets = torch.zeros((NUM_LOCAL_EXPERTS + 1,), device=device, dtype=torch.int32)
        expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)
        total = int(expert_offsets[-1].item())

        token_indices = torch.empty((max(total, 1),), device=device, dtype=torch.int32)
        token_weights = torch.empty((max(total, 1),), device=device, dtype=torch.float32)
        expert_positions = torch.zeros((NUM_LOCAL_EXPERTS,), device=device, dtype=torch.int32)

        if total > 0:
            scatter_kernel[(seq_len,)](
                topk_idx,
                topk_weight,
                token_indices,
                token_weights,
                expert_offsets,
                expert_positions,
                topk_idx.stride(0),
                seq_len,
                int(local_expert_offset),
                TOP_K=TOP_K,
                NUM_LOCAL_EXPERTS=NUM_LOCAL_EXPERTS,
                num_warps=4,
                num_stages=1,
            )

    # Dequantize hidden states (only needed for torch fallback paths).
    a = None
    if USE_TORCH_COMPACTION or USE_TORCH_GEMM:
        if USE_TORCH_DEQUANT:
            num_hidden_blocks = HIDDEN_SIZE // BLOCK
            a = (
                hidden_states.to(torch.float32)
                .view(seq_len, num_hidden_blocks, BLOCK)
                .mul(hidden_states_scale.to(torch.float32).permute(1, 0).unsqueeze(-1))
                .reshape(seq_len, HIDDEN_SIZE)
            )
            if not (USE_TORCH_GEMM or USE_TORCH_COMPACTION or USE_FP32_GEMM or KEEP_A_FP32):
                a = a.to(torch.bfloat16)
        else:
            a = torch.empty((seq_len, HIDDEN_SIZE), device=device, dtype=torch.bfloat16)
            grid = (seq_len, HIDDEN_SIZE // BLOCK)
            dequant_hidden_kernel[grid](
                hidden_states,
                hidden_states_scale,
                a,
                hidden_states.stride(0),
                hidden_states.stride(1),
                hidden_states_scale.stride(0),
                hidden_states_scale.stride(1),
                a.stride(0),
                a.stride(1),
                seq_len,
                BLOCK=BLOCK,
                HIDDEN_SIZE=HIDDEN_SIZE,
                num_warps=4,
                num_stages=2,
            )

    output_fp32 = torch.zeros((seq_len, HIDDEN_SIZE), device=device, dtype=torch.float32)

    if USE_TORCH_COMPACTION:
        local_start = int(local_expert_offset)
        for local_expert in range(NUM_LOCAL_EXPERTS):
            global_expert_id = local_start + local_expert
            if global_expert_id < 0 or global_expert_id >= NUM_EXPERTS:
                continue

            token_mask = (topk_idx == global_expert_id).any(dim=1)
            if not token_mask.any():
                continue

            token_idx = torch.nonzero(token_mask, as_tuple=False).squeeze(1)
            token_weight = weights_full.index_select(0, token_idx)[:, global_expert_id]
            a_expert = a.index_select(0, token_idx).to(torch.float32)

            w13 = _block_dequant_matrix(
                gemm1_weights[local_expert],
                gemm1_weights_scale[local_expert],
                num_gemm1_out_blocks,
                num_hidden_blocks,
                BLOCK,
            )
            w2 = _block_dequant_matrix(
                gemm2_weights[local_expert],
                gemm2_weights_scale[local_expert],
                num_hidden_blocks,
                num_intermediate_blocks,
                BLOCK,
            )

            g1 = a_expert.matmul(w13.t())
            x1 = g1[:, :INTERMEDIATE_SIZE]
            x2 = g1[:, INTERMEDIATE_SIZE:]
            c = torch.nn.functional.silu(x2) * x1
            o = c.matmul(w2.t())
            output_fp32.index_add_(0, token_idx, o * token_weight.unsqueeze(1))
    elif total > 0:
        bm, bn, bk = BLOCK_M, BLOCK_N, BLOCK_K

        if USE_TORCH_GEMM:
            # Per expert fallback: GEMM1 -> SwiGLU -> GEMM2 -> accumulate
            max_m = max(int((expert_offsets[i+1] - expert_offsets[i]).item()) for i in range(NUM_LOCAL_EXPERTS))
            if max_m > 0:
                g1_buf = torch.empty((max_m, 2 * INTERMEDIATE_SIZE), device=device, dtype=torch.float32)
                c_buf = torch.empty((max_m, INTERMEDIATE_SIZE), device=device, dtype=torch.float32)

            for local_expert in range(NUM_LOCAL_EXPERTS):
                start = int(expert_offsets[local_expert].item())
                end = int(expert_offsets[local_expert + 1].item())
                if start == end:
                    continue

                token_idx_slice = token_indices[start:end]
                token_weight_slice = token_weights[start:end]
                m = end - start

                a_expert = a.index_select(0, token_idx_slice).to(torch.float32)
                w13 = _block_dequant_matrix(
                    gemm1_weights[local_expert],
                    gemm1_weights_scale[local_expert],
                    num_gemm1_out_blocks,
                    num_hidden_blocks,
                    BLOCK,
                )
                w2 = _block_dequant_matrix(
                    gemm2_weights[local_expert],
                    gemm2_weights_scale[local_expert],
                    num_hidden_blocks,
                    num_intermediate_blocks,
                    BLOCK,
                )

                g1 = a_expert.matmul(w13.t())
                x1 = g1[:, :INTERMEDIATE_SIZE]
                x2 = g1[:, INTERMEDIATE_SIZE:]
                c = torch.nn.functional.silu(x2) * x1
                o = c.matmul(w2.t())
                output_fp32.index_add_(0, token_idx_slice, o * token_weight_slice.unsqueeze(1))
        else:
            # Grouped kernel path: 3 launches instead of 96
            # Compute m_tile_offsets on CPU for binary search in kernels
            expert_offsets_cpu = expert_offsets.cpu()
            m_tile_counts = torch.zeros(NUM_LOCAL_EXPERTS, dtype=torch.int32)
            for i in range(NUM_LOCAL_EXPERTS):
                count = int(expert_offsets_cpu[i + 1].item()) - int(expert_offsets_cpu[i].item())
                m_tile_counts[i] = triton.cdiv(count, bm) if count > 0 else 0
            m_tile_offsets = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32)
            m_tile_offsets[1:] = torch.cumsum(m_tile_counts, dim=0)
            total_m_tiles = int(m_tile_offsets[-1].item())

            if total_m_tiles > 0:
                m_tile_offsets_dev = m_tile_offsets.to(device)

                # Allocate scratch buffer for SwiGLU output only (no g1_buf needed)
                c_buf = torch.empty((total, INTERMEDIATE_SIZE), device=device, dtype=torch.float32)

                # Fused dequant + GEMM1 + SwiGLU: all experts in one launch
                grid_gemm1 = (total_m_tiles, triton.cdiv(INTERMEDIATE_SIZE, bn))
                grouped_fused_dequant_gemm1_swiglu_kernel[grid_gemm1](
                    hidden_states,
                    hidden_states_scale,
                    gemm1_weights,
                    gemm1_weights_scale,
                    token_indices,
                    c_buf,
                    expert_offsets,
                    m_tile_offsets_dev,
                    INTERMEDIATE_SIZE,
                    HIDDEN_SIZE,
                    hidden_states.stride(0),
                    hidden_states.stride(1),
                    hidden_states_scale.stride(0),
                    hidden_states_scale.stride(1),
                    gemm1_weights.stride(0),
                    gemm1_weights.stride(1),
                    gemm1_weights.stride(2),
                    c_buf.stride(0),
                    c_buf.stride(1),
                    gemm1_weights_scale.stride(0),
                    gemm1_weights_scale.stride(1),
                    gemm1_weights_scale.stride(2),
                    NUM_EXPERTS=NUM_LOCAL_EXPERTS,
                    INPUT_PRECISION=GEMM1_INPUT_PRECISION,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=bk,
                    SCALE_BLOCK=BLOCK,
                    num_warps=8,
                    num_stages=3,
                )

                # Grouped fused GEMM2 + accumulate: all experts in one launch
                grid_gemm2 = (total_m_tiles, triton.cdiv(HIDDEN_SIZE, bn))
                grouped_fused_gemm2_acc_kernel[grid_gemm2](
                    c_buf,
                    gemm2_weights,
                    gemm2_weights_scale,
                    token_indices,
                    token_weights,
                    output_fp32,
                    expert_offsets,
                    m_tile_offsets_dev,
                    HIDDEN_SIZE,
                    INTERMEDIATE_SIZE,
                    c_buf.stride(0),
                    c_buf.stride(1),
                    gemm2_weights.stride(0),
                    gemm2_weights.stride(1),
                    gemm2_weights.stride(2),
                    output_fp32.stride(0),
                    output_fp32.stride(1),
                    gemm2_weights_scale.stride(0),
                    gemm2_weights_scale.stride(1),
                    gemm2_weights_scale.stride(2),
                    NUM_EXPERTS=NUM_LOCAL_EXPERTS,
                    INPUT_PRECISION=GEMM2_INPUT_PRECISION,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=bk,
                    SCALE_BLOCK=BLOCK,
                    num_warps=8,
                    num_stages=2,
                )

    output.copy_(output_fp32.to(torch.bfloat16))
