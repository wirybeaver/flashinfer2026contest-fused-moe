"""
Track A (fused_moe) Triton implementation.

Entry point: kernel(...)
"""

from __future__ import annotations

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
GEMM1_INPUT_PRECISION = "tf32"
GEMM2_INPUT_PRECISION = "tf32"
GEMM1_BLOCK_M = 128
GEMM1_BLOCK_N = 128
GEMM1_BLOCK_K = 64
GEMM2_BLOCK_M = 128
GEMM2_BLOCK_N = 128
GEMM2_BLOCK_K = 64


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

    # Cast to fp32 — tl.exp requires fp32+, and matches PyTorch's .to(float32)
    logits = tl.cast(logits, tl.float32)
    bias = tl.cast(bias, tl.float32)

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
        best_idx = tl.min(tl.where(vals == max1, offs, NUM_EXPERTS), axis=0)
        vals2 = tl.where(offs == best_idx, -float("inf"), vals)
        max2 = tl.max(vals2, axis=0)
        score = max1 + max2
        group_scores = tl.where(group_ids == g, score, group_scores)

    # Select topk_group groups.
    group_keep = tl.zeros([N_GROUP], dtype=tl.int1)
    for _ in tl.static_range(0, TOPK_GROUP):
        masked = tl.where(group_keep, -float("inf"), group_scores)
        best_val = tl.max(masked, axis=0)
        best_idx = tl.min(tl.where(masked == best_val, group_ids, N_GROUP), axis=0)
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
        best_idx = tl.min(tl.where(masked == best_val, offs, NUM_EXPERTS), axis=0)
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
    num_n_tiles,
    total_tiles,
    NUM_EXPERTS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent fused: FP8 dequant + GEMM1 (gate & up) + SwiGLU."""
    start_pid = tl.program_id(0)

    for tile_id in tl.range(start_pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles

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
    num_n_tiles,
    total_tiles,
    NUM_EXPERTS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)

    for tile_id in tl.range(start_pid, total_tiles, NUM_SMS):
        pid_m = tile_id // num_n_tiles
        pid_n = tile_id % num_n_tiles

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

        # Multiply by token weight and atomic scatter-add to output
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
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count

    # Routing: compute topk indices and weights via single Triton kernel.
    topk_idx = torch.empty((seq_len, TOP_K), device=device, dtype=torch.int32)
    topk_weight = torch.empty((seq_len, TOP_K), device=device, dtype=torch.float32)

    routing_kernel[(seq_len,)](
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

    output_fp32 = torch.zeros((seq_len, HIDDEN_SIZE), device=device, dtype=torch.float32)

    if total > 0:
        # Grouped kernel path: 2 launches
        # Compute m_tile_offsets on GPU to avoid CPU-GPU sync
        expert_counts = expert_offsets[1:] - expert_offsets[:-1]
        bm = GEMM1_BLOCK_M
        m_tile_counts = (expert_counts + bm - 1) // bm
        m_tile_offsets = torch.zeros(NUM_LOCAL_EXPERTS + 1, device=device, dtype=torch.int32)
        m_tile_offsets[1:] = torch.cumsum(m_tile_counts, dim=0)
        total_m_tiles = int(m_tile_offsets[-1].item())

        if total_m_tiles > 0:
            # Allocate scratch buffer for SwiGLU output only
            c_buf = torch.empty((total, INTERMEDIATE_SIZE), device=device, dtype=torch.float32)

            # Fused dequant + GEMM1 + SwiGLU: persistent kernel
            num_n_tiles_gemm1 = triton.cdiv(INTERMEDIATE_SIZE, GEMM1_BLOCK_N)
            total_tiles_gemm1 = total_m_tiles * num_n_tiles_gemm1
            grid_gemm1 = (min(NUM_SMS, total_tiles_gemm1),)
            grouped_fused_dequant_gemm1_swiglu_kernel[grid_gemm1](
                hidden_states,
                hidden_states_scale,
                gemm1_weights,
                gemm1_weights_scale,
                token_indices,
                c_buf,
                expert_offsets,
                m_tile_offsets,
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
                num_n_tiles_gemm1,
                total_tiles_gemm1,
                NUM_EXPERTS=NUM_LOCAL_EXPERTS,
                INPUT_PRECISION=GEMM1_INPUT_PRECISION,
                BLOCK_M=GEMM1_BLOCK_M,
                BLOCK_N=GEMM1_BLOCK_N,
                BLOCK_K=GEMM1_BLOCK_K,
                SCALE_BLOCK=BLOCK,
                NUM_SMS=NUM_SMS,
                num_warps=8,
                num_stages=4,
            )

            # Grouped fused GEMM2 + accumulate: persistent kernel
            num_n_tiles_gemm2 = triton.cdiv(HIDDEN_SIZE, GEMM2_BLOCK_N)
            total_tiles_gemm2 = total_m_tiles * num_n_tiles_gemm2
            grid_gemm2 = (min(NUM_SMS, total_tiles_gemm2),)
            grouped_fused_gemm2_acc_kernel[grid_gemm2](
                c_buf,
                gemm2_weights,
                gemm2_weights_scale,
                token_indices,
                token_weights,
                output_fp32,
                expert_offsets,
                m_tile_offsets,
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
                num_n_tiles_gemm2,
                total_tiles_gemm2,
                NUM_EXPERTS=NUM_LOCAL_EXPERTS,
                INPUT_PRECISION=GEMM2_INPUT_PRECISION,
                BLOCK_M=GEMM2_BLOCK_M,
                BLOCK_N=GEMM2_BLOCK_N,
                BLOCK_K=GEMM2_BLOCK_K,
                SCALE_BLOCK=BLOCK,
                NUM_SMS=NUM_SMS,
                num_warps=8,
                num_stages=3,
            )

    output.copy_(output_fp32)
