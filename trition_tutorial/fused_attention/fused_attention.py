"""contiguous FP16 fused-attention forward 실습 뼈대."""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_fwd(
    Q,
    K,
    V,
    O,
    N,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """program 하나가 한 head의 BLOCK_M개 query row를 계산한다."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    head_offset = batch_head * N * HEAD_DIM

    # TODO 1: Q[offs_m, offs_d]를 mask와 함께 한 번 load한다.
    # q_ptrs = Q + head_offset + offs_m[:, None] * HEAD_DIM + offs_d[None, :]

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    for start_n in tl.range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # TODO 2: K[offs_n, offs_d]를 tail mask와 함께 load한다.
        # TODO 3: qk = tl.dot(q, tl.trans(k)) * qk_scale
        # TODO 4: key tail과 CAUSAL mask를 qk에 적용한다.
        # TODO 5: m_i/l_i/acc를 online softmax 식으로 갱신한다.
        # TODO 6: V[offs_n, offs_d]를 load하고 P @ V를 acc에 누적한다.
        _ = offs_n

    # TODO 7: acc / l_i를 O에 query/head-dimension tail mask로 저장한다.


def attention(q, k, v, causal: bool, sm_scale: float):
    """Triton fused-attention 구현 진입점."""
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    batch, n_heads, n_ctx, head_dim = q.shape
    assert 16 <= head_dim <= 128

    output = torch.empty_like(q)
    block_m = 64
    block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(n_ctx, block_m), batch * n_heads)

    # TODO 8: _attention_fwd[grid](...)를 고정 config로 launch한다.
    _ = output, grid, block_n, block_d, causal, sm_scale
    raise NotImplementedError("_attention_fwd의 TODO 1~8을 순서대로 구현하세요.")
