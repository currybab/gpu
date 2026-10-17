"""FA2 방식의 query tile 중심 FP16 forward 실습 뼈대.

program 하나가 Q_i를 맡고 K_j/V_j를 순회한다. acc는 정규화 전 출력이며,
마지막에만 l_i로 나눈다. 글의 수식과 변수 대응은 README를 참고한다.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    N_CTX,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """program 하나가 한 head의 BLOCK_M개 query row를 계산한다."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // N_HEADS
    head_idx = batch_head % N_HEADS

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_head_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_head_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_head_offset = batch_idx * stride_vb + head_idx * stride_vh
    o_head_offset = batch_idx * stride_ob + head_idx * stride_oh

    # TODO 1: Q[offs_m, offs_d]를 mask와 함께 한 번 load한다.
    q_ptrs = q_ptr + q_head_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 글의 m_i, ℓ_i, Õ_i에 대응. exp2를 쓰므로 m_i는 base-2 score 기준.
    # 이 상태는 K/V loop 전체에 걸쳐 유지한다.
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    end_n = N_CTX
    if CAUSAL:
        # 현재 query tile보다 완전히 미래인 K/V block은 계산하지 않는다.
        end_n = tl.minimum(N_CTX, (query_block + 1) * BLOCK_M)
    for start_n in tl.range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # TODO 2: k_ptr/k_head_offset/stride_kn/stride_kd로 K tile을 load한다.
        k_ptrs = k_ptr + k_head_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # TODO 3: qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.dot(q, k.T) * qk_scale

        # TODO 4: key tail과 CAUSAL mask를 qk에 적용한다.
        score_mask = offs_n[None, :] < N_CTX
        if CAUSAL:
            score_mask = score_mask & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(score_mask, qk, float("-inf"))

        # TODO 5: m_i/l_i/acc를 online softmax 식으로 갱신한다.
        # m_new = max(m_i, rowmax(qk)), alpha = exp2(m_i - m_new).
        # p = exp2(qk - m_new), l_i = alpha * l_i + rowsum(p).
        # acc에도 alpha를 곱하고 m_i를 m_new로 갱신한다.
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc
        m_i = m_new

        # TODO 6: v_ptr와 V stride로 tile을 load하고 P @ V를 acc에 누적한다.
        # 여기의 P는 정규화 전 p다. acc += p @ v; 나눗셈은 loop 뒤에서만.
        v_ptrs = v_ptr + v_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        acc = acc + tl.dot(p.to(v.dtype), v)

    # TODO 7: o_ptr/o_head_offset와 O stride로 acc / l_i를 저장한다.
    o_ptrs = o_ptr + o_head_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
    tl.store(o_ptrs, acc / l_i[:, None], mask=o_mask)


def prepare_fa2(q, k, v, causal: bool, sm_scale: float, *,
                block_m=64, block_n=64, num_warps=4, num_stages=2):
    """출력을 미리 할당하고 kernel만 재실행하는 callable을 반환한다."""
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape == k.shape == v.shape
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert q.device == k.device == v.device
    batch, n_heads, n_ctx, head_dim = q.shape
    assert min(q.shape) > 0 and 16 <= head_dim <= 128
    assert block_m in (32, 64, 128) and block_n in (32, 64, 128)
    assert num_warps in (4, 8) and num_stages in (1, 2, 3, 4)

    output = torch.empty_like(q)
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(n_ctx, block_m), batch * n_heads)

    def run():
        # TODO 8: stride와 meta-parameter의 대응을 확인한다.
        run.kernel = _attention_fwd[grid](
            q, k, v, output, n_ctx, sm_scale,
            *q.stride(), *k.stride(), *v.stride(), *output.stride(),
            N_HEADS=n_heads, HEAD_DIM=head_dim, CAUSAL=causal,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
            num_warps=num_warps, num_stages=num_stages)
        return output

    return run


def attention(q, k, v, causal: bool, sm_scale: float, **config):
    """FA2 방식의 forward. baseline과 fa2가 공유하는 구현."""
    return prepare_fa2(q, k, v, causal, sm_scale, **config)()
