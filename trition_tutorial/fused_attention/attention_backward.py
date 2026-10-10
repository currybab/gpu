"""LSE를 이용한 backward 실습. atomic 없이 dQ와 dK/dV를 분리한다."""

import torch
import triton
import triton.language as tl

from attention_variants import _launch_raw, validate_inputs


@triton.jit
def _bwd_preprocess(O, DO, DELTA, N_CTX, HEAD_DIM: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    head = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, BLOCK_D)
    offsets = (head * N_CTX + rows[:, None]) * HEAD_DIM + dims[None, :]
    # TODO BWD 1: O/dO를 mask load하고 FP32로 변환.
    # DELTA[head * N_CTX + rows] = sum(O * dO, axis=D).
    # Delta는 FP32 [B,H,N]. row/dimension tail과 store mask에 주의.
    _ = offsets


@triton.jit
def _bwd_dq(Q, K, V, DO, LSE, DELTA, DQ, N_CTX, sm_scale,
            HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    head = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, BLOCK_D)
    head_offset = head * N_CTX * HEAD_DIM
    dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    # TODO BWD 2: query tile Q/dO, row별 LSE/DELTA를 loop 밖에서 load.
    for start_n in tl.range(0, N_CTX, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        # TODO BWD 3: K/V tile load, P = exp2(QK^T * scale * log2(e) - LSE).
        # invalid row/key와 causal 미래 위치의 P는 0이어야 한다.
        # tail LSE load에는 유한한 other=0을 사용한다.
        # TODO BWD 4: dP = dO @ V^T, dS = P * (dP - DELTA[:, None]).
        # dq += (dS @ K) * sm_scale. log2(e)를 gradient에 다시 곱하지 않는다.
        # dot 입력은 FP16으로, 누적과 softmax 미분은 FP32로 처리한다.
        _ = cols
    # TODO BWD 5: DQ tile을 row/dimension mask와 함께 저장.
    _ = head_offset, rows, dims, dq


@triton.jit
def _bwd_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV, N_CTX, sm_scale,
              HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    head = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    head_offset = head * N_CTX * HEAD_DIM
    dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    # TODO BWD 6: 현재 key tile의 K/V를 loop 밖에서 load.
    # program 하나가 한 key tile을 소유하므로 atomic_add가 필요 없다.
    for start_m in tl.range(0, N_CTX, BLOCK_M):
        rows = start_m + tl.arange(0, BLOCK_M)
        # TODO BWD 7: Q/dO/LSE/DELTA load; BWD 3~4와 같은 P/dS를 재계산.
        # dv += P^T @ dO; dk += (dS^T @ Q) * sm_scale.
        # causal 비교는 여기서도 query row >= key col이다.
        _ = rows
    # TODO BWD 8: DK/DV tile을 key/dimension mask와 함께 저장.
    _ = head_offset, cols, dims, dk, dv


class _AttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        validate_inputs(q, k, v)
        # TODO BWD 0: attention_variants의 SPLIT + LSE TODO를 먼저 완성.
        output, lse = _launch_raw(q, k, v, causal, sm_scale, save_lse=True)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors
        # autograd가 전달하는 dO는 non-contiguous일 수 있다.
        do = grad_output.contiguous()
        batch, heads, n, d = q.shape
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        delta = torch.empty_like(lse)
        block_m, block_n = 32, 32
        block_d = triton.next_power_of_2(d)
        grid_q = (triton.cdiv(n, block_m), batch * heads)
        grid_k = (triton.cdiv(n, block_n), batch * heads)
        _bwd_preprocess[grid_q](output, do, delta, n, HEAD_DIM=d,
                               BLOCK_M=block_m, BLOCK_D=block_d)
        meta = dict(HEAD_DIM=d, CAUSAL=ctx.causal, BLOCK_M=block_m,
                    BLOCK_N=block_n, BLOCK_D=block_d, num_warps=4, num_stages=2)
        _bwd_dq[grid_q](q, k, v, do, lse, delta, dq, n, ctx.sm_scale, **meta)
        _bwd_dkdv[grid_k](q, k, v, do, lse, delta, dk, dv, n, ctx.sm_scale, **meta)
        return dq, dk, dv, None, None


def attention_train(q, k, v, causal, sm_scale):
    """FP16 forward + 1차 backward 실습. double backward는 지원하지 않는다."""
    raise NotImplementedError("SPLIT/LSE 및 BWD 0~8 구현 후 이 raise 삭제")
    return _AttentionAutograd.apply(q, k, v, causal, sm_scale)
