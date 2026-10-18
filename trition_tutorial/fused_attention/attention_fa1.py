"""FA1 방식의 K/V 바깥 loop와 정규화된 O 갱신을 구현하는 실습.

program 하나가 head 하나를 소유한다. Python에서 K/V마다 kernel을 띄우지 않는다.
공식 CUDA 구현의 warp 배치를 재현하는 코드가 아니라 알고리즘 비교용 뼈대다.
"""

import torch
import triton
import triton.language as tl

from attention_variants import validate_inputs


@triton.jit
def _fa1_fwd(Q, K, V, O, O_STATE, M, L, N_CTX, sm_scale,
             HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_D: tl.constexpr):
    head = tl.program_id(0)
    head_offset = head * N_CTX * HEAD_DIM
    dims = tl.arange(0, BLOCK_D)
    qk_scale = sm_scale * 1.4426950408889634 # log2(e)

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        # TODO FA1 1: 현재 K/V tile을 mask load하고 inner Q loop에서 재사용한다.
        # pointer = base + head_offset + cols[:, None]*HEAD_DIM + dims[None, :]
        # K/V tile은 [BLOCK_N, BLOCK_D]. token/dimension tail을 모두 mask한다.
        col_offsets = head_offset + cols[:, None] * HEAD_DIM + dims[None, :]
        col_mask = (cols[:, None] < N_CTX) & (dims[None, :] < HEAD_DIM)
        k_ptrs = K + col_offsets
        v_ptrs = V + col_offsets
        k = tl.load(k_ptrs, mask=col_mask, other=0.0) # [BLOCK_N, BLOCK_D]
        v = tl.load(v_ptrs, mask=col_mask, other=0.0) # [BLOCK_N, BLOCK_D]

        first_m = 0
        if CAUSAL:
            # 완전히 과거인 query tile에는 현재 key block의 기여가 없다.
            first_m = (start_n // BLOCK_M) * BLOCK_M
        for start_m in tl.range(first_m, N_CTX, BLOCK_M):
            rows = start_m + tl.arange(0, BLOCK_M)
            # TODO FA1 2: Q tile을 load. j=0이면 m=-inf, l=0, o=0으로 초기화.
            # j>0이면 M/L/O_STATE를 load. O_STATE는 정규화된 FP32 출력.
            # j=0에서 모든 query 상태를 덮어쓰므로 재실행 전 zero_는 필요 없다.
            stat_offsets = head * N_CTX + rows # M, L: [B, H, N_CTX]이므로 행마다 다른 값을 가짐.
            row_offsets = head_offset + rows[:, None] * HEAD_DIM + dims[None, :] # Q, O_STATE: [B, H, N_CTX, HEAD_DIM]
            row_mask = rows < N_CTX
            tile_mask = row_mask[:, None] & (dims[None, :] < HEAD_DIM)

            q = tl.load(Q + row_offsets, mask=tile_mask, other=0.0) # [BLOCK_M, BLOCK_D]

            if start_n == 0:
                m_old = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32) # [BLOCK_M]
                l_old = tl.zeros((BLOCK_M,), tl.float32) # [BLOCK_M]
                o_old = tl.zeros((BLOCK_M, BLOCK_D), tl.float32) # [BLOCK_M, BLOCK_D]
            else:
                m_old = tl.load(M + stat_offsets, mask=row_mask, other=0.0) # [BLOCK_M]
                l_old = tl.load(L + stat_offsets, mask=row_mask, other=0.0) # [BLOCK_M]
                o_old = tl.load(O_STATE + row_offsets, mask=tile_mask, other=0.0) # [BLOCK_M, BLOCK_D]


            # TODO FA1 3: S=QK^T*qk_scale에 key tail/causal mask를 적용.
            # 블록 내부 max m_block, p_block=exp2(S-m_block), l_block을 계산.
            # causal 대각 영역에는 현재 block의 key가 전부 무효인 row도 있다.
            # 그 row는 exp 계산용 m_block=0, p_block=0, l_block=0으로 NaN을 방지.
            # 최대값을 합칠 때는 무기여 row의 m_block을 -inf로 취급한다.
            s = tl.dot(q, k.T) * qk_scale # [BLOCK_M, BLOCK_N]

            score_mask = cols[None, :] < N_CTX
            if CAUSAL:
                score_mask = score_mask & (rows[:, None] >= cols[None, :])
            s = tl.where(score_mask, s, float("-inf"))

            m_block = tl.max(s, axis=1) # [BLOCK_M]
            has_keys = m_block != -float("inf")

            # 유효한 key가 없는 행은 exp 계산의 기준만 0으로 바꾼다.
            m_for_exp = tl.where(has_keys, m_block, 0.0)
            p_block = tl.exp2(s - m_for_exp[:, None]) # [BLOCK_M, BLOCK_N]
            l_block = tl.sum(p_block, axis=1) # [BLOCK_M]

            # TODO FA1 4: new_m=max(m,m_block), alpha=exp2(m-new_m),
            # beta=exp2(m_block-new_m), new_l=alpha*l+beta*l_block。
            # new_o=(alpha*l*o + beta*(p_block @ v))/new_l[:, None]。
            # p_block은 dot 직전에 FP16으로 변환하고 상태와 누적 계산은 FP32.
            # 기여가 없는 row는 이전 상태를 유지하고 invalid query는 store에서 제외.
            new_m = tl.maximum(m_old, m_block) # [BLOCK_M]
            alpha = tl.exp2(m_old - new_m) # [BLOCK_M]
            beta = tl.exp2(m_block - new_m) # [BLOCK_M]
            new_l = alpha * l_old + beta * l_block # [BLOCK_M]
            pv = tl.dot(p_block.to(v.dtype), v) # [BLOCK_M, BLOCK_D]
            numerator = (alpha * l_old)[:, None] * o_old + beta[:, None] * pv # [BLOCK_M, BLOCK_D]
            new_o = numerator / new_l[:, None] # [BLOCK_M, BLOCK_D]

            # 새 블록의 기여가 없으면 이전 출력을 그대로 유지한다.
            new_o = tl.where(has_keys[:, None], new_o, o_old) # [BLOCK_M, BLOCK_D]

            # TODO FA1 5: 정규화된 new_o와 new_m/new_l을 global memory에 store.
            # O_STATE는 [B,H,N,D], M/L은 [B,H,N]의 contiguous buffer.
            # 같은 head의 의존 관계는 이 program 하나 안에서 완결된다.
            tl.store(O_STATE + row_offsets, new_o, mask=tile_mask)
            tl.store(M + stat_offsets, new_m, mask=row_mask)
            tl.store(L + stat_offsets, new_l, mask=row_mask)
            _ = rows
        _ = cols

    # FA1은 정규화된 O를 FP32로 반복 저장하고, 마지막에 FP16 출력으로 변환한다.
    # 마지막 key block을 건너뛴 causal query도 여기서 출력한다.
    for start_m in tl.range(0, N_CTX, BLOCK_M):
        rows = start_m + tl.arange(0, BLOCK_M)
        # TODO FA1 6: O_STATE를 mask load하고 FP16으로 cast한 뒤 O에 mask store.
        offsets = head_offset + rows[:, None] * HEAD_DIM + dims[None, :]
        mask = (rows[:, None] < N_CTX) & (dims[None, :] < HEAD_DIM)
        out = tl.load(O_STATE + offsets, mask=mask, other=0.0) # [BLOCK_M, BLOCK_D]
        tl.store(O + offsets, out.to(tl.float16), mask=mask)



def prepare_fa1(q, k, v, causal, sm_scale, *,
                block_m=64, block_n=64, num_warps=4, num_stages=2):
    """scratch/output 사전 할당. 매 호출은 같은 buffer를 초기화부터 계산한다."""
    batch, heads, n, d = validate_inputs(q, k, v)
    assert block_m in (32, 64, 128) and block_n in (32, 64, 128)
    assert num_warps in (4, 8) and num_stages in (1, 2, 3, 4)
    output = torch.empty_like(q)
    state = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    m = torch.empty((batch, heads, n), device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)

    def run():
        run.kernel = _fa1_fwd[(batch * heads,)](
            q, k, v, output, state, m, l, n, sm_scale,
            HEAD_DIM=d, CAUSAL=causal, BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_D=triton.next_power_of_2(d), num_warps=num_warps,
            num_stages=num_stages)
        return output

    return run


def attention_fa1(q, k, v, causal, sm_scale, **config):
    return prepare_fa1(q, k, v, causal, sm_scale, **config)()
