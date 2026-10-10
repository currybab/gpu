"""Forward 확장 실습. 계산 TODO를 채운 뒤 해당 wrapper의 raise를 삭제한다.

기본 stride 버전은 fused_attention.py에 유지한다. 이 파일은 contiguous FP16
self-attention에 집중한다. split/persistent/autotune은 같은 raw kernel을 쓴다.
"""

import torch
import triton
import triton.language as tl


def validate_inputs(q, k, v, *, tma=False, warp_specialize=False):
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape == k.shape == v.shape and min(q.shape) > 0
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.device == k.device == v.device
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert all(x.is_contiguous() for x in (q, k, v)), "확장 실습은 contiguous 입력"
    assert 16 <= q.shape[-1] <= 128
    if tma:
        assert torch.version.cuda is not None, "TMA 실습은 NVIDIA CUDA 전용"
        major, _ = torch.cuda.get_device_capability(q.device)
        assert major >= 9, "TMA 실습은 Hopper 이상에서 실행"
        assert q.shape[-1] in (64, 128), "TMA 첫 실습: D=64/128"
        assert q.shape[-2] % 128 == 0, "TMA 첫 실습: N은 128의 배수"
        assert all(x.data_ptr() % 16 == 0 for x in (q, k, v))
        if warp_specialize:
            assert major >= 10, "자동 warp specialization 실습은 Blackwell에서 실행"
    return q.shape


@triton.jit
def _raw_inner(q, K, V, acc, m_i, l_i, start_m, head_offset,
               N_CTX, sm_scale, HEAD_DIM: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_D: tl.constexpr, STAGE: tl.constexpr):
    # STAGE=1: 완전히 과거인 구간, 2: 대각 구간, 3: non-causal 전체.
    if STAGE == 1:
        lo, hi = 0, start_m
    elif STAGE == 2:
        lo, hi = start_m, tl.minimum(start_m + BLOCK_M, N_CTX)
    else:
        lo, hi = 0, N_CTX
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    for start_n in tl.range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # TODO SPLIT 1: K/V를 head_offset + offs_n * HEAD_DIM + offs_d로 load.
        # token/dimension tail을 모두 mask한다. q는 loop 밖에서 전달받는다.
        # TODO SPLIT 2: QK^T * sm_scale * log2(e), key tail을 -inf로 mask.
        # STAGE == 2일 때만 offs_m >= offs_n 삼각 mask를 추가한다.
        # TODO SPLIT 3: online softmax로 m_i/l_i/acc를 갱신하고 P @ V 누적.
        # 이 helper를 두 번 호출해도 상태는 이어져야 한다.
        _ = offs_n, offs_m, offs_d
    return acc, m_i, l_i


@triton.jit
def _raw_fwd(Q, K, V, O, LSE, N_CTX, N_HEADS_TOTAL, sm_scale,
             HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
             SAVE_LSE: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    tiles_per_head = tl.cdiv(N_CTX, BLOCK_M)
    total_tiles = N_HEADS_TOTAL * tiles_per_head
    # tiled: total_tiles개 program, persistent: SM 수 정도의 program.
    # TODO PERSISTENT: 이 mapping을 설명하고 여러 tile에서 상태가 초기화되는지 확인.
    for tile_id in tl.range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        head = tile_id // tiles_per_head
        start_m = (tile_id % tiles_per_head) * BLOCK_M
        head_offset = head * N_CTX * HEAD_DIM
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        # TODO SPLIT 4: Q tile을 mask와 함께 한 번 load하여 아래 q를 교체.
        q = tl.full((BLOCK_M, BLOCK_D), 0, tl.float16)
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        if CAUSAL:
            acc, m_i, l_i = _raw_inner(
                q, K, V, acc, m_i, l_i, start_m, head_offset, N_CTX, sm_scale,
                HEAD_DIM, BLOCK_M, BLOCK_N, BLOCK_D, 1)
            acc, m_i, l_i = _raw_inner(
                q, K, V, acc, m_i, l_i, start_m, head_offset, N_CTX, sm_scale,
                HEAD_DIM, BLOCK_M, BLOCK_N, BLOCK_D, 2)
        else:
            acc, m_i, l_i = _raw_inner(
                q, K, V, acc, m_i, l_i, start_m, head_offset, N_CTX, sm_scale,
                HEAD_DIM, BLOCK_M, BLOCK_N, BLOCK_D, 3)
        # TODO SPLIT 5: O에 acc / l_i를 FP16 store (row/dimension mask).
        if SAVE_LSE:
            # TODO LSE: LSE[head * N_CTX + offs_m]에 m_i + log2(l_i) 저장.
            # LSE는 FP32 [B,H,N], base-2 값이며 row tail만 mask한다.
            _ = LSE, m_i, l_i
        _ = offs_m, offs_d


# TODO TUNE: 고정 설정의 correctness를 통과한 뒤 후보를 확장한다.
# 현재 후보는 causal 구간 경계가 BLOCK_N에 정렬되도록 BM % BN == 0이다.
RAW_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
    for bm, bn, w, s in ((64, 32, 4, 2), (64, 64, 4, 2), (128, 64, 8, 3))
]
_autotuned_raw_fwd = triton.autotune(
    configs=RAW_CONFIGS,
    key=["N_CTX", "N_HEADS_TOTAL", "HEAD_DIM", "CAUSAL", "SAVE_LSE"],
)(_raw_fwd)


def _launch_raw(q, k, v, causal, sm_scale, *, persistent=False,
                autotune=False, save_lse=False, programs_per_sm=1):
    batch, heads, n, d = validate_inputs(q, k, v)
    assert programs_per_sm > 0
    output = torch.empty_like(q)
    lse = torch.empty((batch, heads, n), device=q.device, dtype=torch.float32)
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count

    def grid(meta):
        total = batch * heads * triton.cdiv(n, meta["BLOCK_M"])
        return (min(total, num_sms * programs_per_sm) if persistent else total,)

    kernel = _autotuned_raw_fwd if autotune else _raw_fwd
    config = {} if autotune else dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2)
    kernel[grid](q, k, v, output, lse, n, batch * heads, sm_scale,
                 HEAD_DIM=d, BLOCK_D=triton.next_power_of_2(d),
                 CAUSAL=causal, SAVE_LSE=save_lse, **config)
    return (output, lse) if save_lse else output


def attention_split(q, k, v, causal, sm_scale):
    """Causal off-band/diagonal 분리 실습. raw pointer, contiguous."""
    raise NotImplementedError("_raw_inner/_raw_fwd의 SPLIT 1~5 구현 후 이 raise 삭제")
    return _launch_raw(q, k, v, causal, sm_scale)


def attention_persistent(q, k, v, causal, sm_scale, *, programs_per_sm=1):
    """동일한 query tile 계산을 고정된 수의 program에 분배하는 실습."""
    raise NotImplementedError("SPLIT 완성 후 PERSISTENT mapping과 상태 초기화 확인; raise 삭제")
    return _launch_raw(q, k, v, causal, sm_scale, persistent=True,
                       programs_per_sm=programs_per_sm)


def attention_autotuned(q, k, v, causal, sm_scale):
    """Raw split kernel의 tile/warp/stage 후보를 탐색하는 실습."""
    raise NotImplementedError("SPLIT 완성 후 RAW_CONFIGS 각각 검증; 이 raise 삭제")
    return _launch_raw(q, k, v, causal, sm_scale, autotune=True)


@triton.jit
def _tma_inner(q, desc_k, desc_v, acc, m_i, l_i, start_m, head_row,
               N_CTX, sm_scale, HEAD_DIM: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               STAGE: tl.constexpr, WARP_SPECIALIZE: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m
    elif STAGE == 2:
        lo, hi = start_m, start_m + BLOCK_M
    else:
        lo, hi = 0, N_CTX
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=WARP_SPECIALIZE):
        # TODO TMA 1: desc_k/desc_v.load([head_row + start_n, 0]).
        # tile shape은 [BLOCK_N, HEAD_DIM]; K를 transpose해서 QK^T를 계산.
        # TODO TMA 2: SPLIT의 scale/mask/online softmax/PV를 옮긴다.
        # descriptor는 메모리 이동만 담당하며 softmax/mask는 직접 구현한다.
        # TODO WS: 같은 kernel의 False/True 비교. 이 K/V loop만 WS 대상이다.
        # compiler가 지원하는 loop 형태/accumulator layout인지 확인한다.
        _ = start_n
    return acc, m_i, l_i


@triton.jit
def _tma_fwd(desc_q, desc_k, desc_v, desc_o, N_CTX, sm_scale,
             HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             WARP_SPECIALIZE: tl.constexpr):
    start_m = tl.program_id(0) * BLOCK_M
    head_row = tl.program_id(1) * N_CTX
    # TODO TMA 3: desc_q.load([head_row + start_m, 0])으로 q 교체.
    q = tl.full((BLOCK_M, HEAD_DIM), 0, tl.float16)
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    if CAUSAL:
        acc, m_i, l_i = _tma_inner(
            q, desc_k, desc_v, acc, m_i, l_i, start_m, head_row,
            N_CTX, sm_scale, HEAD_DIM, BLOCK_M, BLOCK_N, 1, WARP_SPECIALIZE)
        acc, m_i, l_i = _tma_inner(
            q, desc_k, desc_v, acc, m_i, l_i, start_m, head_row,
            N_CTX, sm_scale, HEAD_DIM, BLOCK_M, BLOCK_N, 2, WARP_SPECIALIZE)
    else:
        acc, m_i, l_i = _tma_inner(
            q, desc_k, desc_v, acc, m_i, l_i, start_m, head_row,
            N_CTX, sm_scale, HEAD_DIM, BLOCK_M, BLOCK_N, 3, WARP_SPECIALIZE)
    # TODO TMA 4: normalize/cast 후 desc_o.store([head_row + start_m, 0], out).


def _launch_tma(q, k, v, causal, sm_scale, *, warp_specialize=False):
    # 기본 raw 실습이 descriptor API import에 의존하지 않도록 lazy import.
    from triton.tools.tensor_descriptor import TensorDescriptor

    batch, heads, n, d = validate_inputs(q, k, v, tma=True,
                                        warp_specialize=warp_specialize)
    output = torch.empty_like(q)
    block_m, block_n = 64, 64
    # [B,H,N,D]를 [B*H*N,D]로 해석. N이 tile 배수여야 head 경계를 안 넘는다.
    # TODO TMA 5: shape/strides/block_shape의 차이를 확인하고 다른 tile도 실험.
    descriptors = [
        TensorDescriptor(base=x, shape=[batch * heads * n, d], strides=[d, 1],
                         block_shape=[block, d])
        for x, block in ((q, block_m), (k, block_n), (v, block_n), (output, block_m))
    ]
    # host descriptor를 전달하는 Triton kernel의 global scratch allocator.
    def allocator(size, alignment, stream):
        return torch.empty(size, device=q.device, dtype=torch.uint8)

    triton.set_allocator(allocator)
    _tma_fwd[(triton.cdiv(n, block_m), batch * heads)](
        *descriptors, n, sm_scale, HEAD_DIM=d, CAUSAL=causal,
        BLOCK_M=block_m, BLOCK_N=block_n, WARP_SPECIALIZE=warp_specialize,
        num_warps=4, num_stages=2)
    return output


def attention_tma(q, k, v, causal, sm_scale):
    """Host TensorDescriptor 기반 TMA load/store 실습."""
    raise NotImplementedError("TMA 1~5 구현 후 이 raise 삭제")
    return _launch_tma(q, k, v, causal, sm_scale)


def attention_tma_ws(q, k, v, causal, sm_scale):
    """같은 TMA kernel에서 K/V loop 자동 warp specialization을 요청하는 실습."""
    raise NotImplementedError("TMA 완성 후 WS loop/layout 확인; 이 raise 삭제")
    return _launch_tma(q, k, v, causal, sm_scale, warp_specialize=True)
