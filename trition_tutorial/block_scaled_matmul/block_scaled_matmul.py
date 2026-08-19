"""block scaling의 의미론부터 tl.dot_scaled까지 발전시키는 실습 뼈대."""

import torch
import triton
import triton.language as tl


@triton.jit
def _block_scaled_matmul_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_asm,
    stride_ask,
    stride_bsk,
    stride_bsn,
    stride_cm,
    stride_cn,
    VEC_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FP16 payload를 명시적으로 dequantize하는 portable 출발점."""
    pid = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    tile_m = pid % num_m_tiles
    tile_n = pid // num_m_tiles

    # TODO 1: A [BLOCK_M, BLOCK_K], B [BLOCK_K, BLOCK_N] pointer를 만든다.
    # TODO 2: 각 k의 scale index k // VEC_SIZE로 A/B scale tile을 읽는다.
    # TODO 3: (A * scale_a)와 (B * scale_b)를 tl.dot으로 FP32 누적한다.
    # TODO 4: M/N/K tail을 모두 mask하고 C를 저장한다.
    _ = tile_m + tile_n


@triton.jit
def _block_scaled_dot_kernel(
    a_desc,
    a_scale_desc,
    b_desc,
    b_scale_desc,
    c_desc,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Blackwell에서 preshuffled scale과 tl.dot_scaled를 붙일 후속 단계."""
    # 이 kernel은 기본 구현이 통과한 다음 README의 하드웨어 경로에서 완성한다.
    # TODO 5: descriptor tile과 packed scale tile을 load한다.
    # TODO 6: scale layout을 tl.dot_scaled가 요구하는 2D 논리 shape으로 바꾼다.
    # TODO 7: tl.dot_scaled로 누적하고 descriptor store한다.
    return


def block_scaled_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    vec_size: int,
) -> torch.Tensor:
    """C = (A * scale_a) @ (B * scale_b)의 portable 학습용 API."""
    assert a.is_cuda and b.is_cuda and a_scale.is_cuda and b_scale.is_cuda
    assert a.ndim == b.ndim == a_scale.ndim == b_scale.ndim == 2
    assert a.dtype == b.dtype == torch.float16
    assert a_scale.dtype == b_scale.dtype == torch.float32
    assert a.shape[1] == b.shape[0]
    assert vec_size > 0 and a.shape[1] % vec_size == 0

    M, K = a.shape
    _, N = b.shape
    scale_k = K // vec_size
    assert a_scale.shape == (M, scale_k)
    assert b_scale.shape == (scale_k, N)
    _ = M, N, K
    raise NotImplementedError(
        "먼저 explicit-scale _block_scaled_matmul_kernel과 wrapper를 구현하세요."
    )
