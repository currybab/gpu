"""scale을 명시적으로 적용하는 block-scaled matmul 실습 뼈대."""

import torch
import triton
import triton.language as tl


@triton.jit
def _block_scaled_matmul_kernel(
    A,
    B,
    A_SCALE,
    B_SCALE,
    C,
    M,
    N,
    K,
    VEC_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FP16 payload와 FP32 scale로 계산 의미부터 확인한다."""
    tile_id = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    tile_m = tile_id % num_m_tiles
    tile_n = tile_id // num_m_tiles

    # TODO 1: contiguous A/B tile pointer와 M/N/K mask를 만든다.
    # TODO 2: offs_k // VEC_SIZE 위치의 A/B scale을 load한다.
    # TODO 3: tl.dot(A * scale_a, B * scale_b)을 FP32로 누적한다.
    # TODO 4: C tile을 저장한다.
    _ = tile_m + tile_n


def block_scaled_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    vec_size: int,
) -> torch.Tensor:
    """C = (A * scale_a) @ (B * scale_b)의 학습용 API."""
    assert a.is_cuda and b.is_cuda and a_scale.is_cuda and b_scale.is_cuda
    assert a.ndim == b.ndim == a_scale.ndim == b_scale.ndim == 2
    assert a.dtype == b.dtype == torch.float16
    assert a_scale.dtype == b_scale.dtype == torch.float32
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous() and b.is_contiguous()
    assert a_scale.is_contiguous() and b_scale.is_contiguous()
    assert vec_size > 0 and a.shape[1] % vec_size == 0

    M, K = a.shape
    _, N = b.shape
    scale_k = K // vec_size
    assert a_scale.shape == (M, scale_k)
    assert b_scale.shape == (scale_k, N)

    _ = M, N, K
    raise NotImplementedError("_block_scaled_matmul_kernel의 TODO 1~4와 launch를 구현하세요.")
