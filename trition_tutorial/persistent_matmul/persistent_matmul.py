"""기본 tiled GEMM을 persistent scheduling으로 바꾸는 실습 뼈대."""

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """1 program이 C tile 하나를 계산한다."""
    tile_id = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_m = tile_id // num_n_tiles
    tile_n = tile_id % num_n_tiles

    # TODO 1: A/B pointer와 M/N/K mask를 만든다.
    # TODO 2: K축을 순회하며 FP32 acc에 tl.dot을 누적한다.
    # TODO 3: C tile을 저장한다.
    _ = tile_m + tile_n


@triton.jit
def _persistent_matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """program 하나가 일정한 간격으로 여러 C tile을 계산한다."""
    start_tile = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = num_m_tiles * num_n_tiles

    for tile_id in tl.range(start_tile, num_tiles, NUM_PROGRAMS):
        tile_m = tile_id // num_n_tiles
        tile_n = tile_id % num_n_tiles

        # TODO 4: 기본 kernel의 tile GEMM 본문을 이곳에 옮긴다.
        # acc와 pointer는 tile마다 새로 초기화해야 한다.
        _ = tile_m + tile_n


def _shape(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    assert a.is_cuda and b.is_cuda
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype == torch.float16
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous() and b.is_contiguous()
    return a.shape[0], b.shape[1], a.shape[1]


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """README 1단계: 전체 output tile 수만큼 program을 launch한다."""
    M, N, K = _shape(a, b)
    _ = M, N, K
    raise NotImplementedError("_matmul_kernel의 TODO 1~3과 launch를 구현하세요.")


def persistent_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """README 2단계: 고정된 program들이 여러 output tile을 처리한다."""
    M, N, K = _shape(a, b)
    _ = M, N, K
    raise NotImplementedError(
        "기본 matmul 검증 후 _persistent_matmul_kernel의 TODO 4를 구현하세요."
    )
