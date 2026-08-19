"""기본 tiled GEMM에서 persistent GEMM으로 발전시키는 실습 뼈대."""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_tile_to_mn(
    tile_id,
    num_m_tiles,
    num_n_tiles,
    GROUP_SIZE_M: tl.constexpr,
):
    """L2 재사용을 위한 grouped ordering으로 linear tile id를 (m, n)에 매핑한다."""
    tiles_per_group = GROUP_SIZE_M * num_n_tiles
    group_id = tile_id // tiles_per_group
    first_m = group_id * GROUP_SIZE_M
    group_m = tl.minimum(num_m_tiles - first_m, GROUP_SIZE_M)
    tile_in_group = tile_id % tiles_per_group
    tile_m = first_m + tile_in_group % group_m
    tile_n = tile_in_group // group_m
    return tile_m, tile_n


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """1 program이 C tile 하나를 계산하는 기본 GEMM kernel."""
    tile_id = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_m, tile_n = _linear_tile_to_mn(
        tile_id, num_m_tiles, num_n_tiles, GROUP_SIZE_M
    )

    # TODO 1: A/B tile pointer와 K-tail mask를 만든다.
    # TODO 2: K축을 BLOCK_K씩 순회하며 FP32 accumulator에 tl.dot을 누적한다.
    # TODO 3: M/N tail mask로 C tile을 저장한다.
    _ = tile_m + tile_n


@triton.jit
def _persistent_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """고정된 program들이 stride scheduling으로 여러 C tile을 처리한다."""
    start_tile = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = num_m_tiles * num_n_tiles

    # 각 program은 start_tile, start_tile+NUM_SMS, ...를 담당한다.
    for tile_id in tl.range(start_tile, num_tiles, NUM_SMS, flatten=True):
        tile_m, tile_n = _linear_tile_to_mn(
            tile_id, num_m_tiles, num_n_tiles, GROUP_SIZE_M
        )
        # TODO 4: 기본 kernel의 tile GEMM 본문을 이 loop 안으로 옮긴다.
        # accumulator는 tile마다 반드시 0으로 초기화한다.
        _ = tile_m + tile_n


def _validate_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    assert a.is_cuda and b.is_cuda
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype == torch.float16
    assert a.shape[1] == b.shape[0]
    assert a.stride(1) == 1 and b.stride(1) == 1
    return a.shape[0], b.shape[1], a.shape[1]


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 tile/program인 출발점. README의 1단계에서 구현한다."""
    M, N, K = _validate_inputs(a, b)
    _ = M, N, K
    raise NotImplementedError("먼저 _matmul_kernel과 launch wrapper를 구현하세요.")


def persistent_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """고정 program 수로 여러 tile을 처리하는 최적화 버전."""
    M, N, K = _validate_inputs(a, b)
    _ = M, N, K
    raise NotImplementedError(
        "기본 matmul 검증 후 _persistent_matmul_kernel과 launch wrapper를 구현하세요."
    )
