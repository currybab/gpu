"""서로 다른 shape의 GEMM들을 한 persistent launch로 처리하는 실습 뼈대."""

import torch
import triton
import triton.language as tl


@triton.jit
def _group_gemm_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_sizes,
    group_strides,
    group_size,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """flatten한 전체 GEMM tile 공간을 고정 program들이 device에서 스케줄한다."""
    tile_id = tl.program_id(0)
    problem_start = 0

    for group_idx in range(group_size):
        M = tl.load(group_sizes + group_idx * 3)
        N = tl.load(group_sizes + group_idx * 3 + 1)
        K = tl.load(group_sizes + group_idx * 3 + 2)
        num_m_tiles = tl.cdiv(M, BLOCK_M)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        problem_tiles = num_m_tiles * num_n_tiles
        problem_end = problem_start + problem_tiles

        while tile_id >= problem_start and tile_id < problem_end:
            tile_in_problem = tile_id - problem_start
            tile_m = tile_in_problem // num_n_tiles
            tile_n = tile_in_problem % num_n_tiles

            # TODO 1: group_idx의 A/B/C pointer와 stride를 device metadata에서 읽는다.
            # pointer는 .to(tl.pointer_type(tl.float16))로 복원한다.
            # TODO 2: (tile_m, tile_n)의 masked tiled GEMM을 수행한다.
            # TODO 3: M/N tail mask로 현재 C에 저장한다.
            _ = tile_m + tile_n + K

            # 같은 program이 flattened tile 공간의 다음 몫을 가져간다.
            tile_id += NUM_PROGRAMS

        problem_start = problem_end


def _prepare_metadata(group_a, group_b, group_c):
    """Python tensor 목록을 kernel이 읽을 device-side pointer/metadata tensor로 바꾼다."""
    a_ptrs = torch.tensor([x.data_ptr() for x in group_a], device="cuda")
    b_ptrs = torch.tensor([x.data_ptr() for x in group_b], device="cuda")
    c_ptrs = torch.tensor([x.data_ptr() for x in group_c], device="cuda")

    sizes = []
    strides = []
    for a, b, c in zip(group_a, group_b, group_c):
        M, K = a.shape
        _, N = b.shape
        sizes.extend((M, N, K))
        strides.extend((a.stride(0), b.stride(0), c.stride(0)))

    sizes = torch.tensor(sizes, dtype=torch.int32, device="cuda")
    strides = torch.tensor(strides, dtype=torch.int32, device="cuda")
    return a_ptrs, b_ptrs, c_ptrs, sizes, strides


def group_gemm(group_a: list[torch.Tensor], group_b: list[torch.Tensor]):
    """각 i에 대해 group_a[i] @ group_b[i]를 한 Triton launch로 계산한다."""
    assert len(group_a) == len(group_b) and len(group_a) > 0
    for a, b in zip(group_a, group_b):
        assert a.is_cuda and b.is_cuda
        assert a.ndim == b.ndim == 2
        assert a.dtype == b.dtype == torch.float16
        assert a.shape[1] == b.shape[0]
        assert a.stride(1) == 1 and b.stride(1) == 1

    group_c = [
        torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        for a, b in zip(group_a, group_b)
    ]
    metadata = _prepare_metadata(group_a, group_b, group_c)
    _ = metadata
    raise NotImplementedError(
        "_group_gemm_kernel의 TODO를 채우고 고정 NUM_PROGRAMS grid로 launch하세요."
    )
