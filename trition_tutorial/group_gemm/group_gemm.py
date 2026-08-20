"""서로 다른 shape의 GEMM들을 한 launch로 처리하는 실습 뼈대."""

import torch
import triton
import triton.language as tl


@triton.jit
def _group_gemm_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_sizes,
    group_size,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_id = tl.program_id(0)
    problem_start = 0

    for group_idx in range(group_size):
        M = tl.load(group_sizes + group_idx * 3)
        N = tl.load(group_sizes + group_idx * 3 + 1)
        K = tl.load(group_sizes + group_idx * 3 + 2)
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        problem_tiles = tl.cdiv(M, BLOCK_M) * num_n_tiles
        problem_end = problem_start + problem_tiles

        while tile_id >= problem_start and tile_id < problem_end:
            tile_in_problem = tile_id - problem_start
            tile_m = tile_in_problem // num_n_tiles
            tile_n = tile_in_problem % num_n_tiles

            # TODO 1: 현재 A/B/C 주소를 FP16 pointer로 복원한다.
            # TODO 2: contiguous stride K/N을 사용해 masked tiled GEMM을 수행한다.
            # TODO 3: 현재 C tile을 저장한다.
            _ = tile_m + tile_n + K

            tile_id += NUM_PROGRAMS

        problem_start = problem_end


def group_gemm(group_a: list[torch.Tensor], group_b: list[torch.Tensor]):
    """각 i에 대해 group_a[i] @ group_b[i]를 한 Triton launch로 계산한다."""
    assert len(group_a) == len(group_b) and len(group_a) > 0
    for a, b in zip(group_a, group_b):
        assert a.is_cuda and b.is_cuda
        assert a.ndim == b.ndim == 2
        assert a.dtype == b.dtype == torch.float16
        assert a.shape[1] == b.shape[0]
        assert a.is_contiguous() and b.is_contiguous()

    group_c = [
        torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        for a, b in zip(group_a, group_b)
    ]
    a_ptrs = torch.tensor([a.data_ptr() for a in group_a], device="cuda")
    b_ptrs = torch.tensor([b.data_ptr() for b in group_b], device="cuda")
    c_ptrs = torch.tensor([c.data_ptr() for c in group_c], device="cuda")
    sizes = torch.tensor(
        [(a.shape[0], b.shape[1], a.shape[1]) for a, b in zip(group_a, group_b)],
        dtype=torch.int32,
        device="cuda",
    )

    _ = a_ptrs, b_ptrs, c_ptrs, sizes
    raise NotImplementedError("_group_gemm_kernel의 TODO 1~3과 launch를 구현하세요.")
