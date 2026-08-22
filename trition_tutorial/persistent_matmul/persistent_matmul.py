"""기본 tiled GEMM을 persistent scheduling으로 바꾸는 실습 뼈대."""

import torch
import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

_PRINTED_KERNEL_METADATA: set[str] = set()


def _print_kernel_metadata_once(name: str, kernel, config: str) -> None:
    if name in _PRINTED_KERNEL_METADATA:
        return

    _PRINTED_KERNEL_METADATA.add(name)
    print("-" * 80)
    print(f"{name} kernel: {config}")
    print(f"regs/thread : {kernel.n_regs}")
    print(f"spills      : {kernel.n_spills}")
    print(f"shared/CTA  : {kernel.metadata.shared}")
    print(f"num_warps   : {kernel.metadata.num_warps}")
    print("-" * 80)

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
):
    """1 program이 C tile 하나를 계산한다."""
    tile_id = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_m = tile_id // num_n_tiles
    tile_n = tile_id % num_n_tiles

    # TODO 1: a_ptr/b_ptr와 stride로 tile pointer와 M/N/K mask를 만든다.
    offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_row = a_ptr + offsets_m[:, None] * stride_am
    b_col = b_ptr + offsets_n[None, :] * stride_bn
    mask_m = offsets_m[:, None] < M
    mask_n = offsets_n[None, :] < N

    # TODO 2: K축을 순회하며 FP32 acc에 tl.dot을 누적한다.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, num_stages=4):
        offsets_k = k + tl.arange(0, BLOCK_K)
        mask_a = mask_m & (offsets_k[None, :] < K)
        mask_b = mask_n & (offsets_k[:, None] < K)
        a_tile = tl.load(a_row + offsets_k[None, :] * stride_ak, mask=mask_a, other=0.0)
        b_tile = tl.load(b_col + offsets_k[:, None] * stride_bk, mask=mask_b, other=0.0)
        acc = tl.dot(a_tile, b_tile, acc)

    # TODO 3: C tile을 저장한다.
    tl.store(c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn, acc, mask=mask_m & mask_n)


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

        # TODO 4: 기본 kernel의 stride 기반 tile GEMM 본문을 이곳에 옮긴다.
        # acc와 pointer는 tile마다 새로 초기화해야 한다.
        
        # a_ptr/b_ptr와 stride로 tile pointer와 M/N/K mask를 만든다.
        offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a_row = a_ptr + offsets_m[:, None] * stride_am
        b_col = b_ptr + offsets_n[None, :] * stride_bn
        mask_m = offsets_m[:, None] < M
        mask_n = offsets_n[None, :] < N

        # K축을 순회하며 FP32 acc에 tl.dot을 누적한다.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, K, BLOCK_K, num_stages=4):
            offsets_k = k + tl.arange(0, BLOCK_K)
            mask_a = mask_m & (offsets_k[None, :] < K)
            mask_b = mask_n & (offsets_k[:, None] < K)
            a_tile = tl.load(a_row + offsets_k[None, :] * stride_ak, mask=mask_a, other=0.0)
            b_tile = tl.load(b_col + offsets_k[:, None] * stride_bk, mask=mask_b, other=0.0)
            acc = tl.dot(a_tile, b_tile, acc)

        # C tile을 저장한다.
        tl.store(c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn, acc, mask=mask_m & mask_n)


def _shape(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    assert a.is_cuda and b.is_cuda
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype == torch.float16
    assert a.shape[1] == b.shape[0]
    assert a.device == b.device
    return a.shape[0], b.shape[1], a.shape[1]


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """README 1단계: 전체 output tile 수만큼 program을 launch한다."""
    M, N, K = _shape(a, b)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stride_cm, stride_cn = c.stride()
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    k = _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
    )
    _print_kernel_metadata_once(
        "naive tiled",
        k,
        f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}",
    )
    return c


def persistent_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """README 2단계: 고정된 program들이 여러 output tile을 처리한다."""
    M, N, K = _shape(a, b)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stride_cm, stride_cn = c.stride()

    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count * 2
    num_programs = min(num_sms, num_tiles)
    grid = (num_programs,)
    k = _persistent_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        NUM_PROGRAMS=num_programs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        maxnreg=128,
    )
    _print_kernel_metadata_once(
        "persistent",
        k,
        (
            f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, "
            f"NUM_PROGRAMS={num_programs}"
        ),
    )
    return c
