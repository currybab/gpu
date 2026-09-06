"""기본 tiled GEMM을 persistent scheduling으로 바꾸는 실습 뼈대."""

import re
import torch
import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 64
BLOCK_K = 32

_PRINTED_KERNEL_METADATA: set[tuple[object, ...]] = set()


def _print_kernel_metadata_once(
    name: str,
    kernel,
    config: str,
    cache_key: tuple[object, ...],
) -> None:
    if cache_key in _PRINTED_KERNEL_METADATA:
        return

    _PRINTED_KERNEL_METADATA.add(cache_key)
    print("-" * 80)
    print(f"{name} kernel: {config}")
    print(f"regs/thread : {kernel.n_regs}")
    print(f"spills      : {kernel.n_spills}")
    print(f"shared/CTA  : {kernel.metadata.shared}")
    print(f"num_warps   : {kernel.metadata.num_warps}")
    ptx = kernel.asm["ptx"]
    ids = set(re.findall(r"bar\.sync\s+(\d+)", ptx)) | set(re.findall(r"barrier\.cta\.sync[.\w]*\s+(\d+)", ptx))
    print(f"barriers    : {len(ids)}")
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
    NUM_STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """program 하나가 일정한 간격으로 여러 C tile을 계산한다."""
    start_tile = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = num_m_tiles * num_n_tiles

    for tile_id in tl.range(start_tile, num_tiles, NUM_PROGRAMS, num_stages=NUM_STAGES):
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
        for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
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
        num_warps=4,
    )
    _print_kernel_metadata_once(
        "naive tiled",
        k,
        f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}",
        ("naive tiled", BLOCK_M, BLOCK_N, BLOCK_K, 4),
    )
    return c


def persistent_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = BLOCK_M,
    block_n: int = BLOCK_N,
    block_k: int = BLOCK_K,
    num_warps: int = 4,
    num_stages: int = 4,
    programs_per_sm: int = 2,
    maxnreg: int | None = None,
    print_metadata: bool = True,
) -> torch.Tensor:
    """README 2단계: 고정된 program들이 여러 output tile을 처리한다."""
    M, N, K = _shape(a, b)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stride_cm, stride_cn = c.stride()

    num_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_programs = min(num_sms * programs_per_sm, num_tiles)
    grid = (num_programs,)
    launch_options = {
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg

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
        NUM_STAGES=num_stages,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        **launch_options,
    )
    if print_metadata:
        _print_kernel_metadata_once(
            "naive persistent",
            k,
            (
                f"BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}, "
                f"NUM_PROGRAMS={num_programs}, num_stages={num_stages}"
            ),
            (
                "naive persistent",
                block_m,
                block_n,
                block_k,
                num_warps,
                num_stages,
                programs_per_sm,
                maxnreg,
            ),
        )
    return c


# 3단계: program tile swizzling 실습
@triton.jit
def _swizzle_matmul_kernel(
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
    """1 program이 C tile 하나를 계산한다."""
    tile_id = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    # TODO SWIZZLE 1: GROUP_SIZE_M개의 M tile을 한 그룹으로 묶는다.
    # num_tiles_in_group, group_id, first_tile_m을 계산한다.
    num_tiles_in_group = GROUP_SIZE_M * num_n_tiles
    group_id = tile_id // num_tiles_in_group
    tile_id_in_group = tile_id % num_tiles_in_group
    first_tile_m = group_id * GROUP_SIZE_M
    # TODO SWIZZLE 2: 마지막 그룹의 실제 M tile 수 group_size_m을 계산한다.
    # tl.minimum을 사용한다. GROUP_SIZE_M보다 작을 수 있다.
    group_size_m = tl.minimum(GROUP_SIZE_M, num_m_tiles - first_tile_m)
    # TODO SWIZZLE 3: 그룹 내부 linear id에서 M이 먼저 증가하도록 좌표를 만든다.
    # 아래 row-major 두 줄은 임시값이다. grouped mapping으로 교체한다.
    tile_m = first_tile_m + (tile_id_in_group % group_size_m)
    tile_n = tile_id_in_group // group_size_m

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


@triton.jit
def _persistent_swizzle_matmul_kernel(
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
    NUM_STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """program 하나가 일정한 간격으로 여러 C tile을 계산한다."""
    start_tile = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = num_m_tiles * num_n_tiles

    for tile_id in tl.range(start_tile, num_tiles, NUM_PROGRAMS, num_stages=NUM_STAGES):
        # TODO SWIZZLE 1: GROUP_SIZE_M개의 M tile을 한 그룹으로 묶는다.
        # num_tiles_in_group, group_id, first_tile_m을 계산한다.
        num_tiles_in_group = GROUP_SIZE_M * num_n_tiles
        group_id = tile_id // num_tiles_in_group
        tile_id_in_group = tile_id % num_tiles_in_group
        first_tile_m = group_id * GROUP_SIZE_M
        # TODO SWIZZLE 2: 마지막 그룹의 실제 M tile 수 group_size_m을 계산한다.
        # tl.minimum을 사용한다. GROUP_SIZE_M보다 작을 수 있다.
        group_size_m = tl.minimum(GROUP_SIZE_M, num_m_tiles - first_tile_m)
        # TODO SWIZZLE 3: 그룹 내부 linear id에서 M이 먼저 증가하도록 좌표를 만든다.
        # 아래 row-major 두 줄은 임시값이다. grouped mapping으로 교체한다.
        tile_m = first_tile_m + (tile_id_in_group % group_size_m)
        tile_n = tile_id_in_group // group_size_m

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
        for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            offsets_k = k + tl.arange(0, BLOCK_K)
            mask_a = mask_m & (offsets_k[None, :] < K)
            mask_b = mask_n & (offsets_k[:, None] < K)
            a_tile = tl.load(a_row + offsets_k[None, :] * stride_ak, mask=mask_a, other=0.0)
            b_tile = tl.load(b_col + offsets_k[:, None] * stride_bk, mask=mask_b, other=0.0)
            acc = tl.dot(a_tile, b_tile, acc)

        # C tile을 저장한다.
        tl.store(c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn, acc, mask=mask_m & mask_n)


def swizzle_matmul(a: torch.Tensor, b: torch.Tensor, *, group_size_m: int = 8) -> torch.Tensor:
    """3단계 실습: tiled GEMM에 grouped ordering을 추가한다."""
    M, N, K = _shape(a, b)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stride_cm, stride_cn = c.stride()
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    k = _swizzle_matmul_kernel[grid](
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
        GROUP_SIZE_M=group_size_m,
        num_warps=4,
    )
    _print_kernel_metadata_once(
        "swizzle tiled",
        k,
        f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, GROUP_SIZE_M={group_size_m}",
        ("swizzle tiled", BLOCK_M, BLOCK_N, BLOCK_K, 4, group_size_m),
    )
    return c


def persistent_swizzle_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    group_size_m: int = 8,
    block_m: int = BLOCK_M,
    block_n: int = BLOCK_N,
    block_k: int = BLOCK_K,
    num_warps: int = 4,
    num_stages: int = 4,
    programs_per_sm: int = 2,
    maxnreg: int | None = None,
    print_metadata: bool = True,
) -> torch.Tensor:
    """3단계 실습: persistent GEMM에 grouped ordering을 추가한다."""
    M, N, K = _shape(a, b)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stride_cm, stride_cn = c.stride()

    num_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_programs = min(num_sms * programs_per_sm, num_tiles)
    grid = (num_programs,)
    launch_options = {
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    if maxnreg is not None:
        launch_options["maxnreg"] = maxnreg

    k = _persistent_swizzle_matmul_kernel[grid](
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
        NUM_STAGES=num_stages,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=group_size_m,
        **launch_options,
    )
    if print_metadata:
        _print_kernel_metadata_once(
            "swizzle persistent",
            k,
            (
                f"BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}, "
                f"NUM_PROGRAMS={num_programs}, num_stages={num_stages}, GROUP_SIZE_M={group_size_m}"
            ),
            (
                "swizzle persistent",
                block_m,
                block_n,
                block_k,
                num_warps,
                num_stages,
                programs_per_sm,
                maxnreg,
                group_size_m,
            ),
        )
    return c
