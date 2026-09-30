"""기본 tiled GEMM을 persistent scheduling으로 바꾸는 실습 뼈대."""

import re
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

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


# 5단계: cross-tile pipelining (compiler-assisted flattening) 실습


@triton.jit
def cross_tile_pipelining_kernel(
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
    """실습: persistent tile loop를 flatten하여 타일 경계의 overlap을 시도한다."""
    start_tile = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = num_m_tiles * num_n_tiles

    # TODO CROSS 1: flatten=False를 True로 바꿔 중첩 loop flattening을 요청한다.
    # 현재 형태는 비교용 이중 loop다. outer num_stages만으로 cross-tile을 보장하지 않는다.
    # 다음 타일은 tile_id + NUM_PROGRAMS이며, 다음 좌표는 grouped mapping으로 구한다.
    for tile_id in tl.range(
        start_tile, num_tiles, NUM_PROGRAMS,
        num_stages=NUM_STAGES,
        flatten=True,
    ):
        num_tiles_in_group = GROUP_SIZE_M * num_n_tiles
        group_id = tile_id // num_tiles_in_group
        tile_id_in_group = tile_id % num_tiles_in_group
        first_tile_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(GROUP_SIZE_M, num_m_tiles - first_tile_m)
        tile_m = first_tile_m + (tile_id_in_group % group_size_m)
        tile_n = tile_id_in_group // group_size_m

        # TODO CROSS 2: flatten 이후에도 아래 상태가 타일마다 새로 만들어지는지 확인한다.
        # pointer는 현재 tile_m/tile_n 기준, acc는 0부터 시작해야 한다.
        # compiler-assisted 실습에서는 이 초기화와 store의 소스 위치를 유지한다.

        # a_ptr/b_ptr와 stride로 tile pointer와 M/N/K mask를 만든다.
        offsets_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a_row = a_ptr + offsets_m[:, None] * stride_am
        b_col = b_ptr + offsets_n[None, :] * stride_bn
        mask_m = offsets_m[:, None] < M
        mask_n = offsets_n[None, :] < N

        # K축을 순회하며 FP32 acc에 tl.dot을 누적한다.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, K, BLOCK_K):
            offsets_k = k + tl.arange(0, BLOCK_K)
            mask_a = mask_m & (offsets_k[None, :] < K)
            mask_b = mask_n & (offsets_k[:, None] < K)
            a_tile = tl.load(a_row + offsets_k[None, :] * stride_ak, mask=mask_a, other=0.0)
            b_tile = tl.load(b_col + offsets_k[:, None] * stride_bk, mask=mask_b, other=0.0)
            acc = tl.dot(a_tile, b_tile, acc)

        # TODO CROSS 3: K 전체 누적 후 현재 타일에 정확히 한 번 store되는지 검증한다.
        # 여러 타일을 맡는 program, 마지막 K block, 마지막 M/N 타일을 확인한다.
        tl.store(c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn, acc, mask=mask_m & mask_n)


def cross_tile_pipelining_matmul(
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
    """5단계 실습: 같은 grouped ordering에서 loop flattening의 효과를 비교한다."""
    # TODO CROSS 4: 커널의 CROSS 1 적용 후 아래 raise를 삭제하고 check.py를 실행한다.
    # CROSS 2~3은 상태/정확도 확인 항목이다. 성능 검증 방법은 README 5단계 참고.
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

    k = cross_tile_pipelining_kernel[grid](
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
            "cross-tile persistent",
            k,
            (
                f"BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}, "
                f"NUM_PROGRAMS={num_programs}, num_stages={num_stages}, GROUP_SIZE_M={group_size_m}"
            ),
            (
                "cross-tile persistent",
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


# TMA 실습: TensorDescriptor를 사용하는 일반 tiled GEMM
TMA_BLOCK_M = 128
TMA_BLOCK_N = 128
TMA_BLOCK_K = 64


@triton.jit
def _tma_matmul_kernel(
    a_desc, b_desc, c_desc,
    N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 기존 tiled GEMM과 동일: program 하나가 C tile 하나를 담당한다.
    tile_id = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_m = tile_id // num_n_tiles
    tile_n = tile_id % num_n_tiles
    start_m = tile_m * BLOCK_M
    start_n = tile_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in tl.range(0, K, BLOCK_K, warp_specialize=False):
        # TODO 2: a_desc.load와 b_desc.load로 tile을 읽는다.
        # load에는 원소 단위 시작 좌표 두 개를 리스트로 전달한다.
        # A의 축: [M, K], B의 축: [K, N]
        # load 좌표: A [start_m, start_k], B [start_k, start_n]
        # 결과 shape: a_tile[BLOCK_M, BLOCK_K], b_tile[BLOCK_K, BLOCK_N]
        # tl.arange로 pointer를 만들거나 mask를 전달할 필요가 없다.
        a_tile = a_desc.load([start_m, start_k])
        b_tile = b_desc.load([start_k, start_n])
        # TODO 3: a_tile과 b_tile을 tl.dot으로 acc에 누적한다.
        acc = tl.dot(a_tile, b_tile, acc)

    # TODO 4: acc를 FP16으로 변환하고 c_desc.store로 저장한다.
    # C의 축은 [M, N]. store(시작 좌표 리스트, 저장할 tile) 형태다.
    c_tile = acc.to(tl.float16)
    c_desc.store([start_m, start_n], c_tile)


def tma_matmul(a: torch.Tensor, b: torch.Tensor, print_metadata: bool = True) -> torch.Tensor:
    """A[M,K] @ B[K,N] -> C[M,N]. 입력은 contiguous tensor다."""
    assert a.is_cuda and b.is_cuda and a.device == b.device
    assert torch.version.cuda is not None, "NVIDIA CUDA 환경이 필요합니다."
    assert torch.cuda.get_device_capability(a.device)[0] >= 9, "TMA 지원 GPU가 필요합니다."
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype == torch.float16
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b and min(M, N, K) > 0
    # FP16 row stride의 16-byte 정렬. tile 배수일 필요는 없다.
    assert K % 8 == 0 and N % 8 == 0, "첫 실습에서는 K/N을 8의 배수로 사용합니다."
    assert a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # TODO 1: TensorDescriptor(base=..., shape=[...], strides=[...],
    #                          block_shape=[...])로 아래 None 세 개를 교체한다.
    # shape/strides는 전체 tensor의 정보, block_shape는 load/store 한 번의 크기다.
    # a_desc: base=a,   축 [M,K], block [TMA_BLOCK_M,TMA_BLOCK_K]
    # b_desc: base=b,   축 [K,N], block [TMA_BLOCK_K,TMA_BLOCK_N]
    # c_desc: base=c,   축 [M,N], block [TMA_BLOCK_M,TMA_BLOCK_N]
    # stride는 byte가 아닌 원소 단위이며 tensor.stride()에서 얻는다.
    a_desc = TensorDescriptor(base=a, shape=[M, K], strides=[a.stride(0), a.stride(1)], block_shape=[TMA_BLOCK_M, TMA_BLOCK_K])
    b_desc = TensorDescriptor(base=b, shape=[K, N], strides=[b.stride(0), b.stride(1)], block_shape=[TMA_BLOCK_K, TMA_BLOCK_N])
    c_desc = TensorDescriptor(base=c, shape=[M, N], strides=[c.stride(0), c.stride(1)], block_shape=[TMA_BLOCK_M, TMA_BLOCK_N])

    grid = (triton.cdiv(M, TMA_BLOCK_M) * triton.cdiv(N, TMA_BLOCK_N),)
    kernel = _tma_matmul_kernel[grid](
        a_desc, b_desc, c_desc, N, K,
        BLOCK_M=TMA_BLOCK_M, BLOCK_N=TMA_BLOCK_N, BLOCK_K=TMA_BLOCK_K,
        num_warps=4, num_stages=2,
    )
    if print_metadata:
        _print_kernel_metadata_once(
            "TMA tiled",
            kernel,
            (
                f"BLOCK_M={TMA_BLOCK_M}, BLOCK_N={TMA_BLOCK_N}, BLOCK_K={TMA_BLOCK_K}, "
                f"num_stages={kernel.metadata.num_stages}, "
                "warp_specialize=False"
            ),
            (
                "TMA tiled", a.device, a.dtype, M, N, K,
                TMA_BLOCK_M, TMA_BLOCK_N, TMA_BLOCK_K,
                kernel.metadata.num_warps, kernel.metadata.num_stages,
            ),
        )
    return c


# TMA persistent 실습: 고정된 program들이 여러 C tile을 처리한다.
@triton.jit
def _tma_persistent_matmul_kernel(
    a_desc, b_desc, c_desc,
    M, N, K,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    start_tile = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = num_m_tiles * num_n_tiles

    for tile_id in tl.range(
        start_tile, num_tiles, NUM_PROGRAMS, warp_specialize=False, flatten=True
    ):
        # tile_id는 매 iteration 달라진다. program_id로 좌표를 계산하면 안 된다.
        tile_m = tile_id // num_n_tiles
        tile_n = tile_id % num_n_tiles
        start_m = tile_m * BLOCK_M
        start_n = tile_n * BLOCK_N

        # TODO TMA P1: 현재 C tile의 FP32 accumulator를 0으로 만든다.
        # shape은 [BLOCK_M, BLOCK_N]. 반드시 persistent loop 안에서 초기화한다.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start_k in tl.range(0, K, BLOCK_K, warp_specialize=False):
            # A[M,K], B[K,N]에서 현재 타일을 읽는다.
            a_tile = a_desc.load([start_m, start_k])
            b_tile = b_desc.load([start_k, start_n])
            # TODO TMA P2: tl.dot(a_tile, b_tile, acc)로 누적한다.
            # a_tile[BLOCK_M, BLOCK_K] @ b_tile[BLOCK_K, BLOCK_N].
            acc = tl.dot(a_tile, b_tile, acc)

        # TODO TMA P3: K loop가 끝나면 FP16으로 변환하고 현재 C tile을 저장한다.
        # store는 persistent loop 안, K loop 밖에 있어야 한다.
        c_tile = acc.to(tl.float16)
        c_desc.store([start_m, start_n], c_tile)


def tma_persistent_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    print_metadata: bool = True,
) -> torch.Tensor:
    """A[M,K] @ B[K,N]. TMA tiled와 같은 입력, persistent scheduling."""
    assert a.is_cuda and b.is_cuda and a.device == b.device
    assert torch.version.cuda is not None, "NVIDIA CUDA 환경이 필요합니다."
    assert torch.cuda.get_device_capability(a.device)[0] >= 9, "TMA 지원 GPU가 필요합니다."
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype == torch.float16
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b and min(M, N, K) > 0
    assert K % 8 == 0 and N % 8 == 0, "첫 실습에서는 K/N을 8의 배수로 사용합니다."
    assert a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Descriptor는 전체 tensor를 기술한다. tile마다 다시 만들지 않는다.
    a_desc = TensorDescriptor(
        base=a, shape=[M, K], strides=list(a.stride()),
        block_shape=[TMA_BLOCK_M, TMA_BLOCK_K],
    )
    b_desc = TensorDescriptor(
        base=b, shape=[K, N], strides=list(b.stride()),
        block_shape=[TMA_BLOCK_K, TMA_BLOCK_N],
    )
    c_desc = TensorDescriptor(
        base=c, shape=[M, N], strides=list(c.stride()),
        block_shape=[TMA_BLOCK_M, TMA_BLOCK_N],
    )

    num_tiles = triton.cdiv(M, TMA_BLOCK_M) * triton.cdiv(N, TMA_BLOCK_N)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_programs = min(num_sms * 4, num_tiles)
    grid = (num_programs,)

    # grid 크기와 kernel의 NUM_PROGRAMS는 같은 값이어야 한다.

    kernel = _tma_persistent_matmul_kernel[grid](
        a_desc, b_desc, c_desc, M, N, K,
        NUM_PROGRAMS=num_programs,
        BLOCK_M=TMA_BLOCK_M, BLOCK_N=TMA_BLOCK_N, BLOCK_K=TMA_BLOCK_K,
        num_warps=4, num_stages=2,
    )
    if print_metadata:
        _print_kernel_metadata_once(
            "TMA persistent",
            kernel,
            (
                f"BLOCK_M={TMA_BLOCK_M}, BLOCK_N={TMA_BLOCK_N}, BLOCK_K={TMA_BLOCK_K}, "
                f"NUM_PROGRAMS={num_programs}, num_stages={kernel.metadata.num_stages}, "
                "warp_specialize=False"
            ),
            (
                "TMA persistent", a.device, a.dtype, M, N, K,
                TMA_BLOCK_M, TMA_BLOCK_N, TMA_BLOCK_K, num_programs,
                kernel.metadata.num_warps, kernel.metadata.num_stages,
            ),
        )
    return c


# Warp specialization 실습: 같은 커널의 WS=False/True를 비교한다.
@triton.jit
def _tma_warp_specialized_matmul_kernel(
    a_desc, b_desc, c_desc,
    M, N, K,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    start_tile = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = tl.cdiv(M, BLOCK_M) * num_n_tiles

    # 공식 persistent 예제처럼 store 쪽 counter를 분리한다.
    # load/store가 같은 loop-carried 좌표를 공유하는 pipelining 제약을 피한다.
    tile_id_c = start_tile - NUM_PROGRAMS

    # TODO WS 1: 아래 warp_specialize=False를 WARP_SPECIALIZE로 바꾼다.
    # outer persistent loop에서 요청한다. inner K loop에는 중복 적용하지 않는다.
    # warp 역할 분리와 동기화는 컴파일러가 구성한다.
    for tile_id in tl.range(
        start_tile, num_tiles, NUM_PROGRAMS,
        flatten=True, warp_specialize=False,
    ):
        start_m = (tile_id // num_n_tiles) * BLOCK_M
        start_n = (tile_id % num_n_tiles) * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start_k in range(0, K, BLOCK_K):
            # TODO WS 2: A/B descriptor load와 tl.dot 누적을 작성한다.
            # A[M,K]: [start_m, start_k] -> [BLOCK_M, BLOCK_K]
            # B[K,N]: [start_k, start_n] -> [BLOCK_K, BLOCK_N]
            # 기존 입력 layout을 유지하므로 B tile을 전치하지 않는다.
            pass

        tile_id_c += NUM_PROGRAMS
        store_m = (tile_id_c // num_n_tiles) * BLOCK_M
        store_n = (tile_id_c % num_n_tiles) * BLOCK_N
        # TODO WS 3: acc를 FP16으로 바꾸고 [store_m, store_n]에 store한다.
        # store는 K loop 밖, persistent loop 안에서 타일마다 한 번 실행한다.


def tma_warp_specialized_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    warp_specialize: bool = True,
    print_metadata: bool = True,
) -> torch.Tensor:
    """TMA persistent WS 실습. TODO 완성 후 같은 설정의 False/True를 비교한다."""
    # TODO WS 4: WS 1~3을 구현한 뒤 이 raise를 삭제한다.
    # 미완성 output이 정확도 검사/벤치마크에 들어가지 않도록 pending으로 둔다.
    raise NotImplementedError("Warp specialization 실습: TODO WS 1~4를 구현하세요.")

    assert a.is_cuda and b.is_cuda and a.device == b.device
    assert torch.version.cuda is not None, "NVIDIA CUDA 환경이 필요합니다."
    major, _ = torch.cuda.get_device_capability(a.device)
    assert major >= 9, "TMA 지원 GPU가 필요합니다."
    if warp_specialize:
        assert major >= 10, "이 자동 WS 실습은 Blackwell GPU에서 진행하세요."
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype == torch.float16
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b and min(M, N, K) > 0
    assert K % 8 == 0 and N % 8 == 0, "K/N은 8의 배수로 사용합니다."
    assert a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    a_desc = TensorDescriptor(
        base=a, shape=[M, K], strides=list(a.stride()),
        block_shape=[TMA_BLOCK_M, TMA_BLOCK_K],
    )
    b_desc = TensorDescriptor(
        base=b, shape=[K, N], strides=list(b.stride()),
        block_shape=[TMA_BLOCK_K, TMA_BLOCK_N],
    )
    c_desc = TensorDescriptor(
        base=c, shape=[M, N], strides=list(c.stride()),
        block_shape=[TMA_BLOCK_M, TMA_BLOCK_N],
    )
    num_tiles = triton.cdiv(M, TMA_BLOCK_M) * triton.cdiv(N, TMA_BLOCK_N)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_programs = min(num_sms * 4, num_tiles)

    kernel = _tma_warp_specialized_matmul_kernel[(num_programs,)](
        a_desc, b_desc, c_desc, M, N, K,
        NUM_PROGRAMS=num_programs,
        BLOCK_M=TMA_BLOCK_M, BLOCK_N=TMA_BLOCK_N, BLOCK_K=TMA_BLOCK_K,
        WARP_SPECIALIZE=warp_specialize,
        num_warps=4, num_stages=2,
    )
    if print_metadata:
        _print_kernel_metadata_once(
            "TMA WS practice",
            kernel,
            (
                f"BLOCK_M={TMA_BLOCK_M}, BLOCK_N={TMA_BLOCK_N}, BLOCK_K={TMA_BLOCK_K}, "
                f"NUM_PROGRAMS={num_programs}, num_stages={kernel.metadata.num_stages}, "
                f"warp_specialize={warp_specialize}"
            ),
            (
                "TMA WS practice", a.device, a.dtype, M, N, K, warp_specialize,
                TMA_BLOCK_M, TMA_BLOCK_N, TMA_BLOCK_K, num_programs,
                kernel.metadata.num_warps, kernel.metadata.num_stages,
            ),
        )
    return c
