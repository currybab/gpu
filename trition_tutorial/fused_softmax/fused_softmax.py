import os

import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = driver.active.get_active_torch_device()
DIAGNOSTICS = os.environ.get("GPU_LAB_DIAGNOSTICS") == "1"


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols  # n_cols <= BLOCK_SIZE 라는 가정이 들어 있음.

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        input_ptrs = row_start_ptr + col_offsets # 담당할 입력들
        row = tl.load(input_ptrs, mask=mask, other=-float('inf')) # mask가 False인 자리는 메모리를 건드리지 않고 대신 -inf를 채워 넣음.

        row_minus_max = row - tl.max(row) # -inf가 포함되어 있어도 정상적으로 작동함. -inf가 max 연산의 항등원임.
        numerator = tl.exp(row_minus_max) # 개별 항들
        denominator = tl.sum(numerator) # 분모
        softmax_output = numerator / denominator # 소프트맥스 결과

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}
reported_kernel_configs = set()


def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols) # n_cols 보다 크거나 같은 최소의 2의 거듭제곱으로 지정.
    num_warps = 8 # BLOCK_SIZE가 n_cols에 의해 강제되므로, 행 하나에 더 많은 스레드를 배정하기 위해 사용할 수 있음.

    # SMEM의 크기에 따른 software pipelining stages 수 조정
    num_stages = 1 # 4 if SIZE_SMEM > 200000 else 2
    # print(f"num_stages: {num_stages}, SIZE_SMEM: {SIZE_SMEM}")

    # 띄울 프로그램 수를 정하기 위해 커널이 레지스터/smem을 얼마난 쓰는 지 알아야하는데 컴파일 해봐야 앎.
    # 먼저 컴파일만 -> 자원 사용량 조회 -> grid 결정 -> 실행
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, )) 
                                   # grid는 형식상 필요한 더미로 컴파일 결과에 영향을 주지 않음. 
                                   # x,y도 데이터를 읽는 게 아니라, dtype/포인터 타입 정보만 뽑는 용도임.
    kernel._init_handles() # n_regs는 컴파일러가 계산한 값이 아니라 드라이버가 로드된 함수를 보고 알려주는 값이라, 모듈이 올라가 있지 않으면 읽을 수가 없음. _init_handles()가 그 로딩을 강제.
    n_regs = kernel.n_regs # 스레드 1개당 레지스터 수
    n_spills = kernel.n_spills
    size_smem = kernel.metadata.shared # 프로그램 1개당 shared memory (bytes)

    register_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    shared_memory_occupancy = SIZE_SMEM // size_smem if size_smem else None
    occupancy = register_occupancy
    if shared_memory_occupancy is not None:
        occupancy = min(occupancy, shared_memory_occupancy)

    config = (str(x.dtype), BLOCK_SIZE, num_stages, num_warps)
    if DIAGNOSTICS and config not in reported_kernel_configs:
        reported_kernel_configs.add(config)
        print(
            "[Triton kernel] "
            f"dtype={x.dtype}, BLOCK_SIZE={BLOCK_SIZE}, "
            f"num_stages={num_stages}, num_warps={num_warps}"
        )
        print(
            f"  n_regs={n_regs}, n_spills={n_spills}, "
            f"shared_memory={size_smem} bytes"
        )
        print(
            "  programs/SM: "
            f"register_limit={register_occupancy}, "
            f"shared_memory_limit={shared_memory_occupancy or 'unlimited'}, "
            f"selected={occupancy}"
        )
    
    num_programs = NUM_SM * occupancy # SM 수와 occupancy를 곱해 전체 프로그램 수 계산
    num_programs = min(num_programs, n_rows) # 프로그램 수를 행 수와 비교해 최소값으로 제한

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y


def profile_torch_softmax(x: torch.Tensor) -> None:
    """Torch softmax가 실제로 띄우는 CUDA 커널을 한 번 기록한다."""
    # CUDA context 초기화와 최초 라이브러리 준비 비용은 profile에서 제외한다.
    for _ in range(5):
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    print("[Torch softmax CUDA profiler]")
    print(
        profiler.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30,
        )
    )


torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
print("Softmax test passed!")

if DIAGNOSTICS:
    profile_torch_softmax(x)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton",
            "torch",
            "naive_softmax",
        ],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == "naive_softmax":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)

# RTX 5090 (이론 최대 1792 GB/s (약 1.79 TB/s))
# run1 (range, num_stage=2)     Triton 1457.1 / Torch 1502.8
# run2 (tl.range, num_stage=2)  Triton 1411.0 / Torch 1502.8
# run3 (tl.range, num_stages=1)  Triton 1457.6 / Torch 1502.2
# run4 (range, num_stage=1)  Triton 1457.4 / Torch 1502.7
# tutorial code인 일반 range를 적용할 경우 software pipelining이 동작하지 않음.
