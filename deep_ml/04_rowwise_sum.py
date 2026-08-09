import torch
import triton
import triton.language as tl

# You may assume N <= 16384.
# 이게 N 방향의 loop는 안써도 되게 해주는 조건임.
# num_warps=4(128 스레드) 기준 N=16384면 스레드당 16384/128 = 128 레지스터임. 
# 좀만 늘어도 스레드당 256 레지스터를 사용하게 됨. 그래서 스레드당 255개 레지스터 제한에 걸림.
# 즉 16384는 단일 타일 구조가 성립하는 마지막 크기

# 레지스터 관련 주요 제약(CUDA)
# SM당 레지스터 파일 크기: 256KB
# 스레드 블록당 최대: 256KB
# 스레드당 최대: 255개 레지스터

@triton.jit
def row_sum_kernel(x_ptr, output_ptr, M, N, stride_xm, stride_xn, BLOCK_SIZE_N: tl.constexpr):
    # TODO
    pid_m = tl.program_id(0)
    x_row = x_ptr + pid_m * stride_xm
    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N
    x_vals = tl.load(x_row + offsets * stride_xn, mask=mask, other=0.0)
    x_sum = tl.sum(x_vals)
    tl.store(output_ptr + pid_m, x_sum)


def row_sum(x: torch.Tensor) -> torch.Tensor:
    # TODO
    M, N = x.shape
    stride_xm, stride_xn = x.stride()
    output = torch.empty((M,), dtype=x.dtype, device=x.device)
    grid = (M,)
    row_sum_kernel[grid](x, output, M, N, stride_xm, stride_xn, BLOCK_SIZE_N=triton.next_power_of_2(N))
    return output


torch.manual_seed(0)
x = torch.randn(32, 100, device='cuda')
ref = x.sum(dim=1)
out = row_sum(x)
print(torch.allclose(out, ref, atol=1e-4))
