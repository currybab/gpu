import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, M, N, stride_xm, stride_xn,
                   BLOCK_SIZE_N: tl.constexpr):
    # TODO
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N 
    row = tl.load(input_ptr + pid * stride_xm + offsets * stride_xn, mask=mask, other=-float('inf'))
    row_max = tl.max(row)
    row_exp = tl.exp(row - row_max)
    row_sum = tl.sum(row_exp)
    output_vals = row_exp / row_sum
    tl.store(output_ptr + pid * stride_xm + offsets * stride_xn, output_vals, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    # TODO
    output = torch.empty_like(x)
    M, N = x.shape
    stride_xm, stride_xn = x.stride()
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](output, x, M, N, stride_xm, stride_xn, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return output


torch.manual_seed(0)
x = torch.randn(64, 1000, device='cuda')
ref = torch.softmax(x, dim=1)
out = softmax(x)
print(torch.allclose(out, ref, atol=1e-5))
