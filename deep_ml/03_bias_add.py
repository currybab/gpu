import torch
import triton
import triton.language as tl

BLOCK_SIZE_N = 256

@triton.jit
def bias_add_kernel(x_ptr, b_ptr, output_ptr, M, N, stride_xm, stride_xn,
                    BLOCK_SIZE_N: tl.constexpr):
    # TODO: 2-D grid; compute row pointer offset and column-block offsets
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N
    x_vals = tl.load(x_ptr + pid_m * stride_xm + offsets * stride_xn, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    output_vals = x_vals + b_vals
    tl.store(output_ptr + pid_m * stride_xm + offsets * stride_xn, output_vals, mask=mask)

def bias_add(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO
    M, N = x.shape
    stride_xm, stride_xn = x.stride()
    grid = (M, triton.cdiv(N, BLOCK_SIZE_N))
    output = torch.empty_like(x)
    bias_add_kernel[grid](x, b, output, M, N, stride_xm, stride_xn, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return output


torch.manual_seed(0)
x = torch.randn(64, 300, device='cuda')
b = torch.randn(300, device='cuda')
ref = x + b
out = bias_add(x, b)
print(torch.allclose(out, ref, atol=1e-5))
