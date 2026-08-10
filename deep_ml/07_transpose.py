import torch
import triton
import triton.language as tl

BLOCK_M = BLOCK_N = 32

@triton.jit
def transpose_kernel(x_ptr, out_ptr, M, N, stride_xm, stride_xn,
                     stride_om, stride_on,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # TODO: compute 2-D tile offsets, load, transpose, store
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)

    x_tile = tl.load(x_ptr + offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn, mask=mask) # (m, n)
    tl.store(out_ptr + offsets_n[None, :] * stride_om + offsets_m[:, None] * stride_on, x_tile, mask=mask) #(n, m)

def transpose(x: torch.Tensor) -> torch.Tensor:
    # TODO
    M, N = x.shape
    stride_xm, stride_xn = x.stride()
    output = torch.empty((N, M), dtype=x.dtype, device=x.device)
    stride_om, stride_on = output.stride()
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](x, output, M, N, stride_xm, stride_xn, stride_om, stride_on, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return output


torch.manual_seed(0)
x = torch.randn(64, 96, device='cuda')
out = transpose(x)
ref = x.T.contiguous()
print(out.shape, torch.allclose(out, ref))
