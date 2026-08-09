import torch
import triton
import triton.language as tl

@triton.jit
def mean_var_kernel(x_ptr, mean_ptr, var_ptr, M, N, stride_xm, stride_xn, BLOCK_SIZE_N: tl.constexpr):
    # TODO
    pid_m = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < N
    x_row = x_ptr + pid_m * stride_xm
    x_vals = tl.load(x_row + offsets * stride_xn, mask=mask, other=0.0)
    x_mean = tl.sum(x_vals) / N
    tl.store(mean_ptr + pid_m, x_mean)
    x_move = tl.where(mask, x_vals - x_mean, 0.0)
    x_var =  tl.sum(x_move * x_move) / N
    tl.store(var_ptr + pid_m, x_var)    

def mean_var(x: torch.Tensor):
    # TODO
    M, N = x.shape
    stride_xm, stride_xn = x.stride()
    mean = torch.empty((M,), dtype=x.dtype, device=x.device)
    var = torch.empty((M,), dtype=x.dtype, device=x.device)
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    grid = (M,)
    mean_var_kernel[grid](x, mean, var, M, N, stride_xm, stride_xn, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return mean, var


torch.manual_seed(0)
x = torch.randn(16, 200, device='cuda')
m_ref = x.mean(dim=1); v_ref = x.var(dim=1, unbiased=False)
m, v = mean_var(x)
print(torch.allclose(m, m_ref, atol=1e-5), torch.allclose(v, v_ref, atol=1e-4))
