import torch
import triton
import triton.language as tl

BLOCK_SIZE = 256

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # TODO
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_vals = tl.load(x_ptr + offsets, mask=mask)
    output_vals = tl.maximum(x_vals, 0.0)
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def relu(x: torch.Tensor) -> torch.Tensor:
    # TODO
    n_elements = x.numel()
    output = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


torch.manual_seed(0)
x = torch.randn(4096, device='cuda')
ref = torch.relu(x)
out = relu(x)
print(torch.allclose(out, ref, atol=1e-6))
