import torch
import triton
import triton.language as tl

BLOCK_SIZE = 256

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # TODO: standard tiling preamble, then load/add/store with mask
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x_split = tl.load(x_ptr + offset, mask=mask)
    y_split = tl.load(y_ptr + offset, mask=mask)
    add_split = x_split + y_split
    tl.store(output_ptr + offset, add_split, mask=mask)
    

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # TODO: allocate output, compute grid, launch kernel, return output
    output = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE = BLOCK_SIZE)
    return output


x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
y = torch.tensor([10.0, 20.0, 30.0], device='cuda')
print(add(x, y).cpu().tolist())
