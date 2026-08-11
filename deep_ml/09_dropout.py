import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024

@triton.jit
def dropout_kernel(x_ptr, output_ptr, n_elements, p, seed,
                   BLOCK_SIZE: tl.constexpr):
    # TODO: standard tiling preamble, draw rand, apply mask + scaling, store
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_mask = offsets < n_elements

    x_val = tl.load(x_ptr + offsets, mask=offset_mask, other=0.0)
    rand = tl.rand(seed, offsets)
    keep = rand > p
    output = tl.where(keep, x_val / (1.0 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=offset_mask)

def dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    # TODO
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    dropout_kernel[grid](x, output, n_elements, p, seed, BLOCK_SIZE=BLOCK_SIZE)
    return output

x = torch.randn(50000, device='cuda')
out1 = dropout(x, p=0.3, seed=42)
out2 = dropout(x, p=0.3, seed=42)
out3 = dropout(x, p=0.3, seed=99)
print(torch.equal(out1, out2), torch.equal(out1, out3))
