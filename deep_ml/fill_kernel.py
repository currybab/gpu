import torch
import triton
import triton.language as tl

DEFAULT_BLOCK_SIZE = 1024

@triton.jit
def fill_kernel(output_ptr, n, value, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n
    tl.store(output_ptr + offset, value, mask=mask)


def fill(n: int, value: float) -> torch.Tensor:
    output = torch.empty((n,), dtype=torch.float32, device="cuda")
    if n == 0:
        return output
    grid = (triton.cdiv(n, DEFAULT_BLOCK_SIZE),)
    fill_kernel[grid](output, n, value, BLOCK_SIZE=DEFAULT_BLOCK_SIZE)
    return output


out = fill(5, 3.14)
print([round(v, 2) for v in out.cpu().tolist()])
