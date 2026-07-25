#!POPCORN leaderboard vectoradd_v2
#!POPCORN gpu B200

from task import input_t, output_t
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    assert A.device == B.device == output.device
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](A, B, output, n_elements, BLOCK_SIZE=2048, num_warps=8)
    return output

# A100 (num_warps=4)
# BLOCK_SIZE = 512, 947.200μs
# BLOCK_SIZE = 1024, 896.683μs 
# BLOCK_SIZE = 2048, 950.955μs

# B200
# BLOCK_SIZE = 1024, num_warps=4, 235.139μs
# BLOCK_SIZE = 2048, num_warps=8, 235.804μs
