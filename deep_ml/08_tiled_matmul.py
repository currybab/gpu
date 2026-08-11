import torch
import triton
import triton.language as tl

BLOCK_M = BLOCK_N = BLOCK_K = 32

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    # TODO: tiled matmul with float32 accumulator
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offsets_m[:, None] < M
    mask_n = offsets_n[None, :] < N
    a_row = a_ptr + offsets_m[:, None] * stride_am 
    b_col = b_ptr + offsets_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(tl.cdiv(K, BLOCK_K)):
        offsets_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_a = mask_m & (offsets_k[None, :] < K)
        mask_b = (offsets_k[:, None] < K) & mask_n
        tile_a = tl.load(a_row + offsets_k[None, :] * stride_ak, mask=mask_a, other=0.0)
        tile_b = tl.load(b_col + offsets_k[:, None] * stride_bk, mask=mask_b, other=0.0)
        acc += tl.dot(tile_a, tile_b)
    
    tl.store(c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn, acc, mask=mask_m&mask_n)
    

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO
    M, K = a.shape
    _, N = b.shape
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stride_cm, stride_cn = c.stride()
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, c, M, N, K, 
                        stride_am, stride_ak, stride_bk, stride_bn,
                        stride_cm, stride_cn,
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
                        )
    return c



torch.manual_seed(0)
a = torch.randn(64, 96, device='cuda')
b = torch.randn(96, 32, device='cuda')
out = matmul(a, b)
ref = a @ b
print(torch.allclose(out, ref, atol=1e-3))
