import torch
import triton

from persistent_matmul import matmul, persistent_matmul, swizzle_matmul, persistent_swizzle_matmul


def tflops(M, N, K, milliseconds):
    return 2 * M * N * K / (milliseconds * 1e9)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    M, N, K = 16384, 16384, 4096
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    print(f"M={M}, N={N}, K={K}")
    
    implementations = (
        ("torch", lambda: torch.matmul(a, b)),
        ("naive tiled", lambda: matmul(a, b)),
        ("naive persistent", lambda: persistent_matmul(a, b, programs_per_sm=2)),
        ("swizzle tiled", lambda: swizzle_matmul(a, b, group_size_m=8)),
        ("swizzle persistent", lambda: persistent_swizzle_matmul(a, b, group_size_m=8, programs_per_sm=2)),
    )
    for name, fn in implementations:
        try:
            fn()
            milliseconds = triton.testing.do_bench(fn)
        except NotImplementedError as error:
            print(f"{name} pending: {error}")
            continue
        print(
            f"{name}: {milliseconds:.4f} ms, "
            f"{tflops(M, N, K, milliseconds):.2f} TFLOPS"
        )


if __name__ == "__main__":
    main()
