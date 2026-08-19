import torch
import triton

from persistent_matmul import matmul, persistent_matmul


def tflops(M, N, K, milliseconds):
    return 2 * M * N * K / (milliseconds * 1e9)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    M, N, K = 4096, 4096, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    implementations = (
        ("torch", lambda: torch.matmul(a, b)),
        ("basic", lambda: matmul(a, b)),
        ("persistent", lambda: persistent_matmul(a, b)),
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
