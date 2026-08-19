import torch
import triton

from block_scaled_matmul import block_scaled_matmul
from check import block_scaled_reference


def tflops(M, N, K, milliseconds):
    return 2 * M * N * K / (milliseconds * 1e9)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    M, N, K, vec_size = 2048, 2048, 1024, 32
    a = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.25
    b = torch.randn((K, N), device="cuda", dtype=torch.float16) * 0.25
    a_scale = torch.rand((M, K // vec_size), device="cuda") + 0.5
    b_scale = torch.rand((K // vec_size, N), device="cuda") + 0.5

    implementations = (
        (
            "torch-explicit",
            lambda: block_scaled_reference(a, b, a_scale, b_scale, vec_size),
        ),
        (
            "triton-fused",
            lambda: block_scaled_matmul(a, b, a_scale, b_scale, vec_size),
        ),
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
