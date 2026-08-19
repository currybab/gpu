import torch
import triton

from group_gemm import group_gemm


def total_tflops(shapes, milliseconds):
    flops = sum(2 * M * N * K for M, N, K in shapes)
    return flops / (milliseconds * 1e9)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    shapes = tuple((M, 1024, 1024) for M in (128, 256, 512, 1024))
    group_a = [
        torch.randn((M, K), device="cuda", dtype=torch.float16)
        for M, _, K in shapes
    ]
    group_b = [
        torch.randn((K, N), device="cuda", dtype=torch.float16)
        for _, N, K in shapes
    ]

    implementations = (
        (
            "torch-loop",
            lambda: [torch.matmul(a, b) for a, b in zip(group_a, group_b)],
        ),
        ("group-gemm-e2e", lambda: group_gemm(group_a, group_b)),
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
            f"{total_tflops(shapes, milliseconds):.2f} aggregate TFLOPS"
        )


if __name__ == "__main__":
    main()
