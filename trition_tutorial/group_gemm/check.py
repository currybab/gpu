import torch

from group_gemm import group_gemm


@torch.no_grad()
def check_group_gemm():
    torch.manual_seed(0)
    shapes = (
        (128, 128, 64),
        (256, 128, 128),
        (65, 129, 33),
        (193, 71, 160),
    )
    group_a = [
        torch.randn((M, K), device="cuda", dtype=torch.float16)
        for M, _, K in shapes
    ]
    group_b = [
        torch.randn((K, N), device="cuda", dtype=torch.float16)
        for _, N, K in shapes
    ]

    expected = [torch.matmul(a, b) for a, b in zip(group_a, group_b)]
    actual = group_gemm(group_a, group_b)
    assert len(actual) == len(expected)

    for index, (want, got) in enumerate(zip(expected, actual)):
        error = (got - want).abs().max().item()
        torch.testing.assert_close(got, want, atol=2e-2, rtol=1e-2)
        print(
            f"group GEMM passed: problem={index}, shape={shapes[index]}, "
            f"max_abs_error={error:.6f}"
        )


if __name__ == "__main__":
    check_group_gemm()
