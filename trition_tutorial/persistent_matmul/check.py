import torch

from persistent_matmul import matmul, persistent_matmul, swizzle_matmul, persistent_swizzle_matmul


@torch.no_grad()
def check_one(name, fn):
    torch.manual_seed(0)
    for M, N, K, padding in (
        (128, 128, 64, 0),
        (257, 193, 97, 7),
        (512, 384, 160, 0),
        (1024, 1024, 512, 0),
        (1153, 193, 97, 7),  # M tile 10개: GROUP_SIZE_M=8의 마지막 그룹은 2개
        (4096, 1024, 64, 0),  # 여러 그룹 + persistent program의 반복 처리
    ):
        a = torch.randn((M, K + padding), device="cuda", dtype=torch.float16)[:, :K]
        b = torch.randn((K, N + padding), device="cuda", dtype=torch.float16)[:, :N]
        expected = torch.matmul(a, b)
        actual = fn(a, b)
        error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)
        print(
            f"{name} passed: M={M}, N={N}, K={K}, padding={padding}, "
            f"max_abs_error={error:.6f}"
        )


if __name__ == "__main__":
    for implementation_name, implementation in (
        ("naive tiled", matmul),
        ("naive persistent", persistent_matmul),
        ("swizzle tiled (group=1)", lambda a, b: swizzle_matmul(a, b, group_size_m=1)),
        ("swizzle tiled (group=8)", lambda a, b: swizzle_matmul(a, b, group_size_m=8)),
        ("swizzle persistent (group=1)", lambda a, b: persistent_swizzle_matmul(a, b, group_size_m=1)),
        ("swizzle persistent (group=8)", lambda a, b: persistent_swizzle_matmul(a, b, group_size_m=8)),
    ):
        try:
            check_one(implementation_name, implementation)
        except NotImplementedError as error:
            print(f"{implementation_name} pending: {error}")
