import torch

from persistent_matmul import matmul, persistent_matmul


@torch.no_grad()
def check_one(name, fn):
    torch.manual_seed(0)
    for M, N, K in ((128, 128, 64), (257, 193, 97), (512, 384, 160)):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        expected = torch.matmul(a, b)
        actual = fn(a, b)
        error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)
        print(f"{name} passed: M={M}, N={N}, K={K}, max_abs_error={error:.6f}")


if __name__ == "__main__":
    for implementation_name, implementation in (
        ("basic", matmul),
        ("persistent", persistent_matmul),
    ):
        try:
            check_one(implementation_name, implementation)
        except NotImplementedError as error:
            print(f"{implementation_name} pending: {error}")
