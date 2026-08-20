import torch

from block_scaled_matmul import block_scaled_matmul


def block_scaled_reference(a, b, a_scale, b_scale, vec_size: int):
    """scale 하나가 K축 vec_size개 원소에 broadcast되는 의미론 reference."""
    scale_a = a_scale.repeat_interleave(vec_size, dim=1)
    scale_b = b_scale.repeat_interleave(vec_size, dim=0)
    dequant_a = a.float() * scale_a
    dequant_b = b.float() * scale_b
    return torch.matmul(dequant_a, dequant_b)


@torch.no_grad()
def check_block_scaled_matmul():
    torch.manual_seed(0)
    for M, N, K, vec_size, padding in (
        (128, 96, 144, 16, 0),
        (67, 75, 160, 32, 5),
    ):
        a = (
            torch.randn((M, K + padding), device="cuda", dtype=torch.float16) * 0.25
        )[:, :K]
        b = (
            torch.randn((K, N + padding), device="cuda", dtype=torch.float16) * 0.25
        )[:, :N]
        scale_k = K // vec_size
        a_scale = (torch.rand((M, scale_k + padding), device="cuda") + 0.5)[
            :, :scale_k
        ]
        b_scale = (torch.rand((scale_k, N + padding), device="cuda") + 0.5)[
            :, :N
        ]

        expected = block_scaled_reference(a, b, a_scale, b_scale, vec_size)
        actual = block_scaled_matmul(a, b, a_scale, b_scale, vec_size)
        error = (actual.float() - expected).abs().max().item()
        torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=1e-2)
        print(
            f"block-scaled matmul passed: M={M}, N={N}, K={K}, "
            f"VEC_SIZE={vec_size}, padding={padding}, max_abs_error={error:.6f}"
        )


if __name__ == "__main__":
    check_block_scaled_matmul()
