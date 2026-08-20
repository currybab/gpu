import torch

from fused_attention import attention


def attention_reference(q, k, v, causal: bool, sm_scale: float):
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if causal:
        n = q.shape[-2]
        mask = torch.ones((n, n), device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


@torch.no_grad()
def check_attention(attention_fn):
    """기본 tile, sequence tail, head-dimension padding을 reference와 비교한다."""
    torch.manual_seed(0)

    cases = (
        (1, 2, 128, 64, 0),  # 기본 tile
        (2, 3, 129, 64, 8),  # sequence tail + padded row stride
        (1, 2, 129, 80, 0),  # BLOCK_D padding
    )
    for batch, n_heads, n_ctx, head_dim, padding in cases:
        shape = (batch, n_heads, n_ctx, head_dim)
        storage_shape = (batch, n_heads, n_ctx, head_dim + padding)
        sm_scale = head_dim**-0.5
        q = torch.randn(storage_shape, device="cuda", dtype=torch.float16)[..., :head_dim]
        k = torch.randn(storage_shape, device="cuda", dtype=torch.float16)[..., :head_dim]
        v = torch.randn(storage_shape, device="cuda", dtype=torch.float16)[..., :head_dim]

        for causal in (False, True):
            expected = attention_reference(q, k, v, causal, sm_scale)
            actual = attention_fn(q, k, v, causal, sm_scale)
            max_abs_error = (actual - expected).abs().max().item()

            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=0)
            print(
                f"attention forward passed: shape={shape}, padding={padding}, causal={causal}, "
                f"max_abs_error={max_abs_error:.6f}"
            )


if __name__ == "__main__":
    check_attention(attention)
