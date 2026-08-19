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
    """구현한 attention의 작은 FP16 forward 결과를 reference와 비교한다."""
    torch.manual_seed(0)

    batch, n_heads, n_ctx, head_dim = 1, 2, 128, 64
    shape = (batch, n_heads, n_ctx, head_dim)
    sm_scale = head_dim**-0.5

    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    for causal in (False, True):
        expected = attention_reference(q, k, v, causal, sm_scale)
        actual = attention_fn(q, k, v, causal, sm_scale)
        max_abs_error = (actual - expected).abs().max().item()

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=0)
        print(
            f"attention forward passed: causal={causal}, "
            f"max_abs_error={max_abs_error:.6f}"
        )


if __name__ == "__main__":
    check_attention(attention)
