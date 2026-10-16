import argparse

import torch

from implementations import IMPLEMENTATIONS, unsupported_reason


def attention_reference(q, k, v, causal: bool, sm_scale: float):
    # 작은 shape에서 FP32 eager 계산을 기준으로 삼는다 (SDPA dispatch와 독립).
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale

    if causal:
        n = q.shape[-2]
        mask = torch.ones((n, n), device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def cases_for(name):
    if name in ("tma", "tma-ws"):
        return ((1, 2, 128, 64, 0), (2, 3, 256, 128, 0), (1, 2, 1024, 64, 0))
    return (
        (1, 2, 17, 32, 0),  # 한 tile보다 짧은 sequence
        (1, 2, 128, 64, 0),
        (2, 3, 129, 64, 8 if name == "baseline" else 0),
        (1, 2, 129, 80, 0),  # sequence + head-dimension tail
        (1, 2, 1000, 128, 0),  # 여러 K/V tile
        (4, 16, 257, 64, 0),  # persistent program의 여러 query tile 처리
    )


@torch.no_grad()
def check_attention(attention_fn, name="baseline"):
    """기본 tile, sequence tail, head-dimension padding을 reference와 비교한다."""
    torch.manual_seed(0)

    for batch, n_heads, n_ctx, head_dim, padding in cases_for(name):
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
                f"{name} forward passed: shape={shape}, padding={padding}, causal={causal}, "
                f"max_abs_error={max_abs_error:.6f}"
            )


def selected_names(args):
    return list(IMPLEMENTATIONS) if "all" in args.impl else list(dict.fromkeys(args.impl))


def main():
    parser = argparse.ArgumentParser(description="Fused attention forward 정확도 검사")
    parser.add_argument("--impl", nargs="+", choices=[*IMPLEMENTATIONS, "all"],
                        default=["baseline"])
    args = parser.parse_args()
    passed, incomplete = 0, 0
    for name in selected_names(args):
        reason = unsupported_reason(name)
        if reason:
            print(f"SKIP {name}: {reason}")
            continue
        try:
            check_attention(IMPLEMENTATIONS[name], name)
        except NotImplementedError as exc:
            print(f"TODO {name}: {exc}")
            incomplete += 1
            continue
        passed += 1
    print(f"완료: {passed}개 구현 검증, {incomplete}개 미구현")
    # 미구현/전체 SKIP을 correctness 통과로 오인하지 않도록 nonzero exit.
    return 0 if passed and not incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
