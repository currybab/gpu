import argparse

import torch
import torch.nn.functional as F
import triton

from check import check_attention, selected_names
from implementations import IMPLEMENTATIONS, unsupported_reason


def attention_tflops(batch, n_heads, n_ctx, head_dim, causal, milliseconds):
    # QK^T와 PV 두 matmul. causal은 유효 score 영역이 대략 절반이다.
    flops = 4 * batch * n_heads * n_ctx * n_ctx * head_dim
    if causal:
        flops *= 0.5
    return flops / (milliseconds * 1e9)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="검증한 forward 변형과 Torch SDPA 비교")
    parser.add_argument("--impl", nargs="+", choices=[*IMPLEMENTATIONS, "all"],
                        default=["baseline"])
    args = parser.parse_args()
    verified = []
    for name in selected_names(args):
        reason = unsupported_reason(name)
        if reason:
            print(f"SKIP {name}: {reason}")
            continue
        try:
            check_attention(IMPLEMENTATIONS[name], name)
        except NotImplementedError as exc:
            print(f"TODO {name}: {exc}")
            continue
        verified.append(name)
    if not verified:
        print("측정할 검증 완료 구현이 없습니다.")
        return 1

    torch.manual_seed(0)
    batch, n_heads, n_ctx, head_dim = 4, 16, 2048, 64
    shape = (batch, n_heads, n_ctx, head_dim)
    sm_scale = head_dim**-0.5
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    for causal in (False, True):
        implementations = [
            (
                "torch-sdpa",
                lambda: F.scaled_dot_product_attention(
                    q, k, v, is_causal=causal, scale=sm_scale
                ),
            ),
        ]
        implementations.extend(
            (name, lambda fn=IMPLEMENTATIONS[name]: fn(q, k, v, causal, sm_scale))
            for name in verified
        )
        for name, fn in implementations:
            # 작은 case뿐 아니라 실제 측정 shape도 검증. 첫 호출의 autotune 제외.
            actual = fn()
            expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                                      scale=sm_scale)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=0)
            milliseconds = triton.testing.do_bench(fn)
            print(
                f"{name} forward (wrapper 포함): causal={causal}, {milliseconds:.4f} ms, "
                f"{attention_tflops(*shape, causal, milliseconds):.2f} TFLOPS"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
