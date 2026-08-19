import torch
import torch.nn.functional as F
import triton

from check import check_attention
from fused_attention import attention


def attention_tflops(batch, n_heads, n_ctx, head_dim, causal, milliseconds):
    # QK^T와 PV 두 matmul. causal은 유효 score 영역이 대략 절반이다.
    flops = 4 * batch * n_heads * n_ctx * n_ctx * head_dim
    if causal:
        flops *= 0.5
    return flops / (milliseconds * 1e9)


@torch.no_grad()
def main():
    # 잘못된 kernel을 의미 없는 속도로 측정하지 않도록 correctness를 먼저 확인한다.
    check_attention(attention)

    torch.manual_seed(0)
    batch, n_heads, n_ctx, head_dim = 4, 16, 2048, 64
    shape = (batch, n_heads, n_ctx, head_dim)
    sm_scale = head_dim**-0.5
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    for causal in (False, True):
        implementations = (
            (
                "torch-sdpa",
                lambda: F.scaled_dot_product_attention(
                    q, k, v, is_causal=causal, scale=sm_scale
                ),
            ),
            (
                "triton",
                lambda: attention(q, k, v, causal=causal, sm_scale=sm_scale),
            ),
        )
        for name, fn in implementations:
            fn()
            milliseconds = triton.testing.do_bench(fn)
            print(
                f"{name}: causal={causal}, {milliseconds:.4f} ms, "
                f"{attention_tflops(*shape, causal, milliseconds):.2f} TFLOPS"
            )


if __name__ == "__main__":
    main()
