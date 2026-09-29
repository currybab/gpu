import torch
import triton

from persistent_matmul import matmul, persistent_matmul, swizzle_matmul, persistent_swizzle_matmul
from persistent_matmul import cross_tile_pipelining_matmul, tma_matmul, tma_persistent_matmul


def tflops(M, N, K, milliseconds):
    return 2 * M * N * K / (milliseconds * 1e9)


@torch.no_grad()
def main(M=16384, N=16384, K=4096):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    # TMA 입력 준비는 측정 밖에서 한 번만 수행한다.
    b_t = b.T.contiguous()
    expected = torch.matmul(a, b)
    print(f"M={M}, N={N}, K={K}")
    print("TMA: B.T.contiguous() 준비 비용 제외, warp_specialize=False")
    
    implementations = (
        ("torch", lambda: torch.matmul(a, b)),
        # ("naive tiled", lambda: matmul(a, b)),
        # ("naive persistent", lambda: persistent_matmul(a, b, programs_per_sm=2)),
        # ("swizzle tiled", lambda: swizzle_matmul(a, b, group_size_m=8)),
        # ("swizzle persistent", lambda: persistent_swizzle_matmul(a, b, group_size_m=8, programs_per_sm=2)),
        # ("cross-tile persistent", lambda: cross_tile_pipelining_matmul(a, b, group_size_m=8, programs_per_sm=2)),
        ("TMA tiled", lambda: tma_matmul(a, b_t)),
        ("TMA persistent", lambda: tma_persistent_matmul(a, b_t)),
    )
    for name, fn in implementations:
        try:
            actual = fn()  # 최초 compile/launch 및 정확도 검증은 측정 밖
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)
            del actual
            milliseconds = triton.testing.do_bench(fn)
        except NotImplementedError as error:
            print(f"{name} pending: {error}")
            continue
        print(
            f"{name}: {milliseconds:.4f} ms, "
            f"{tflops(M, N, K, milliseconds):.2f} TFLOPS"
        )


if __name__ == "__main__":
    main()
