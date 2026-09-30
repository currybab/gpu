"""TMA tiled/persistent/WS 정확도 검사. 미구현은 각각 pending으로 표시한다."""

from functools import partial

import torch

from persistent_matmul import (
    TMA_BLOCK_M, TMA_BLOCK_N,
    tma_matmul, tma_persistent_matmul, tma_warp_specialized_matmul,
)


@torch.no_grad()
def check_one(name, fn, shapes):
    torch.manual_seed(0)
    for M, N, K in shapes:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        actual = fn(a, b)
        expected = a @ b
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)
        error = (actual - expected).abs().max().item()
        print(f"{name} passed: M={M}, N={N}, K={K}, max_abs_error={error:.6f}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU에서 실행하세요.")
    print(f"GPU: {torch.cuda.get_device_name()}")
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    shapes = (
        (TMA_BLOCK_M, TMA_BLOCK_N, 64),  # full C tile 하나
        (128, 192, 128),    # 여러 output tile과 K iteration
        (13, 24, 16),       # 세 축 모두 tile보다 작음
        (257, 200, 104),    # M/N/K tail, row stride 정렬은 유지
        (1024, 1024, 512),
        # grid가 4*SM이므로 8*SM+1개 tile: 모든 program이 여러 tile을 맡고
        # 마지막 program별 iteration 수가 다르며 M/N/K tail도 존재한다.
        (8 * num_sms * TMA_BLOCK_M + 1, TMA_BLOCK_N - 8, 104),
    )
    for name, fn in (
        ("TMA tiled", tma_matmul),
        ("TMA persistent", tma_persistent_matmul),
        ("TMA WS practice (False)", partial(tma_warp_specialized_matmul, warp_specialize=False)),
        ("TMA WS practice (True)", partial(tma_warp_specialized_matmul, warp_specialize=True)),
    ):
        try:
            check_one(name, fn, shapes)
        except NotImplementedError as error:
            print(f"{name} pending: {error}")


if __name__ == "__main__":
    main()
