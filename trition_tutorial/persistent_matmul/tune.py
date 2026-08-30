import torch
import triton

from persistent_matmul import persistent_matmul


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
CONFIGS = (
    # warp tile 64x32 계열
    (128, 128, 32, 8, 4),
    (256, 128, 32, 16, 3),
    (128, 64, 32, 4, 4),
    # warp tile 32x32 계열
    (128, 128, 32, 16, 4),
    (128, 64, 32, 8, 4),
    (64, 64, 32, 4, 4),
    # warp tile 32x16 계열
    (128, 64, 32, 16, 4),
    # BLOCK_K 실험
    (128, 128, 64, 8, 2),
    (128, 64, 64, 8, 3),
)

M = 4096
N = 4096
K = 4096
PROGRAMS_PER_SM = 2
MAXNREG = None

Config = tuple[int, int, int, int, int]


def tflops(milliseconds: float) -> float:
    return 2 * M * N * K / (milliseconds * 1e9)


def config_name(config: Config) -> str:
    block_m, block_n, block_k, num_warps, num_stages = config
    return (
        f"BM={block_m}, BN={block_n}, BK={block_k}, "
        f"warps={num_warps}, stages={num_stages}"
    )


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    expected = torch.matmul(a, b)

    torch_p20, torch_median, torch_p80 = triton.testing.do_bench(
        lambda: torch.matmul(a, b),
        quantiles=[0.2, 0.5, 0.8],
    )
    print(f"shape: M={M}, N={N}, K={K}")
    print(f"programs_per_sm={PROGRAMS_PER_SM}, maxnreg={MAXNREG}")
    print(
        f"torch: median={torch_median:.4f} ms "
        f"[p20={torch_p20:.4f}, p80={torch_p80:.4f}], "
        f"{tflops(torch_median):.2f} TFLOPS"
    )

    results: list[tuple[float, float, float, float, Config]] = []
    failures: list[tuple[Config, str]] = []

    for config in CONFIGS:
        block_m, block_n, block_k, num_warps, num_stages = config

        def run() -> torch.Tensor:
            return persistent_matmul(
                a,
                b,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                programs_per_sm=PROGRAMS_PER_SM,
                maxnreg=MAXNREG,
            )

        print(f"\ntrying: {config_name(config)}")
        try:
            actual = run()
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)
            p20, median, p80 = triton.testing.do_bench(
                run,
                quantiles=[0.2, 0.5, 0.8],
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            failures.append((config, message))
            print(f"skipped: {message}")
            continue

        throughput = tflops(median)
        results.append((median, p20, p80, throughput, config))
        print(
            f"result: median={median:.4f} ms "
            f"[p20={p20:.4f}, p80={p80:.4f}], "
            f"{throughput:.2f} TFLOPS"
        )

    results.sort(key=lambda result: result[0])
    print("\nranking")
    print("rank  median(ms)  p20(ms)  p80(ms)  TFLOPS   config")
    for rank, (median, p20, p80, throughput, config) in enumerate(results, 1):
        print(
            f"{rank:>4}  {median:>10.4f}  {p20:>7.4f}  {p80:>7.4f}  "
            f"{throughput:>7.2f}  {config_name(config)}"
        )

    if failures:
        print("\ninvalid configs")
        for config, message in failures:
            print(f"- {config_name(config)}: {message}")


if __name__ == "__main__":
    main()
