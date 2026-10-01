"""Epilogue subtiling correctness and optional fixed-config wrapper benchmark."""

import argparse
from functools import partial

import torch
import triton

from persistent_matmul import tma_epilogue_matmul


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--programs-per-sm", type=int, default=1)
    parser.add_argument("--warp-specialize", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--size", type=int, default=4096)
    args = parser.parse_args()
    bm, bn, bk = args.block_m, args.block_n, args.block_k
    if any(v < 16 or v & (v - 1) for v in (bm, bn, bk)):
        parser.error("block sizes must be powers of two, at least 16")
    if min(args.num_stages, args.programs_per_sm, args.size) <= 0 or args.size % 8:
        parser.error("stages/programs/size must be positive; size must be divisible by 8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU에서 실행하세요.")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"BM/BN/BK={bm}/{bn}/{bk}, warps={args.num_warps}, stages={args.num_stages}, "
          f"programs/SM={args.programs_per_sm}, WS={args.warp_specialize}")
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    shapes = (
        (bm, bn, bk),
        (bm + 1, max(8, bn // 2 - 8), bk + 8),  # right half entirely out of bounds
        (bm + 3, bn // 2 + 8, bk + 8),          # right half partially in bounds
        (2 * bm + 1, bn + 8, bk + 8),
        (2 * sms * args.programs_per_sm * bm + 1, bn - 8, bk + 8),
    )
    ready = []
    for subtile in (False, True):
        label = f"EPILOGUE_SUBTILE={subtile}"
        run = partial(
            tma_epilogue_matmul, epilogue_subtile=subtile,
            warp_specialize=args.warp_specialize, block_m=bm, block_n=bn, block_k=bk,
            num_warps=args.num_warps, num_stages=args.num_stages,
            programs_per_sm=args.programs_per_sm,
        )
        # Distinct random data between paths prevents stale allocator contents passing as output.
        torch.manual_seed(int(subtile))
        try:
            for m, n, k in shapes:
                a = torch.randn((m, k), device="cuda", dtype=torch.float16)
                b = torch.randn((k, n), device="cuda", dtype=torch.float16)
                actual = run(a, b)
                torch.testing.assert_close(actual, a @ b, atol=2e-2, rtol=1e-2)
                print(f"{label} passed: {m}x{n}x{k}")
            ready.append((label, run))
        except NotImplementedError as error:
            print(f"{label} pending: {error}")
        except triton.OutOfResources as error:
            print(f"{label} resource_limited (not passed): {error}")

    if args.benchmark:
        a = torch.randn((args.size, args.size), device="cuda", dtype=torch.float16)
        b = torch.randn_like(a)
        expected = a @ b
        print("Wrapper timing: output allocation/descriptor creation included; compare False/True here.")
        for label, run in ready:
            try:
                torch.testing.assert_close(run(a, b), expected, atol=2e-2, rtol=1e-2)
                p20, median, p80 = triton.testing.do_bench(
                    lambda: run(a, b, print_metadata=False), quantiles=[0.2, 0.5, 0.8],
                )
                throughput = 2 * args.size**3 / (median * 1e9)
                print(f"{label}: {median:.4f} ms [p20={p20:.4f}, p80={p80:.4f}], {throughput:.2f} TFLOPS")
            except triton.OutOfResources as error:
                print(f"{label} resource_limited (not measured): {error}")


if __name__ == "__main__":
    main()
