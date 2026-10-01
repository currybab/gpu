"""Fixed-tile factorial comparison: stages x warp specialization x epilogue subtiling."""

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

import persistent_matmul as pm
from tune_tma import measure, write_csv


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--block-n", type=int, choices=(128, 256), default=256)
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--programs-per-sm", type=int, default=1)
    parser.add_argument("--stages", type=int, nargs="+", default=[2, 3, 4, 5])
    args = parser.parse_args()
    if args.size <= 0 or args.size % 8 or args.programs_per_sm <= 0 or min(args.stages) <= 0:
        parser.error("positive arguments required; size must be divisible by 8")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("이 WS 비교는 Blackwell CUDA GPU에서 실행하세요.")
    out = Path(os.environ.get("OUT_DIR", "tuning_results"))
    out.mkdir(parents=True, exist_ok=True)
    prefix = "epilogue_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    size, bm, bn, bk = args.size, 128, args.block_n, 64
    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    expected = a @ b
    properties = torch.cuda.get_device_properties("cuda")
    programs = min(properties.multi_processor_count * args.programs_per_sm,
                   triton.cdiv(size, bm) * triton.cdiv(size, bn))
    rows, runnable = [], []
    rng = random.Random(0)
    jobs = list(itertools.product(args.stages, (False, True), (False, True)))
    rng.shuffle(jobs)
    for stages, ws, subtile in jobs:
        row = dict(stages=stages, warp_specialize=ws, epilogue_subtile=subtile)
        try:
            descs = (
                TensorDescriptor(a, list(a.shape), list(a.stride()), [bm, bk]),
                TensorDescriptor(b, list(b.shape), list(b.stride()), [bk, bn]),
                TensorDescriptor(c, list(c.shape), list(c.stride()), [bm, bn // 2 if subtile else bn]),
            )
            run = partial(
                pm._tma_epilogue_matmul_kernel[(programs,)], *descs, size, size, size,
                NUM_PROGRAMS=programs, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                WARP_SPECIALIZE=ws, EPILOGUE_SUBTILE=subtile,
                num_warps=args.num_warps, num_stages=stages,
            )
            c.fill_(float("nan"))
            kernel = run()
            torch.cuda.synchronize()
            torch.testing.assert_close(c, expected, atol=2e-2, rtol=1e-2)
            row.update(status="ok", shared_bytes=kernel.metadata.shared, regs=kernel.n_regs,
                       spills=kernel.n_spills, compiled_num_warps=kernel.metadata.num_warps, samples_ms=[])
            runnable.append((row, run))
        except (triton.OutOfResources, triton.CompilationError) as error:
            row.update(status="compile_error", error=str(error))
        rows.append(row)
        print(f"stages={stages}, WS={ws}, subtile={subtile}: {row['status']}", flush=True)

    reference = dict(name="torch", samples_ms=[])
    runners = [*runnable, (reference, partial(torch.mm, a, b, out=c))]
    for repeat in range(3):
        rng.shuffle(runners)
        for row, run in runners:
            row["samples_ms"].append(measure(run, 100)["median_ms"])
        print(f"Timing round {repeat + 1}/3 complete", flush=True)
    reference["median_ms"] = statistics.median(reference["samples_ms"])
    for row, _ in runnable:
        row["median_ms"] = statistics.median(row["samples_ms"])
        row["tflops"] = 2 * size**3 / (row["median_ms"] * 1e9)
        row["ratio_vs_torch"] = reference["median_ms"] / row["median_ms"]
    rows.sort(key=lambda r: (r["warp_specialize"], r["stages"], r["epilogue_subtile"]))
    metadata = dict(gpu=properties.name, torch=torch.__version__, triton=triton.__version__,
                    M=size, N=size, K=size, block_m=bm, block_n=bn, block_k=bk,
                    num_warps=args.num_warps, num_programs=programs, programs_per_sm=args.programs_per_sm,
                    shared_limit_bytes=properties.shared_memory_per_block_optin,
                    kernel_sha256=hashlib.sha256(Path(pm.__file__).read_bytes()).hexdigest(),
                    fp16_reduced_precision_reduction=False,
                    timing="CUDA Graph, preallocated output/descriptors, reused inputs; 3 x 100ms median")
    (out / f"{prefix}.json").write_text(json.dumps(dict(metadata=metadata, torch=reference, results=rows), indent=2))
    write_csv(out / f"{prefix}.csv", rows)
    lines = ["# Epilogue / stages / WS comparison", "", json.dumps(metadata, ensure_ascii=False), "",
             "| WS | Stages | Subtile | Shared KiB | Time μs | TFLOPS |",
             "|---|---:|---|---:|---:|---:|"]
    for r in rows:
        if r["status"] == "ok":
            line = (f"| {r['warp_specialize']} | {r['stages']} | {r['epilogue_subtile']} | "
                    f"{r['shared_bytes']/1024:.2f} | {r['median_ms']*1000:.2f} | {r['tflops']:.1f} |")
        else:
            line = f"| {r['warp_specialize']} | {r['stages']} | {r['epilogue_subtile']} | compile error | — | — |"
        lines.append(line)
        print(line, flush=True)
    (out / f"{prefix}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
