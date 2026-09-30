"""TMA shape/config sweep. CUDA Graph GPU timing; allocation/compilation excluded."""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

import persistent_matmul as pm


VARIANTS = ("tma_tiled", "tma_persistent", "ws_off", "ws_on")
DEFAULT_SHAPES = (
    (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
    (8192, 8192, 8192), (8192, 2048, 4096), (2048, 8192, 4096),
    (16384, 16384, 4096),
)


@dataclass(frozen=True)
class Config:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    programs_per_sm: int = 0  # tiled uses the complete tile grid


def configs(preset, variant):
    if preset == "smoke":
        tiles, warps, stages, programs = [(64, 64, 32), (128, 128, 64)], [4], [2], [1, 4]
    elif preset == "quick":
        # Include the official tutorial's BK=128 and stage=3 candidates.
        # Stage 4 may exceed shared memory while stage 3 still fits.
        tiles = [(64, 64, 32), (128, 64, 64), (128, 128, 64),
                 (128, 256, 64), (128, 128, 128), (128, 256, 128)]
        warps, stages, programs = [4, 8], [2, 3, 4], [1, 2, 4]
    else:
        tiles = list(itertools.product([64, 128], [64, 128, 256], [32, 64, 128]))
        warps, stages, programs = [4, 8], [2, 3, 4], [1, 2, 4]
    if variant == "tma_tiled":
        programs = [0]
    return [Config(*tile, w, s, p) for tile, w, s, p in itertools.product(tiles, warps, stages, programs)]


def parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
        if len(shape) != 3 or min(shape) <= 0 or shape[1] % 8 or shape[2] % 8:
            raise ValueError
        return shape
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape은 MxNxK, 양수이며 N/K는 8의 배수여야 합니다.") from error


def prepare(variant, config, a, b, c):
    """Reuse descriptors/output across correctness checks and graph replays."""
    m, k = a.shape
    n = b.shape[1]
    bm, bn, bk = config.block_m, config.block_n, config.block_k
    descs = (
        TensorDescriptor(a, list(a.shape), list(a.stride()), [bm, bk]),
        TensorDescriptor(b, list(b.shape), list(b.stride()), [bk, bn]),
        TensorDescriptor(c, list(c.shape), list(c.stride()), [bm, bn]),
    )
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    options = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                   num_warps=config.num_warps, num_stages=config.num_stages)
    if variant == "tma_tiled":
        return partial(pm._tma_matmul_kernel[(tiles,)], *descs, n, k, **options), tiles
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    programs = min(sms * config.programs_per_sm, tiles)
    options["NUM_PROGRAMS"] = programs
    if variant == "tma_persistent":
        kernel = pm._tma_persistent_matmul_kernel
    else:
        kernel = pm._tma_warp_specialized_matmul_kernel
        options["WARP_SPECIALIZE"] = variant == "ws_on"
    return partial(kernel[(programs,)], *descs, m, n, k, **options), programs


def measure(run, rep):
    p20, median, p80 = triton.testing.do_bench_cudagraph(
        run, rep=rep, quantiles=[0.2, 0.5, 0.8],
    )
    if not math.isfinite(median) or median <= 0:
        raise RuntimeError(f"Invalid GPU timing: {median}")
    return dict(p20_ms=p20, median_ms=median, p80_ms=p80)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(out, prefix, metadata, rows):
    """Keep failed configs and incomplete searches visible, including on budget exit."""
    write_csv(out / f"{prefix}_all.csv", rows)
    best = []
    for shape in metadata["shapes"]:
        for variant in ("torch", *metadata["variants"]):
            trials = [r for r in rows if [r["M"], r["N"], r["K"]] == list(shape)
                      and r["variant"] == variant]
            candidates = [r for r in trials if r["status"] == "ok"]
            # Finalists are remeasured with a longer timing window.
            finalists = [r for r in candidates if r["phase"] == "final"]
            if any(r["phase"] == "final" for r in trials) and not finalists:
                continue  # Do not promote a config whose final validation failed.
            if finalists or candidates:
                winner = dict(min(finalists or candidates, key=lambda r: r["median_ms"]))
                winner["selection"] = "remeasured" if finalists else "screening_only"
                best.append(winner)
    for row in best:
        reference = next((r for r in best if r["variant"] == "torch"
                          and (r["M"], r["N"], r["K"]) == (row["M"], row["N"], row["K"])), None)
        row["speedup_vs_torch"] = reference["median_ms"] / row["median_ms"] if reference else None
    write_csv(out / f"{prefix}_best.csv", best)
    (out / f"{prefix}_summary.json").write_text(json.dumps(
        {"metadata": metadata, "best": best}, indent=2, ensure_ascii=False,
    ) + "\n")
    lines = ["# TMA tuning results", "", f"GPU: {metadata['gpu']}",
             f"Search complete: {metadata['complete']}",
             "CUDA Graph replay, preallocated output/descriptors, reused inputs (warm cache).",
             "Best among tested configurations/shapes; not a hardware peak or wrapper latency.", "",
             "| M×N×K | Variant | BM/BN/BK | Warps/Stages/P per SM | ms | TFLOPS | vs torch | Selection |",
             "|---|---|---|---|---:|---:|---:|---|"]
    for r in best:
        tile = "/".join(str(r.get(key, "-")) for key in ("block_m", "block_n", "block_k"))
        launch = "/".join(str(r.get(key, "-")) for key in ("num_warps", "num_stages", "programs_per_sm"))
        speedup = f"{r['speedup_vs_torch']:.2f}x" if r['speedup_vs_torch'] is not None else "-"
        lines.append(f"| {r['M']}×{r['N']}×{r['K']} | {r['variant']} | {tile} | {launch} | "
                     f"{r['median_ms']:.4f} | {r['tflops']:.2f} | {speedup} | {r['selection']} |")
    lines += ["", "## Highest measured throughput per variant", ""]
    for variant in ("torch", *metadata["variants"]):
        candidates = [r for r in best if r["variant"] == variant]
        if candidates:
            r = max(candidates, key=lambda row: row["tflops"])
            lines.append(f"- {variant}: {r['tflops']:.2f} TFLOPS at {r['M']}×{r['N']}×{r['K']}")
    errors = [r for r in rows if r["status"] != "ok"]
    lines += ["", f"Failed/skipped trials: {len(errors)} (details in all.csv)"]
    (out / f"{prefix}_report.md").write_text("\n".join(lines) + "\n")
    return best


def plot_best(out, prefix, best):
    if not best:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = list(dict.fromkeys((r["M"], r["N"], r["K"]) for r in best))
    variants = list(dict.fromkeys(r["variant"] for r in best))
    fig, axes = plt.subplots(2, 1, figsize=(max(9, len(shapes) * 1.7), 8), sharex=True)
    width = 0.8 / len(variants)
    for index, variant in enumerate(variants):
        selected = {(r["M"], r["N"], r["K"]): r for r in best if r["variant"] == variant}
        x = [i - 0.4 + width * (index + 0.5) for i in range(len(shapes))]
        for ax, metric in zip(axes, ("tflops", "speedup_vs_torch")):
            values = [selected.get(s, {}).get(metric) or float("nan") for s in shapes]
            ax.bar(x, values, width, label=variant)
    axes[0].set_ylabel("TFLOPS")
    axes[0].legend(ncol=min(5, len(variants)), fontsize=8)
    axes[0].set_title("Best tested configs per shape — CUDA Graph replay, reused inputs")
    axes[1].set_ylabel("Speedup vs PyTorch")
    axes[1].axhline(1, color="gray", linewidth=1, linestyle="--")
    axes[1].set_xticks(range(len(shapes)), [f"{m}×{n}×{k}" for m, n, k in shapes], rotation=25, ha="right")
    axes[1].set_xlabel("M × N × K")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_best.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "quick", "full"), default="quick")
    parser.add_argument("--shapes", nargs="+", type=parse_shape)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--rep-ms", type=int, default=20)
    parser.add_argument("--final-rep-ms", type=int, default=100)
    parser.add_argument("--final-repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--output-dir", default=os.environ.get("OUT_DIR", "tuning_results"))
    args = parser.parse_args(argv)
    if min(args.rep_ms, args.final_rep_ms, args.final_repeats, args.top_k, args.max_seconds) <= 0:
        parser.error("timing/top-k/budget values must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU에서 실행하세요.")
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("TMA 지원 GPU가 필요합니다.")
    shapes = args.shapes or ([(256, 256, 256), (1024, 1024, 1024)] if args.preset == "smoke" else list(DEFAULT_SHAPES))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "tma_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    rng = random.Random(0)
    metadata = dict(
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        num_sms=torch.cuda.get_device_properties("cuda").multi_processor_count,
        torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
        kernel_sha256=hashlib.sha256(Path(pm.__file__).read_bytes()).hexdigest(),
        preset=args.preset, shapes=shapes, variants=args.variants, dtype="float16",
        rep_ms=args.rep_ms, final_rep_ms=args.final_rep_ms,
        final_repeats=args.final_repeats, top_k=args.top_k,
        max_seconds=args.max_seconds, seed=0, complete=False,
        timing="CUDA Graph replay; preallocated output/descriptors; reused inputs, warm cache",
        fp16_reduced_precision_reduction=False,
        search_space={v: [asdict(c) for c in configs(args.preset, v)] for v in args.variants},
    )
    rows = []
    started = time.monotonic()
    deadline = started + args.max_seconds
    print(f"GPU={metadata['gpu']}, preset={args.preset}, shapes={shapes}", flush=True)
    print(f"Output prefix: {out / prefix}", flush=True)

    def trial(variant, config, a, b, c, expected, phase):
        m, k = a.shape
        n = b.shape[1]
        row = dict(M=m, N=n, K=k, variant=variant, phase=phase,
                   **(asdict(config) if config else {}))
        try:
            if variant == "ws_on" and metadata["capability"][0] < 10:
                row.update(status="unsupported", error="Automatic WS experiment requires Blackwell")
                return row
            if variant == "torch":
                run = partial(torch.mm, a, b, out=c)
            else:
                run, programs = prepare(variant, config, a, b, c)
                row["num_programs"] = programs
            c.fill_(float("nan"))
            kernel = run()  # compilation, warmup and validation are outside timing
            torch.cuda.synchronize()
            torch.testing.assert_close(c, expected, atol=2e-2, rtol=1e-2)
            if variant != "torch":
                row.update(regs=kernel.n_regs, spills=kernel.n_spills,
                           shared_bytes=kernel.metadata.shared,
                           compiled_num_warps=kernel.metadata.num_warps)
            timings = [measure(run, args.final_rep_ms if phase == "final" else args.rep_ms)
                       for _ in range(args.final_repeats if phase == "final" else 1)]
            row.update({key: statistics.median(t[key] for t in timings) for key in timings[0]})
            row["sample_medians_ms"] = json.dumps([t["median_ms"] for t in timings])
            row.update(status="ok", tflops=2 * m * n * k / (row["median_ms"] * 1e9))
        except AssertionError as error:
            row.update(status="incorrect", error=str(error)[:2000])
        except (triton.OutOfResources, triton.CompilationError) as error:
            row.update(status="compile_error", error=f"{type(error).__name__}: {error}"[:2000])
        # CUDA runtime failures (e.g. illegal access) must stop the run: the context may be invalid.
        return row

    try:
        for m, n, k in shapes:
            if time.monotonic() >= deadline:
                return
            a = torch.randn((m, k), device="cuda", dtype=torch.float16)
            b = torch.randn((k, n), device="cuda", dtype=torch.float16)
            c = torch.empty((m, n), device="cuda", dtype=torch.float16)
            expected = a @ b
            print(f"\nShape {m}x{n}x{k}", flush=True)
            rows.append(trial("torch", None, a, b, c, expected, "screen"))
            jobs = [(v, config) for v in args.variants for config in configs(args.preset, v)]
            # Interleave variants to reduce ordering/thermal bias; deterministic seed.
            rng.shuffle(jobs)
            candidates = {v: [] for v in args.variants}
            for index, (variant, config) in enumerate(jobs, 1):
                if time.monotonic() >= deadline:
                    return
                row = trial(variant, config, a, b, c, expected, "screen")
                rows.append(row)
                if row["status"] == "ok":
                    candidates[variant].append((row["median_ms"], config))
                else:
                    print(f"  {variant} {config}: {row['status']} {row.get('error', '')[:160]}", flush=True)
                if index % 10 == 0 or index == len(jobs):
                    print(f"  {index}/{len(jobs)} configs, elapsed={time.monotonic() - started:.0f}s", flush=True)
                    report(out, prefix, metadata, rows)
            finalists = [(v, cfg) for v, results in candidates.items()
                         for _, cfg in sorted(results, key=lambda pair: pair[0])[:args.top_k]]
            rng.shuffle(finalists)
            for variant, config in finalists:
                if time.monotonic() >= deadline:
                    return
                rows.append(trial(variant, config, a, b, c, expected, "final"))
            # Refresh PyTorch after compilation/search, in the same warmed-up run.
            rows.append(trial("torch", None, a, b, c, expected, "final"))
            best = report(out, prefix, metadata, rows)
            for row in best:
                if (row["M"], row["N"], row["K"]) == (m, n, k):
                    print(f"  BEST {row['variant']}: {row['median_ms']:.4f} ms, {row['tflops']:.2f} TFLOPS", flush=True)
            del a, b, c, expected
        metadata["complete"] = True
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - started
        best = report(out, prefix, metadata, rows)
        plot_best(out, prefix, best)
        print(f"Saved results: {out / prefix} (complete={metadata['complete']})", flush=True)


if __name__ == "__main__":
    main()
