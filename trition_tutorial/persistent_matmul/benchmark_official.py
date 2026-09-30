"""Compare pinned official Triton tutorial kernels against PyTorch on one GPU."""

import argparse
import csv
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import urllib.request

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from tune_tma import Config, VARIANTS, measure, prepare


COMMIT = "2e5ace0605bd5005da5c907a25de14d9d1707a98"
SOURCE_URL = f"https://raw.githubusercontent.com/triton-lang/triton/{COMMIT}/python/tutorials/09-persistent-matmul.py"
SOURCE_SHA256 = "43e51750a2271af05ed5fb7221c811b8d140b2a06665863eaff8799e474a7e03"


def load_official(out):
    source = urllib.request.urlopen(SOURCE_URL, timeout=30).read()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256, "Official source hash mismatch"
    path = out / "official_09_persistent_matmul.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("official_persistent", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_official(module, kind, ws, a, bt, c, expected):
    """Keep official kernels/search spaces unchanged; freeze winner and preallocate C."""
    m, k = a.shape
    n = bt.shape[0]
    raw = kind in ("raw_tiled", "raw_persistent")
    persistent = kind.endswith("persistent")
    if raw:
        wrapper = module.matmul_persistent if persistent else module.matmul
        tuner = module.matmul_kernel_persistent if persistent else module.matmul_kernel
        actual = wrapper(a, bt.T)
    else:
        wrapper = module.matmul_tma_persistent if persistent else module.matmul_tma
        tuner = module.matmul_kernel_tma_persistent if persistent else module.matmul_kernel_tma
        actual = wrapper(a, bt, ws)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)
    cfg = tuner.best_config
    options = cfg.all_kwargs()
    bm, bn, bk = (cfg.kwargs[key] for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"))
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    if persistent:
        options["NUM_SMS"] = sms
    if raw:
        operands = (a, bt.T, c, m, n, k, *a.stride(), *bt.T.stride(), *c.stride())
    else:
        subtile = cfg.kwargs.get("EPILOGUE_SUBTILE", False)
        descs = (
            TensorDescriptor(a, list(a.shape), list(a.stride()), [bm, bk]),
            TensorDescriptor(bt, list(bt.shape), list(bt.stride()), [bn, bk]),
            TensorDescriptor(c, list(c.shape), list(c.stride()), [bm, bn // 2 if subtile else bn]),
        )
        operands = (*descs, m, n, k)
        options.update(FP8_OUTPUT=False, WARP_SPECIALIZE=ws)
    run = partial(tuner.fn[(min(sms, tiles) if persistent else tiles,)], *operands, **options)
    return run, options


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--tuned-configs", type=json.loads, default={},
                        help='JSON mapping of variant names to Config fields; remeasure beside initial quick winners')
    args = parser.parse_args()
    if args.size <= 0 or args.size % 8:
        parser.error("size must be positive and divisible by 8")
    if not isinstance(args.tuned_configs, dict) or any(v not in VARIANTS for v in args.tuned_configs):
        parser.error("tuned-configs must map supported variant names to Config fields")
    out = Path(os.environ.get("OUT_DIR", "tuning_results"))
    out.mkdir(parents=True, exist_ok=True)
    prefix = "official_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    module = load_official(out)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    size = args.size
    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    bt = b.T.contiguous()  # official tutorial stores B as contiguous [N,K]
    c = torch.empty_like(a)
    expected = a @ b
    runs = {
        "torch_original_layout": (partial(torch.mm, a, b, out=c), {}, "original K,N"),
        "torch_official_layout": (partial(torch.mm, a, bt.T, out=c), {}, "pretransposed N,K"),
    }
    for name, config in (
        ("tma_tiled", Config(128, 256, 64, 4, 4)),
        ("tma_persistent", Config(128, 128, 64, 4, 4, 1)),
        ("ws_off", Config(128, 128, 64, 4, 4, 1)),
        ("ws_on", Config(128, 128, 64, 4, 4, 1)),
    ):
        run, _ = prepare(name, config, a, b, c)
        runs[f"ours_{name}"] = (run, config.__dict__, "original K,N")
    for name, fields in args.tuned_configs.items():
        config = Config(**fields)
        run, _ = prepare(name, config, a, b, c)
        runs[f"ours_retuned_{name}"] = (run, config.__dict__, "original K,N")
    for kind, ws in (("raw_tiled", False), ("raw_persistent", False),
                     ("tma_tiled", False), ("tma_tiled", True),
                     ("tma_persistent", False), ("tma_persistent", True)):
        name = f"official_{kind}_ws_{ws}"
        print(f"Autotuning {name} ...", flush=True)
        run, config = prepare_official(module, kind, ws, a, bt, c, expected)
        runs[name] = (run, config, "pretransposed N,K")
        print(f"  selected: {config}", flush=True)

    resources = {}
    for name, (run, _, _) in runs.items():
        c.fill_(float("nan"))
        kernel = run()
        torch.cuda.synchronize()
        torch.testing.assert_close(c, expected, atol=2e-2, rtol=1e-2)
        if not name.startswith("torch"):
            resources[name] = dict(regs=kernel.n_regs, spills=kernel.n_spills,
                                   shared_bytes=kernel.metadata.shared, compiled_num_warps=kernel.metadata.num_warps)
    samples = {name: [] for name in runs}
    rng = random.Random(0)
    for repeat in range(3):
        order = list(runs)
        rng.shuffle(order)
        for name in order:
            samples[name].append(measure(runs[name][0], 100)["median_ms"])
        print(f"Measurement round {repeat + 1}/3 complete", flush=True)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    rows = []
    for name, (_, config, layout) in runs.items():
        ref = "torch_original_layout" if layout == "original K,N" else "torch_official_layout"
        row = dict(name=name, layout=layout, median_ms=medians[name],
                   tflops=2 * size**3 / (medians[name] * 1e9),
                   ratio_vs_same_layout_torch=medians[ref] / medians[name],
                   config=config, samples_ms=samples[name], **resources.get(name, {}))
        rows.append(row)
        print(f"{name}: {row['median_ms']:.5f} ms, {row['tflops']:.1f} TFLOPS, "
              f"{100 * row['ratio_vs_same_layout_torch']:.1f}% of same-layout torch", flush=True)
    transpose_ms = measure(lambda: bt.copy_(b.T), 100)["median_ms"]
    metadata = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                    M=size, N=size, K=size, dtype="float16", official_commit=COMMIT,
                    official_url=SOURCE_URL, official_sha256=SOURCE_SHA256,
                    transpose_copy_ms=transpose_ms, fp16_reduced_precision_reduction=False,
                    ours_baseline="initial quick winners at 4096^3", ours_retuned_configs=args.tuned_configs,
                    measurement="CUDA Graph replay, preallocated C, 3x100ms median; official autotune before timing",
                    layout_note="Official kernels/PyTorch use pretransposed B[N,K]; transpose cost excluded")
    (out / f"{prefix}.json").write_text(json.dumps(dict(metadata=metadata, results=rows), indent=2) + "\n")
    with (out / f"{prefix}.csv").open("w", newline="") as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"One B transpose copy: {transpose_ms:.5f} ms", flush=True)


if __name__ == "__main__":
    main()
