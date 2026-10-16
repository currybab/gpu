"""FA1/FA2 고정 설정 및 개별 튜닝 비교. 계산 TODO 완성 후 실행한다.

버퍼 사전 할당 + CUDA Graph 반복 시간. 실패 후보도 CSV에 즉시 기록한다.
공식 CUDA 라이브러리 비교나 절대적인 최고 성능을 의미하지 않는다.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
import triton
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from attention_fa1 import prepare_fa1
from attention_fa2 import prepare_fa2
from check import attention_reference


@dataclass(frozen=True)
class Config:
    block_m: int = 64
    block_n: int = 64
    num_warps: int = 4
    num_stages: int = 2


PREPARE = {"fa1": prepare_fa1, "fa2": prepare_fa2}


def configs_for(preset):
    fixed = Config()
    if preset == "fixed":
        return [fixed]
    if preset == "quick":
        return [fixed, Config(32, 64, 4, 2), Config(64, 32, 4, 2),
                Config(64, 128, 4, 3), Config(128, 64, 8, 3), Config(128, 128, 8, 3)]
    # 같은 탐색 범위에서도 각 알고리즘의 최고 설정은 따로 고른다.
    return [fixed] + [c for values in itertools.product(
        (32, 64, 128), (32, 64, 128), (4, 8), (1, 2, 3, 4)
    ) if (c := Config(*values)) != fixed]


def inputs(shape):
    return [torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3)]


def validate_candidate(prepare, config, causal, d):
    # 모든 후보에서 짧은 sequence와 이중 tail을 검사한다.
    for shape in ((1, 2, 17, d), (1, 2, 129, 80)):
        q, k, v = inputs(shape)
        scale = shape[-1] ** -0.5
        run = prepare(q, k, v, causal, scale, **asdict(config))
        expected = attention_reference(q, k, v, causal, scale)
        torch.testing.assert_close(run(), expected, atol=1e-2, rtol=0)
        # FA1 state가 이전 호출에서 누적되지 않는지도 확인한다.
        torch.testing.assert_close(run(), expected, atol=1e-2, rtol=0)


def measure(run, rep_ms, repeats):
    return [float(triton.testing.do_bench_cudagraph(run, rep=rep_ms))
            for _ in range(repeats)]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="FA1/FA2 고정 설정 및 개별 튜닝 비교")
    parser.add_argument("--preset", choices=("fixed", "quick", "full"), default="quick")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 2048])
    parser.add_argument("--head-dim", type=int, choices=(32, 64, 80, 128), default=64)
    parser.add_argument("--causal", choices=("both", "on", "off"), default="both")
    parser.add_argument("--rep-ms", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(os.environ.get("OUT_DIR", "tuning_results")))
    args = parser.parse_args()
    if min(args.batch, args.heads, args.rep_ms, args.repeats, *args.seq_lens) <= 0:
        parser.error("shape와 측정 횟수/시간은 양수여야 합니다.")
    if not torch.cuda.is_available():
        parser.exit(1, "CUDA GPU가 필요합니다.\n")
    torch.manual_seed(0)
    probe = inputs((1, 1, 17, 64))
    incomplete = False
    for name, prepare in PREPARE.items():
        try:
            prepare(*probe, False, 64**-0.5)
        except NotImplementedError as exc:
            print(f"TODO {name}: {exc}")
            incomplete = True
    if incomplete:
        print("두 구현을 완성한 뒤 비교하세요. 성능 결과는 생성하지 않았습니다.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = args.out_dir / f"fa_compare_{stamp}"
    source_dir = Path(__file__).parent
    report = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "triton": triton.__version__, "cuda": torch.version.cuda,
        "timing": "preallocated CUDA Graph, repeated same buffers, no L2 flush",
        "args": {**vars(args), "out_dir": str(args.out_dir)},
        "source_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                          for name in ("attention_fa1.py", "attention_fa2.py", "compare_fa.py",
                                       "attention_variants.py", "check.py")},
        "comparisons": [], "complete": False,
    }
    fieldnames = ["shape", "causal", "impl", "phase", "block_m", "block_n",
                  "num_warps", "num_stages", "status", "median_ms", "samples_ms",
                  "registers", "spills", "shared_bytes", "error"]
    candidates = configs_for(args.preset)
    modes = (False, True) if args.causal == "both" else (args.causal == "on",)
    missing = False
    print(f"GPU={report['gpu']}, candidates={len(candidates)} per implementation")
    try:
        with stem.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()

            def record(shape, causal, name, phase, config, samples=None, error=None, run=None):
                row = dict(shape=str(shape), causal=causal, impl=name, phase=phase,
                           status="failed" if error else "passed",
                           median_ms=median(samples) if samples else "",
                           samples_ms=json.dumps(samples or []), error=error or "")
                row.update(asdict(config) if config else {})
                kernel = getattr(run, "kernel", None)
                row.update(registers=getattr(kernel, "n_regs", ""),
                           spills=getattr(kernel, "n_spills", ""),
                           shared_bytes=getattr(getattr(kernel, "metadata", None), "shared", ""))
                writer.writerow(row)
                stream.flush()

            for n, causal in itertools.product(args.seq_lens, modes):
                shape = (args.batch, args.heads, n, args.head_dim)
                q, k, v = inputs(shape)
                scale = args.head_dim**-0.5
                expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
                winners, fixed = {}, {}
                for name, prepare in PREPARE.items():
                    successes = []
                    for config in candidates:
                        try:
                            validate_candidate(prepare, config, causal, args.head_dim)
                            run = prepare(q, k, v, causal, scale, **asdict(config))
                            torch.testing.assert_close(run(), expected, atol=1e-2, rtol=0)
                            torch.testing.assert_close(run(), expected, atol=1e-2, rtol=0)
                            samples = measure(run, args.rep_ms, 1)
                            # graph replay 이후에도 올바른 결과인지 검사한다.
                            torch.testing.assert_close(run(), expected, atol=1e-2, rtol=0)
                        except (AssertionError, CompilationError, OutOfResources) as exc:
                            record(shape, causal, name, "search", config,
                                   error=f"{type(exc).__name__}: {exc}")
                            print(f"FAIL {name} {shape} causal={causal} {config}: {type(exc).__name__}")
                            continue
                        record(shape, causal, name, "search", config, samples, run=run)
                        successes.append((median(samples), config))
                        if config == Config():
                            fixed[name] = config
                        print(f"OK {name} {shape} causal={causal} {config}: {median(samples):.4f} ms")
                    if successes:
                        winners[name] = min(successes, key=lambda item: item[0])[1]
                    else:
                        missing = True

                # 고정 설정과 탐색 최고 설정을 각각 번갈아 재측정한다.
                result = {"shape": shape, "causal": causal}
                for phase, selected in (("fixed", fixed), ("tuned", winners)):
                    runs = {name: PREPARE[name](q, k, v, causal, scale, **asdict(config))
                            for name, config in selected.items()}
                    samples = {name: [] for name in runs}
                    for repeat in range(args.repeats):
                        order = list(runs) if repeat % 2 == 0 else list(reversed(runs))
                        for name in order:
                            samples[name].extend(measure(runs[name], args.rep_ms, 1))
                    values = {}
                    for name, run in runs.items():
                        torch.testing.assert_close(run(), expected, atol=1e-2, rtol=0)
                        record(shape, causal, name, phase, selected[name], samples[name], run=run)
                        values[name] = {"config": asdict(selected[name]),
                                        "median_ms": median(samples[name]), "samples_ms": samples[name]}
                    if len(values) == 2:
                        ratio = values["fa1"]["median_ms"] / values["fa2"]["median_ms"]
                        values["fa2_speedup"] = ratio
                        print(f"{phase}: shape={shape}, causal={causal}, FA1/FA2 time={ratio:.3f}x")
                    result[phase] = values
                report["comparisons"].append(result)
                stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        report["complete"] = True
    finally:
        stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"saved: {stem}.csv / {stem}.json")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
