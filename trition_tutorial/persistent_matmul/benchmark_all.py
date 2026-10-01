"""One-device comparison of studied GEMMs, with explicit implementation/config metadata."""

import argparse
from dataclasses import asdict
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
from benchmark_official import COMMIT, SOURCE_SHA256, load_official, prepare_official
from tune_tma import Config, measure, prepare, write_csv


def write_report(out, prefix, metadata, rows):
    (out / f"{prefix}.json").write_text(json.dumps(
        dict(metadata=metadata, results=rows), indent=2, ensure_ascii=False,
    ) + "\n")
    write_csv(out / f"{prefix}.csv", rows)
    lines = ["# 통합 GEMM 측정", "", f"GPU: {metadata['gpu']} ({metadata['gpu_uuid']})",
             f"M=N=K={metadata['size']}, FP16, Torch {metadata['torch']}, Triton {metadata['triton']}",
             f"측정: 동일 GPU에서 순서를 섞어 {metadata['rounds']}회 × {metadata['rep_ms']}ms CUDA Graph.",
             "출력/descriptor 사전 할당, 컴파일·autotune·B 전치 비용 제외, 동일 입력 재사용.",
             "우리 구현은 표에 표시한 기존 설정을 재측정하며 새 전체 튜닝은 수행하지 않는다.",
             "공식 구현은 이번 실행의 공식 autotune 선택값이다. 기본값 행은 최적 성능이라는 뜻이 아니다.",
             "B=KN은 원래 contiguous B[K,N], NK는 B를 contiguous [N,K]로 미리 전치한 저장 배치다.",
             "처리량 비율의 기준은 같은 실행·같은 B 배치의 PyTorch다.", "",
             "| ID | 구현/설정 출처 | 상태 | B | F/WS/E | GROUP | BM/BN/BK | W 요청→실제 | S launch/loop | P/SM; grid | μs | TFLOPS | Torch 대비 | Shared KiB |",
             "|---|---|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|"]
    def flag(value):
        return "—" if value is None else str(int(value))
    for r in rows:
        flags = "/".join(flag(r[k]) for k in ("flatten", "warp_specialize", "epilogue_subtile"))
        tile = "/".join(str(r.get(k, "—")) for k in ("block_m", "block_n", "block_k"))
        warps = f"{r.get('num_warps', '—')}→{r.get('compiled_num_warps', '—')}"
        stages = f"{r.get('compiled_num_stages', r.get('num_stages', '—'))}/{r.get('loop_stages', 'default')}"
        programs = f"{r.get('programs_per_sm', '—')}; {r.get('grid', '—')}"
        timing = (f"{r['median_ms']*1000:.2f} | {r['tflops']:.1f} | {r['ratio_vs_torch']*100:.1f}%"
                  if r["status"] == "ok" else "— | — | —")
        shared = f"{r['shared_bytes']/1024:.2f}" if 'shared_bytes' in r else "—"
        lines.append(f"| {r['id']} | {r['label']} / {r['config_source']} | {r['status']} | "
                     f"{r['b_layout']} | {flags} | {r.get('group_size_m', '—')} | {tile} | "
                     f"{warps} | {stages} | {programs} | {timing} | {shared} |")
    lines += ["", "F=flatten, WS=warp_specialize, E=epilogue_subtile. 1=켜짐, 0=꺼짐, —=해당 없음.",
              "P=0은 전체 tile grid. S의 loop 값은 명시적인 tl.range num_stages이며 launch 값과 구분한다.",
              "warp specialization의 컴파일된 warp 수는 요청 수보다 클 수 있다. F/WS는 소스의 요청 설정이며 실제 overlap을 증명하는 profiler 지표는 아니다.", "",
              "## 실행되지 않은 조합", ""]
    lines += [f"- {r['id']} {r['label']}: {r.get('error', '')}" for r in rows if r['status'] != 'ok']
    if "transpose_ms" in metadata:
        lines += ["", f"B 전치 복사 시간(별도): {metadata['transpose_ms']*1000:.2f} μs."]
    lines += ["", f"완료: {metadata['complete']}. GPU 실행 오류는 중단하며, resource/compile 오류만 건너뛴다."]
    (out / f"{prefix}.md").write_text("\n".join(lines) + "\n")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()
    if args.size <= 0 or args.size % 8 or min(args.rounds, args.rep_ms) <= 0:
        parser.error("positive values required; size must be divisible by 8")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("B200 등 Blackwell CUDA GPU에서 실행하세요.")
    out = Path(os.environ.get("OUT_DIR", "tuning_results"))
    out.mkdir(parents=True, exist_ok=True)
    prefix = "all_gemm_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    module = load_official(out)
    properties = torch.cuda.get_device_properties("cuda")
    sms = properties.multi_processor_count
    size = args.size
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    bt = b.T.contiguous()
    c = torch.empty_like(a)
    expected = a @ b
    rows, runners = [], []
    metadata = dict(gpu=properties.name, gpu_uuid=str(properties.uuid), num_sms=sms,
                    size=size, dtype="float16", torch=torch.__version__, triton=triton.__version__,
                    cuda=torch.version.cuda, rounds=args.rounds, rep_ms=args.rep_ms, seed=0,
                    fp16_reduced_precision_reduction=False, complete=False,
                    official_commit=COMMIT, official_sha256=SOURCE_SHA256,
                    kernel_sha256=hashlib.sha256(Path(pm.__file__).read_bytes()).hexdigest(),
                    timing="single GPU, sequential shuffled CUDA Graph replay; preallocated output/descriptors; warm inputs")

    def add(label, source, build, *, layout="KN", flatten=None, ws=False, epi=False, group=1, **config):
        row = dict(id=f"R{len(rows):02d}", label=label, config_source=source, b_layout=layout,
                   flatten=flatten, warp_specialize=ws, epilogue_subtile=epi, group_size_m=group, **config)
        try:
            run, details = build()
            row.update(details)
            c.fill_(float("nan"))
            kernel = run()
            torch.cuda.synchronize()
            torch.testing.assert_close(c, expected, atol=2e-2, rtol=1e-2)
            if not label.startswith("PyTorch"):
                row.update(kernel=kernel.name, regs=kernel.n_regs, spills=kernel.n_spills,
                           shared_bytes=kernel.metadata.shared, compiled_num_warps=kernel.metadata.num_warps,
                           compiled_num_stages=kernel.metadata.num_stages)
            row.update(status="ok", samples_ms=[])
            runners.append((row, run))
        except (triton.OutOfResources, triton.CompilationError) as error:
            row.update(status="compile_error", error=f"{type(error).__name__}: {error}")
        rows.append(row)
        print(f"{row['id']} {label} / {source}: {row['status']}", flush=True)

    try:
        for layout, operand in (("KN", b), ("NK", bt.T)):
            add(f"PyTorch {layout}", "reference", lambda operand=operand: (partial(torch.mm, a, operand, out=c), {}),
                layout=layout, ws=None, epi=None, group=None)

        # Original raw-pointer kernels, using their wrapper defaults. These are not newly tuned.
        for kind, kernel, persistent, group, flat in (
            ("raw tiled", pm._matmul_kernel, False, 1, None),
            ("raw persistent", pm._persistent_matmul_kernel, True, 1, False),
            ("swizzle tiled", pm._swizzle_matmul_kernel, False, 8, None),
            ("swizzle persistent", pm._persistent_swizzle_matmul_kernel, True, 8, False),
            ("cross-tile persistent", pm.cross_tile_pipelining_kernel, True, 8, True),
        ):
            bm, bn, bk = pm.BLOCK_M, pm.BLOCK_N, pm.BLOCK_K
            tiles = triton.cdiv(size, bm) * triton.cdiv(size, bn)
            programs = min(2 * sms, tiles) if persistent else tiles
            options = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=4)
            if persistent:
                options.update(NUM_PROGRAMS=programs, NUM_STAGES=4, num_stages=4)
            if group > 1:
                options['GROUP_SIZE_M'] = group
            run = partial(kernel[(programs,)], a, b, c, size, size, size,
                          *a.stride(), *b.stride(), *c.stride(), **options)
            add("ours " + kind, "wrapper defaults", lambda run=run: (run, {}),
                flatten=flat, group=group, block_m=bm, block_n=bn, block_k=bk, num_warps=4,
                num_stages=4 if persistent else None, programs_per_sm=2 if persistent else 0, grid=programs,
                loop_stages="outer=4" if flat else "outer=4,K=4" if persistent else "K=4")

        configs = {
            "wrapper defaults": dict(tma_tiled=Config(128,128,64,4,2), tma_persistent=Config(128,128,64,4,2,4),
                                     ws_off=Config(128,128,64,4,2,4), ws_on=Config(128,128,64,4,2,4)),
            "initial quick": dict(tma_tiled=Config(128,256,64,4,4), tma_persistent=Config(128,128,64,4,4,1),
                                  ws_off=Config(128,128,64,4,4,1), ws_on=Config(128,128,64,4,4,1)),
            "full candidate": dict(tma_tiled=Config(128,256,32,4,4), tma_persistent=Config(128,256,64,4,3,2),
                                   ws_off=Config(128,256,64,4,3,2), ws_on=Config(128,128,128,8,3,1)),
        }
        for source, choices in configs.items():
            for kind, cfg in choices.items():
                def build(kind=kind, cfg=cfg):
                    run, grid = prepare(kind, cfg, a, b, c)
                    return run, dict(grid=grid)
                add("ours " + kind, source, build, flatten=None if kind == "tma_tiled" else True,
                    ws=kind == "ws_on", **asdict(cfg))

        # Epilogue defaults and the controlled stage/WS/subtile experiment, all in this run.
        epi_cases = [(128,2,False,sub,"wrapper defaults") for sub in (False,True)]
        epi_cases += [(256,s,w,e,"controlled sweep") for s,w,e in itertools.product((2,3,4,5),(False,True),(False,True))]
        for bn, stages, ws, subtile, source in epi_cases:
            bm, bk = 128, 64
            programs = min(sms, triton.cdiv(size,bm)*triton.cdiv(size,bn))
            def build(bn=bn, stages=stages, ws=ws, subtile=subtile, programs=programs):
                descs = (TensorDescriptor(a,list(a.shape),list(a.stride()),[128,64]),
                         TensorDescriptor(b,list(b.shape),list(b.stride()),[64,bn]),
                         TensorDescriptor(c,list(c.shape),list(c.stride()),[128,bn//2 if subtile else bn]))
                return partial(pm._tma_epilogue_matmul_kernel[(programs,)], *descs, size,size,size,
                               NUM_PROGRAMS=programs, BLOCK_M=128, BLOCK_N=bn, BLOCK_K=64,
                               WARP_SPECIALIZE=ws, EPILOGUE_SUBTILE=subtile, num_warps=4, num_stages=stages), {}
            add("ours epilogue", source, build, flatten=True, ws=ws, epi=subtile,
                block_m=bm,block_n=bn,block_k=bk,num_warps=4,num_stages=stages,programs_per_sm=1,grid=programs)

        for kind, ws in (("raw_tiled",False),("raw_persistent",False),("tma_tiled",False),
                         ("tma_tiled",True),("tma_persistent",False),("tma_persistent",True)):
            persistent = kind.endswith("persistent")
            def build(kind=kind, ws=ws, persistent=persistent):
                run, opts = prepare_official(module,kind,ws,a,bt,c,expected)
                bm,bn,bk = (opts[k] for k in ('BLOCK_SIZE_M','BLOCK_SIZE_N','BLOCK_SIZE_K'))
                tiles = triton.cdiv(size,bm)*triton.cdiv(size,bn)
                return run, dict(block_m=bm,block_n=bn,block_k=bk,num_warps=opts['num_warps'],
                                 num_stages=opts['num_stages'],programs_per_sm=1 if persistent else 0,
                                 grid=min(sms,tiles) if persistent else tiles,
                                 epilogue_subtile=opts.get('EPILOGUE_SUBTILE',False),official_options=opts)
            print(f"Official autotune: {kind}, WS={ws}",flush=True)
            add("official " + kind,"official autotune this run",build,layout="NK",
                flatten=True if persistent else None,ws=ws,group=8)

        for repeat in range(args.rounds):
            random.Random(repeat).shuffle(runners)
            for row, run in runners:
                row['samples_ms'].append(measure(run,args.rep_ms)['median_ms'])
            print(f"Round {repeat+1}/{args.rounds}: {len(runners)} implementations/configs",flush=True)
        refs = {r['b_layout']:statistics.median(r['samples_ms']) for r in rows if r['label'].startswith('PyTorch')}
        for row, _ in runners:
            row['median_ms'] = statistics.median(row['samples_ms'])
            row['min_sample_ms'],row['max_sample_ms'] = min(row['samples_ms']),max(row['samples_ms'])
            row['tflops'] = 2*size**3/(row['median_ms']*1e9)
            row['ratio_vs_torch'] = refs[row['b_layout']]/row['median_ms']
            row['ratio_vs_original_torch'] = refs['KN']/row['median_ms']
        metadata['transpose_ms'] = measure(lambda: bt.copy_(b.T),args.rep_ms)['median_ms']
        metadata['complete'] = True
    finally:
        # Save incomplete runs as well. Never rank unmeasured/incorrect output as a success.
        for r in rows:
            if r['status']=='ok' and 'median_ms' not in r:
                r['status']='validated_not_measured'
        write_report(out,prefix,metadata,rows)
        print(f"Saved {out/prefix}, complete={metadata['complete']}",flush=True)
    for r in rows:
        if r['status']=='ok':
            print(f"{r['id']} {r['label']} / {r['config_source']}: {r['median_ms']*1000:.2f} us, "
                  f"{r['tflops']:.1f} TFLOPS, {100*r['ratio_vs_torch']:.1f}%",flush=True)


if __name__ == "__main__":
    main()
