"""Tune the five raw-pointer GEMMs, then compare defaults/winners/TMA on one GPU."""

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import partial
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import time

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

import persistent_matmul as pm
from tune_tma import Config, measure, prepare, write_csv

KINDS = ('raw_tiled', 'raw_persistent', 'swizzle_tiled', 'swizzle_persistent', 'cross_tile')
KERNELS = dict(zip(KINDS, (pm._matmul_kernel, pm._persistent_matmul_kernel,
                         pm._swizzle_matmul_kernel, pm._persistent_swizzle_matmul_kernel,
                         pm.cross_tile_pipelining_kernel)))


@dataclass(frozen=True)
class RawConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    programs_per_sm: int = 0
    group_size_m: int = 1
    loop_stages: int | None = None


def persistent(kind):
    return kind in ('raw_persistent', 'swizzle_persistent', 'cross_tile')


def grouped(kind):
    return kind in ('swizzle_tiled', 'swizzle_persistent', 'cross_tile')


def search_configs(kind, preset):
    tiles = list(itertools.product((64,128,256), (64,128,256), (32,64,128)))
    if preset == 'smoke':
        tiles = [(128,64,32), (128,256,64)]
    warps, stages = ((4,), (4,)) if preset == 'smoke' else ((4,8), (2,3,4))
    programs = (1,2,4) if persistent(kind) else (0,)
    return [RawConfig(*tile,w,s,p,8 if grouped(kind) else 1)
            for tile,w,s,p in itertools.product(tiles,warps,stages,programs)]


def default_config(kind):
    return RawConfig(pm.BLOCK_M,pm.BLOCK_N,pm.BLOCK_K,4,
                     4 if persistent(kind) else 3,2 if persistent(kind) else 0,
                     8 if grouped(kind) else 1,4)


def prepare_raw(kind, cfg, a, b, c):
    m,k = a.shape
    n = b.shape[1]
    tiles = triton.cdiv(m,cfg.block_m)*triton.cdiv(n,cfg.block_n)
    sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = min(tiles,sms*cfg.programs_per_sm) if persistent(kind) else tiles
    opts = dict(BLOCK_M=cfg.block_m,BLOCK_N=cfg.block_n,BLOCK_K=cfg.block_k,
                num_warps=cfg.num_warps,num_stages=cfg.num_stages,
                NUM_STAGES=cfg.loop_stages if cfg.loop_stages is not None else cfg.num_stages)
    if persistent(kind):
        opts['NUM_PROGRAMS'] = grid
    if grouped(kind):
        opts['GROUP_SIZE_M'] = cfg.group_size_m
    return partial(KERNELS[kind][(grid,)],a,b,c,m,n,k,*a.stride(),*b.stride(),*c.stride(),**opts), grid


def save(out, prefix, metadata, trials, comparison):
    winners = {}
    for kind in KINDS:
        candidates = [r for r in comparison if r['kind']==kind and r['status']=='ok']
        if candidates:
            winners[kind] = min(candidates,key=lambda r:r['median_ms'])
    reference = next((r for r in comparison if r['kind']=='torch' and r['status']=='ok'),None)
    for row in comparison:
        if row['status']=='ok' and reference:
            row['ratio_vs_torch'] = reference['median_ms']/row['median_ms']
    (out/f'{prefix}.json').write_text(json.dumps(dict(metadata=metadata,winners=winners,comparison=comparison),indent=2)+'\n')
    write_csv(out/f'{prefix}_trials.csv',trials)
    write_csv(out/f'{prefix}_comparison.csv',comparison)
    lines=['# Raw GEMM 확대 튜닝', '', json.dumps(metadata,ensure_ascii=False), '',
           '기본값과 새 후보를 같은 GPU/입력에서 순서를 섞어 재측정했다. TMA 참고 커널도 같은 실행에서 측정했다.',
           'BM/BN/BK, 요청 warps, launch/loop stages, programs/SM, GROUP 순서로 설정을 표시한다.', '',
           '| 구현 | 기본 μs | 선택 μs | 처리량 향상 | 선택 TFLOPS | 설정 |',
           '|---|---:|---:|---:|---:|---|']
    for kind,winner in winners.items():
        base = next((r for r in comparison if r['kind']==kind and r['phase']=='baseline' and r['status']=='ok'),None)
        cfg=winner['config']; loop=cfg.get('loop_stages') or cfg['num_stages']
        text=f"{cfg['block_m']}/{cfg['block_n']}/{cfg['block_k']}, W={cfg['num_warps']}, S={cfg['num_stages']}/{loop}, P={cfg['programs_per_sm']}, G={cfg['group_size_m']}"
        if base:
            lines.append(f"| {kind} | {base['median_ms']*1000:.2f} | {winner['median_ms']*1000:.2f} | {base['median_ms']/winner['median_ms']:.3f}× | {winner['tflops']:.1f} | {text} |")
    lines += ['', '| 같은 실행의 참고 구현 | μs | TFLOPS |', '|---|---:|---:|']
    for r in comparison:
        if r['phase']=='reference' and r['status']=='ok':
            lines.append(f"| {r['kind']} | {r['median_ms']*1000:.2f} | {r['tflops']:.1f} |")
    lines += ['', '선택값은 탐색한 범위 안의 관측값이다. 기본값이 더 빠르면 기본값을 유지한다.',
              '그룹 크기는 1차 GROUP=8 탐색의 상위 후보에서만 1/4/16/32로 확장했다.',
              '실행 상태와 실패 원인은 trials CSV에 기록한다. 정확도 실패는 즉시 중단한다.']
    (out/f'{prefix}.md').write_text('\n'.join(lines)+'\n')
    return winners


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size',type=int,default=4096)
    parser.add_argument('--preset',choices=('smoke','full'),default='full')
    parser.add_argument('--top-k',type=int,default=5)
    parser.add_argument('--rep-ms',type=int,default=20)
    parser.add_argument('--final-rep-ms',type=int,default=100)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--max-seconds',type=int,default=1500)
    args=parser.parse_args()
    if min(args.size,args.top_k,args.rep_ms,args.final_rep_ms,args.rounds,args.max_seconds)<=0 or args.size%8:
        parser.error('positive values required; size must be divisible by 8')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    out=Path(os.environ.get('OUT_DIR','tuning_results'));out.mkdir(parents=True,exist_ok=True)
    prefix='raw_tuning_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    props=torch.cuda.get_device_properties('cuda')
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    n=args.size
    a=torch.randn((n,n),device='cuda',dtype=torch.float16)
    b=torch.randn_like(a);c=torch.empty_like(a);expected=a@b
    metadata=dict(gpu=props.name,gpu_uuid=str(props.uuid),num_sms=props.multi_processor_count,
                  M=n,N=n,K=n,torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,
                  preset=args.preset,complete=False,seed=0,b_layout='contiguous K,N',dtype='float16',
                  fp16_reduced_precision_reduction=False,
                  kernel_sha256=hashlib.sha256(Path(pm.__file__).read_bytes()).hexdigest(),
                  search_space={kind:[asdict(x) for x in search_configs(kind,args.preset)] for kind in KINDS},
                  group_refinement=[1,4,16,32],top_k=args.top_k,rep_ms=args.rep_ms,
                  rounds=args.rounds,final_rep_ms=args.final_rep_ms,max_seconds=args.max_seconds,
                  timing='CUDA Graph, preallocated output, reused inputs; shuffled candidates and final rounds')
    trials,comparison=[],[]
    candidates={kind:[] for kind in KINDS};seen=set()
    rng=random.Random(0);start=time.monotonic();deadline=start+args.max_seconds

    def validate(kind,cfg,phase,run=None):
        row=dict(kind=kind,phase=phase,config=asdict(cfg) if cfg else {},
                 flatten=(kind=='cross_tile' if persistent(kind) else None) if kind in KINDS else (True if kind in ('ws_on','epilogue_ws') else None),
                 warp_specialize=kind in ('ws_on','epilogue_ws') if kind!='torch' else None,
                 epilogue_subtile=kind=='epilogue_ws' if kind!='torch' else None)
        if run is None:
            run,grid=prepare_raw(kind,cfg,a,b,c)
            row['grid']=grid
        try:
            c.fill_(float('nan'));kernel=run();torch.cuda.synchronize()
            torch.testing.assert_close(c,expected,atol=2e-2,rtol=1e-2)
            row['status']='ok'
            if kind!='torch':
                row.update(shared_bytes=kernel.metadata.shared,regs=kernel.n_regs,spills=kernel.n_spills,
                           compiled_num_warps=kernel.metadata.num_warps,compiled_num_stages=kernel.metadata.num_stages)
        except (triton.OutOfResources,triton.CompilationError) as error:
            row.update(status='compile_error',error=f'{type(error).__name__}: {error}')
        return row,run

    def screen(jobs,phase):
        rng.shuffle(jobs)
        for i,(kind,cfg) in enumerate(jobs,1):
            if time.monotonic()>=deadline:
                return False
            if (kind,cfg) in seen:
                continue
            seen.add((kind,cfg))
            row,run=validate(kind,cfg,phase)
            if row['status']=='ok':
                row.update(measure(run,args.rep_ms));row['tflops']=2*n**3/(row['median_ms']*1e9)
                candidates[kind].append((row['median_ms'],cfg))
            trials.append(row)
            if i%25==0 or i==len(jobs):
                print(f'{phase}: {i}/{len(jobs)}, elapsed={time.monotonic()-start:.0f}s',flush=True)
                save(out,prefix,metadata,trials,comparison)
        return True

    try:
        jobs=[(kind,cfg) for kind in KINDS for cfg in search_configs(kind,args.preset)]
        print(f'GPU={props.name}, size={n}, initial candidates={len(jobs)}',flush=True)
        if not screen(jobs,'screen'):
            return
        refinements=[(kind,replace(cfg,group_size_m=group)) for kind in KINDS if grouped(kind)
                     for _,cfg in sorted(candidates[kind],key=lambda x:x[0])[:args.top_k]
                     for group in (1,4,16,32)]
        if not screen(refinements,'group_refinement'):
            return
        runners=[]
        for kind in KINDS:
            choices=[('baseline',default_config(kind))]
            choices += [('final',cfg) for _,cfg in sorted(candidates[kind],key=lambda x:x[0])[:args.top_k]]
            for phase,cfg in choices:
                row,run=validate(kind,cfg,phase);comparison.append(row)
                if row['status']=='ok':
                    row['samples_ms']=[];runners.append((row,run))
        # Fresh PyTorch and previously validated TMA configurations in the same GPU run.
        references=[('torch',None,partial(torch.mm,a,b,out=c))]
        for kind,cfg in [('tma_tiled',Config(128,256,64,4,4)),('ws_on',Config(128,128,128,8,3,1))]:
            if torch.cuda.get_device_capability()[0]>=10:
                run,_=prepare(kind,cfg,a,b,c);references.append((kind,cfg,run))
        if torch.cuda.get_device_capability()[0]>=10:
            ds=(TensorDescriptor(a,[n,n],[n,1],[128,64]),TensorDescriptor(b,[n,n],[n,1],[64,256]),
                TensorDescriptor(c,[n,n],[n,1],[128,128]))
            programs=min(props.multi_processor_count,triton.cdiv(n,128)*triton.cdiv(n,256))
            run=partial(pm._tma_epilogue_matmul_kernel[(programs,)],*ds,n,n,n,NUM_PROGRAMS=programs,
                        BLOCK_M=128,BLOCK_N=256,BLOCK_K=64,WARP_SPECIALIZE=True,EPILOGUE_SUBTILE=True,
                        num_warps=4,num_stages=4)
            references.append(('epilogue_ws',Config(128,256,64,4,4,1),run))
        for kind,cfg,run in references:
            row,run=validate(kind,cfg,'reference',run);comparison.append(row)
            if row['status']=='ok':
                row['samples_ms']=[];runners.append((row,run))
        for repeat in range(args.rounds):
            rng.shuffle(runners)
            for row,run in runners:
                row['samples_ms'].append(measure(run,args.final_rep_ms)['median_ms'])
            print(f'Final comparison round {repeat+1}/{args.rounds}',flush=True)
        for row,_ in runners:
            row['median_ms']=statistics.median(row['samples_ms'])
            row['tflops']=2*n**3/(row['median_ms']*1e9)
        metadata['complete']=True
    finally:
        metadata['elapsed_seconds']=time.monotonic()-start
        for row in comparison:
            if row['status']=='ok' and 'median_ms' not in row:
                row['status']='validated_not_measured'
        winners=save(out,prefix,metadata,trials,comparison)
        for kind,row in winners.items():
            print(f"BEST {kind}: {row['median_ms']*1000:.2f} us, {row['tflops']:.1f} TFLOPS, {row['config']}",flush=True)
        print(f'Saved {out/prefix}, complete={metadata["complete"]}',flush=True)


if __name__=='__main__':
    main()
