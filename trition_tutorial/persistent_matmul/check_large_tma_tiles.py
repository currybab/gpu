"""Bounded experiment: 3 tiles x 2 stages x 4 TMA-family kernels, no autotune."""

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
from tune_tma import Config, measure, prepare, write_csv

TILES = ((128, 256, 64), (256, 256, 32), (256, 256, 64))
KINDS = ('tma_tiled', 'tma_persistent', 'ws_on', 'ws_subtile')


@torch.no_grad()
def main():
    out = Path(os.environ.get('OUT_DIR', 'tuning_results'))
    out.mkdir(parents=True, exist_ok=True)
    prefix = 'large_tma_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    size = 4096
    a = torch.randn((size,size),device='cuda',dtype=torch.float16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    expected = a @ b
    props = torch.cuda.get_device_properties('cuda')
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError('Run this WS comparison on Blackwell, e.g. B200.')
    rows, runners = [], []
    for kind, tile, stages in itertools.product(KINDS, TILES, (3,4)):
        bm,bn,bk = tile
        row = dict(kind=kind,block_m=bm,block_n=bn,block_k=bk,num_warps=8,num_stages=stages,
                   programs_per_sm=0 if kind=='tma_tiled' else 1,
                   flatten=None if kind=='tma_tiled' else True,
                   warp_specialize=kind in ('ws_on','ws_subtile'),epilogue_subtile=kind=='ws_subtile')
        try:
            if kind != 'ws_subtile':
                run, grid = prepare(kind,Config(bm,bn,bk,8,stages,row['programs_per_sm']),a,b,c)
            else:
                ds = (TensorDescriptor(a,[size,size],[size,1],[bm,bk]),
                      TensorDescriptor(b,[size,size],[size,1],[bk,bn]),
                      TensorDescriptor(c,[size,size],[size,1],[bm,bn//2]))
                grid = min(props.multi_processor_count,triton.cdiv(size,bm)*triton.cdiv(size,bn))
                run = partial(pm._tma_epilogue_matmul_kernel[(grid,)],*ds,size,size,size,
                              NUM_PROGRAMS=grid,BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,
                              WARP_SPECIALIZE=True,EPILOGUE_SUBTILE=True,num_warps=8,num_stages=stages)
            row['grid'] = grid
            c.fill_(float('nan'))
            kernel = run()
            torch.cuda.synchronize()
            torch.testing.assert_close(c,expected,atol=2e-2,rtol=1e-2)
            row.update(status='ok',shared_bytes=kernel.metadata.shared,regs=kernel.n_regs,
                       spills=kernel.n_spills,compiled_num_warps=kernel.metadata.num_warps,samples_ms=[])
            runners.append((row,run))
        except (triton.OutOfResources,triton.CompilationError) as error:
            row.update(status='compile_error',error=str(error))
        rows.append(row)
        print(kind,tile,'stages=',stages,row['status'],flush=True)
    reference = dict(kind='torch',samples_ms=[])
    runners.append((reference,partial(torch.mm,a,b,out=c)))
    rng = random.Random(0)
    for repeat in range(3):
        rng.shuffle(runners)
        for row,run in runners:
            row['samples_ms'].append(measure(run,50)['median_ms'])
        print(f'Round {repeat+1}/3 complete',flush=True)
    for row,_ in runners:
        row['median_ms'] = statistics.median(row['samples_ms'])
        row['tflops'] = 2*size**3/(row['median_ms']*1e9)
    for row in rows:
        if row['status']=='ok':
            row['ratio_vs_torch'] = reference['median_ms']/row['median_ms']
    metadata = dict(gpu=props.name,gpu_uuid=str(props.uuid),M=size,N=size,K=size,dtype='float16',
                    torch=torch.__version__,triton=triton.__version__,num_warps=8,programs_per_sm=1,
                    tiles=TILES,stages=[3,4],rounds=3,rep_ms=50,case_count=24,
                    kernel_sha256=hashlib.sha256(Path(pm.__file__).read_bytes()).hexdigest(),
                    fp16_reduced_precision_reduction=False,
                    timing='Same GPU, preallocated CUDA Graph, shuffled 3 x 50ms medians, reused input')
    (out/f'{prefix}.json').write_text(json.dumps(dict(metadata=metadata,torch=reference,results=rows),indent=2)+'\n')
    write_csv(out/f'{prefix}.csv',rows)
    lines=['# Large TMA tile check','',json.dumps(metadata,ensure_ascii=False),'',
           '| Kernel | BM/BN/BK | Stages | Status | μs | TFLOPS | Shared KiB |',
           '|---|---|---:|---|---:|---:|---:|']
    for r in rows:
        result=(f"{r['median_ms']*1000:.2f} | {r['tflops']:.1f} | {r['shared_bytes']/1024:.2f}"
                if r['status']=='ok' else '— | — | —')
        line=f"| {r['kind']} | {r['block_m']}/{r['block_n']}/{r['block_k']} | {r['num_stages']} | {r['status']} | {result} |"
        lines.append(line);print(line,flush=True)
    lines += ['', 'Only the stated 24 cases were tested. Failure here does not mean a tile cannot run with other stages/warps.',
              'This compares tile choices at fixed warps=8 and persistent programs/SM=1; it does not replace earlier independently tuned settings.']
    (out/f'{prefix}.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
