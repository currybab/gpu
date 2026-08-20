# Triton Tutorial 학습 가이드

이 폴더에는 Triton kernel을 기본 구현부터 correctness 검증과 성능 최적화까지 단계적으로 연습할 자료가 있다. 각 주제는 가능한 한 다음 파일 구조를 따른다.

```text
topic/
├── README.md       # 개념, 구현 순서, 최적화 단계
├── <topic>.py      # TODO가 포함된 kernel/wrapper 뼈대
├── check.py        # PyTorch reference와 정확도 비교
└── benchmark.py    # 기준 구현과 runtime/처리량 비교
```

## 구현 가이드

| 주제 | 핵심 내용 | 자료 |
|---|---|---|
| Fused Softmax | 1D row mapping, reduction, register/occupancy 측정 | [코드와 측정 기록](./fused_softmax/) |
| Fused Attention | tiled QKᵀ/PV, online softmax, causal/tail mask | [구현 가이드](./fused_attention/) |
| Persistent Matmul | 기본 tiled GEMM, grouped ordering, persistent scheduling | [구현 가이드](./persistent_matmul/) |
| Group GEMM | 여러 shape의 GEMM metadata와 device-side scheduling | [구현 가이드](./group_gemm/) |
| Block-scaled Matmul | explicit scale fusion, FP8/FP4, `tl.dot_scaled` | [구현 가이드](./block_scaled_matmul/) |

## 권장 학습 순서

공통 기초는 다음 순서다.

```text
vector add → fused softmax → 기본 tiled matmul
```

그 뒤에는 목표에 따라 나눈다.

- Attention 트랙: `기본 matmul + fused softmax → fused attention`
- GEMM scheduling 트랙: `기본 matmul → persistent matmul → group GEMM`
- 저정밀 하드웨어 트랙: `기본 matmul → block-scaled matmul`

Group GEMM과 Block-scaled Matmul은 Fused Attention의 선수 과목이 아니다. attention 구현이 현재 목표라면 Fused Attention을 계속하면 된다.

기본 matmul의 pointer 계산, K loop, tail mask가 아직 낯설다면 [Persistent Matmul 가이드](./persistent_matmul/)의 기본 GEMM 단계까지만 먼저 구현하고 돌아온다. Triton의 scheduling 최적화를 체계적으로 익히려면 Persistent Matmul 다음 Group GEMM을 진행한다. Block-scaled Matmul은 FP8/FP4와 Blackwell 전용 개념이 추가되므로 가장 나중에 보는 편이 좋다.

## 자료를 사용하는 방법

각 주제는 다음 순서로 진행한다.

1. README에서 첫 구현 범위와 tensor shape을 확인한다.
2. 구현 파일의 TODO를 한 단계만 채운다.
3. `check.py`에서 해당 단계의 작은 shape를 통과시킨다.
4. full tile 뒤에 M/N/K 또는 sequence tail case를 추가한다.
5. correctness가 모두 맞은 뒤 `benchmark.py`를 실행한다.
6. block 크기나 scheduling 변수를 하나씩 바꾸며 결과를 기록한다.
7. 마지막에 TMA, TensorDescriptor, autotune 같은 하드웨어 최적화를 붙인다.

속도가 빠르더라도 correctness를 통과하지 않은 kernel은 측정하지 않는다. 최적화할 때는 여러 변수를 동시에 바꾸지 않고 기준 kernel과 한 가지 차이만 둔다.

## Modal에서 실행하기

명령은 저장소 루트에서 실행한다. 최초 1회 `uv run modal setup`이 필요하다.

### Fused Softmax

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/fused_softmax/fused_softmax.py \
  --gpu B200
```

### Fused Attention

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/fused_attention/check.py \
  --gpu B200

uv run modal run modal_run.py \
  --script trition_tutorial/fused_attention/benchmark.py \
  --gpu B200
```

### Persistent Matmul

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/persistent_matmul/check.py \
  --gpu B200

uv run modal run modal_run.py \
  --script trition_tutorial/persistent_matmul/benchmark.py \
  --gpu B200
```

### Group GEMM

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/group_gemm/check.py \
  --gpu B200

uv run modal run modal_run.py \
  --script trition_tutorial/group_gemm/benchmark.py \
  --gpu B200
```

### Block-scaled Matmul

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/block_scaled_matmul/check.py \
  --gpu B200

uv run modal run modal_run.py \
  --script trition_tutorial/block_scaled_matmul/benchmark.py \
  --gpu B200
```

`--diagnostics`는 지원하는 스크립트에서 Triton register/spill 정보와 Torch CUDA profiler 결과를 함께 확인할 때 사용한다.
