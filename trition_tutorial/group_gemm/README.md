# Triton Group GEMM 구현 가이드

Group GEMM은 shape이 서로 다른 여러 행렬곱을 하나의 kernel launch로 처리한다.

```text
C0 = A0[M0, K0] @ B0[K0, N0]
C1 = A1[M1, K1] @ B1[K1, N1]
...
```

같은 shape을 batch 차원으로 묶는 batched GEMM과 다르다. MoE의 expert별 token 수처럼 각 문제의 M/N/K가 달라도 하나의 work queue로 합치는 것이 목적이다.

이 주제는 Persistent Matmul 다음에 하는 것이 좋다. 공식 구현도 고정된 CTA가 여러 tile을 stride 방식으로 가져가는 persistent scheduling을 사용한다.

## 파일과 첫 범위

```text
group_gemm/
├── group_gemm.py  # device-side scheduling kernel과 wrapper 뼈대
├── check.py       # 여러 torch.matmul 결과와 비교
├── benchmark.py   # Torch loop와 end-to-end 비교
└── README.md
```

공개 API는 다음과 같다.

```python
outputs = group_gemm(group_a, group_b)
```

첫 구현의 제약:

- 각 A/B는 contiguous FP16 2D tensor
- 문제마다 M/N/K가 달라도 됨
- M/N/K tail 지원
- output은 FP16, accumulator는 FP32
- 한 kernel launch
- TMA, FP8, autotune은 제외

## 왜 그냥 Python loop가 아닌가

```python
outputs = [torch.matmul(a, b) for a, b in zip(group_a, group_b)]
```

이 코드는 문제마다 별도 library/kernel launch가 필요하다. GEMM 하나가 작거나 expert 수가 많으면 launch와 host scheduling 비중이 커진다.

Group GEMM은 모든 문제의 output tile을 하나의 연속된 논리 공간으로 본다.

```text
problem 0: tile 0 ... tile T0-1
problem 1: tile T0 ... tile T0+T1-1
problem 2: ...
```

고정된 program `p`는 `p, p + NUM_PROGRAMS, p + 2*NUM_PROGRAMS, ...` tile을 처리한다. 각 tile이 어느 problem에 속하는지는 kernel이 device metadata를 읽어 판단한다.

## host metadata

Python의 tensor list 자체를 Triton kernel에 넘길 수 없다. wrapper가 다음 device tensor를 만든다.

| metadata | shape | 내용 |
|---|---:|---|
| `group_a_ptrs` | `[G]` | 각 A의 64-bit device address |
| `group_b_ptrs` | `[G]` | 각 B의 address |
| `group_c_ptrs` | `[G]` | 각 C의 address |
| `group_sizes` | `[G, 3]` | 각 `(M, N, K)` |

첫 뼈대는 모든 tensor를 contiguous로 제한하므로 stride metadata를 따로 만들지 않는다. A의 row stride는 `K`, B와 C의 row stride는 `N`에서 바로 얻는다. pointer tensor가 살아 있는 동안 원래 A/B/C tensor도 반드시 살아 있어야 한다. pointer 숫자만 남는다고 tensor 수명이 자동 보장되는 것은 아니다.

kernel에서는 주소를 다시 typed pointer로 복원한다.

```python
a_ptr = tl.load(group_a_ptrs + group_idx)
a_ptr = a_ptr.to(tl.pointer_type(tl.float16))
```

## flattened tile scheduling

문제 `g`의 tile 수는 다음과 같다.

```text
Tm[g] = ceil(M[g] / BLOCK_M)
Tn[g] = ceil(N[g] / BLOCK_N)
T[g]  = Tm[g] * Tn[g]
```

`problem_start`는 앞 문제들의 tile 수 누적합이다.

```text
problem_start <= tile_id < problem_start + T[g]
```

이면 현재 tile이 문제 g에 속한다. 문제 내부 좌표는:

```python
tile_in_problem = tile_id - problem_start
tile_m = tile_in_problem // num_n_tiles
tile_n = tile_in_problem % num_n_tiles
```

현재 program이 이 problem의 tile을 처리한 뒤 `tile_id += NUM_PROGRAMS`한다. 다음 tile이 같은 problem일 수도, 뒤 problem일 수도 있다.

## 1단계: full-tile Group GEMM

처음에는 `check.py`에서 앞의 두 shape만 남겨 block 배수 문제부터 맞춘다.

`_group_gemm_kernel`의 TODO를 다음 순서로 채운다.

1. A/B/C raw address load와 FP16 pointer cast
2. 현재 `M/N/K`와 `tile_m/tile_n`에서 pointer 계산
3. K loop와 FP32 `tl.dot`
4. C store

시작 config:

```python
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
num_warps = 4
num_stages = 2
```

wrapper의 첫 grid는 다음처럼 둔다.

```python
num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
num_programs = num_sms
grid = (num_programs,)
```

kernel의 `NUM_PROGRAMS`에도 같은 값을 전달한다.

## 2단계: 모든 tail 지원

공식 raw-pointer 예제의 첫 kernel은 설명을 단순하게 하려고 full tile을 가정한다. 이 실습에서는 그 다음 단계로 일반적인 M/N/K tail을 붙인다.

```python
offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = start_k + tl.arange(0, BLOCK_K)

a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
```

A/B의 범위 밖은 0으로 load한다. `check.py`의 뒤 두 irregular shape를 다시 활성화해 검증한다.

## 3단계: launch overhead를 분리해 측정

전체 Python 함수 시간을 재면 매 호출마다 pointer/size/stride tensor를 만드는 비용도 포함된다. 두 값을 따로 봐야 한다.

1. end-to-end: metadata 준비 + kernel
2. kernel-only: metadata를 한 번 준비하고 kernel launch만 반복

비교 대상:

- Python loop의 `torch.matmul`
- Group GEMM end-to-end
- Group GEMM kernel-only

기본 end-to-end benchmark는 다음처럼 실행한다.

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/group_gemm/benchmark.py \
  --gpu B200
```

작은 GEMM이 많을수록 Group GEMM의 목적이 잘 드러난다. 큰 GEMM 몇 개만 있으면 vendor library의 개별 호출이 더 빠를 수 있다.

## 4단계: scheduling과 config 최적화

정확도 뒤에 한 변수씩 실험한다.

1. `NUM_PROGRAMS`: SM 수의 1/2, 1배
2. `BLOCK_M/BLOCK_N`: 64와 128
3. `BLOCK_K`: 32와 64
4. `num_warps`, `num_stages`
5. 문제 목록을 큰 순서/작은 순서로 재배치

정적 stride scheduling은 단순하지만 problem tile 크기가 매우 불균형하면 program별 일이 고르지 않을 수 있다. 이것은 numerical bug가 아니라 load imbalance다.

## 검증

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/group_gemm/check.py \
  --gpu B200
```

현재 wrapper는 의도적으로 `NotImplementedError`를 낸다. kernel TODO와 launch를 완성하면 네 개의 서로 다른 GEMM 결과가 출력된다.

## 이후 공식 튜토리얼 경로

raw-pointer 구현이 통과한 뒤:

1. problem별 stride metadata를 추가해 non-contiguous layout 지원
2. `@triton.autotune(key=["group_size"])`
3. Hopper 이상에서 problem별 `tl.make_tensor_descriptor`
4. B를 `[N, K]`로 저장해 descriptor-friendly load
5. FP8

TMA 버전에서도 flattened scheduling은 그대로고, A/B/C tile을 읽고 쓰는 방식만 descriptor로 바뀐다.

## 자주 생기는 오류

- Python list 자체를 kernel argument로 넘기려 함
- device address를 FP16 pointer로 cast하지 않음
- `problem_start`를 현재 problem 뒤에서 갱신하지 않음
- problem 내부 tile id와 전체 flattened tile id를 혼동함
- `tile_id += NUM_PROGRAMS`를 빠뜨려 loop가 끝나지 않음
- contiguous 제한을 제거했는데 problem별 stride metadata를 추가하지 않음
- metadata 준비 시간을 kernel 성능으로 잘못 해석함

## 완료 조건

- 여러 shape 결과가 각 `torch.matmul`과 일치
- M/N/K irregular tail 통과
- GEMM 수와 관계없이 kernel launch가 한 번
- 고정 program들이 flattened tile 공간을 stride 방식으로 처리
- end-to-end와 kernel-only 시간을 분리해 기록

## 참고

- [Triton Group GEMM 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
- [Triton Persistent Matmul 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
