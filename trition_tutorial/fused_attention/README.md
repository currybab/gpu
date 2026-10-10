# Triton Fused Attention 개념별 구현 가이드

Attention 계산과 GPU 최적화에 필요한 개념을 직접 구현해 보는 실습이다.
기본 forward에서는 다음 계산을
`하나의 Triton forward kernel`에 담는다.

```text
QK^T -> scale / causal mask -> online softmax -> PV -> O
```

전체 `N x N` score와 probability를 HBM에 저장하지 않는 것이 핵심이다. 첫 구현은 raw pointer와 `tl.load`/`tl.store`만 사용한다. 확장 단계도 커널 구조와 launch는 제공하지만 핵심 계산은 TODO로 남겨두었다. FP8은 후속 과제로 남긴다.

## 개념과 실습 코드

실습 개수나 순서를 채우는 것이 목표는 아니다. 공통 기반은 tile 단위 QKᵀ/PV,
online softmax, mask와 수치 안정성이다. 그 기반을 이해한 뒤 관심 있는 주제를
선택한다.

- 학습 연산: LSE를 이용한 probability 재계산과 backward.
- 데이터 이동: raw pointer와 TMA descriptor, load/계산 overlap과 warp specialization.
- 작업 배치와 튜닝: causal 계산 생략, persistent scheduling, tile/warp/stage 선택.

backward를 이해하는 데 TMA나 persistent를 먼저 구현할 필요는 없다.
이 저장소의 backward 뼈대는 코드 재사용을 위해 raw split forward에 연결되어 있다.
TMA와 WS는 그와 별개로 선택하는 하드웨어 최적화 실습이다.

| `--impl` | 학습할 내용 | 구현 위치 | 입력 범위 |
|---|---|---|---|
| `baseline` | stride, tail mask, online softmax | `fused_attention.py` | FP16, 임의 stride, N tail, D=16~128 |
| `split` | causal 과거/대각 구간 분리 | `attention_variants.py`: SPLIT | contiguous, N/D tail |
| `persistent` | 제한된 program이 query tile 여러 개 처리 | 같은 파일: PERSISTENT | split과 동일 |
| `tma` | host TensorDescriptor load/store | 같은 파일: TMA | Hopper 이상, contiguous, N=128의 배수, D=64/128 |
| `tma-ws` | K/V loop의 자동 warp specialization | 같은 파일: WS | Blackwell, TMA와 동일 |
| `autotune` | BLOCK_M/N, warp/stage 탐색 | 같은 파일: RAW_CONFIGS | split과 동일 |
| `train` | LSE 저장, backward, autograd 연결 | `attention_backward.py`: BWD | contiguous, N/D tail, 1차 gradient |

모든 변형은 같은 self-attention 수식을 계산한다. `split/persistent/autotune/train`은
raw forward 본문을 공유하고, `tma/tma-ws`는 descriptor 본문을 공유한다.
TMA·persistent·WS는 서로 다른 선택이다. 데이터 이동 방식, program의 작업 배치,
warp의 역할 분할을 각각 바꾼다. persistent가 항상 빠르다는 가정은 하지 않는다.

## 어디서 시작할까

Fused Attention에 필요한 직접적인 선수 지식은 tiled matmul의 pointer/K loop와 fused softmax다. Persistent Matmul, Group GEMM, Block-scaled Matmul은 필수 선수 과목이 아니다.

- 기본 matmul의 `tl.dot`, pointer, tail mask가 익숙하면 이 가이드를 계속 진행한다.
- 그 부분이 아직 낯설면 [Persistent Matmul 가이드의 기본 GEMM 단계](../persistent_matmul/)까지만 먼저 구현하고 돌아온다.
- Group GEMM과 Block-scaled Matmul은 각각 scheduling과 저정밀 하드웨어를 다루는 별도 트랙이다.

## 파일 구성

```text
fused_attention/
├── fused_attention.py     # stride 기반 forward 뼈대
├── attention_variants.py  # forward 최적화 실습과 launch
├── attention_backward.py  # preprocess/dQ/dK+dV와 autograd 뼈대
├── implementations.py     # --impl 이름, 하드웨어 조건
├── check.py               # FP32 reference, forward/gradient 검사
├── benchmark.py           # 검증한 forward와 Torch SDPA 비교
└── README.md              # 구현 가이드
```

`fused_attention.py`의 공개 함수는 다음 시그니처를 유지한다.

```python
def attention(q, k, v, causal: bool, sm_scale: float):
    ...
```

`check.py`가 이 함수를 PyTorch reference와 비교한다.

각 확장 wrapper의 `raise NotImplementedError`는 미완성 커널 실행을 막는 장치다.
메시지에 적힌 TODO를 채운 뒤 **해당 wrapper의 raise만** 지운다. launch 코드는
그 바로 아래에 준비되어 있다. 새 변형을 추가할 때 이전 단계를 지울 필요가 없다.

```bash
# 로컬: 구현한 단계만 선택 (기본값은 baseline)
.venv/bin/python trition_tutorial/fused_attention/check.py --impl split persistent
.venv/bin/python trition_tutorial/fused_attention/check.py --impl tma tma-ws
.venv/bin/python trition_tutorial/fused_attention/check.py --impl train
.venv/bin/python trition_tutorial/fused_attention/benchmark.py --impl split tma

# 전체 진행 상황: TODO 또는 전부 SKIP이면 종료 코드 1
.venv/bin/python trition_tutorial/fused_attention/check.py --impl all

# Modal에도 같은 인자 전달
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/fused_attention/check.py \
  --script-args '--impl tma tma-ws'
```

`check.py`의 TODO는 미구현, SKIP은 하드웨어 미지원이다. 실제 수치 불일치나
컴파일 오류는 실패로 그대로 전파한다. `benchmark.py`는 미구현을 제외하고
correctness를 통과한 구현만 측정하며, 하나도 없으면 종료 코드 1을 반환한다.

## 1차 구현 범위

입력 layout은 `[B, H, N, D]`다.

| 기호 | 의미 |
|---|---|
| `B` | batch 크기 |
| `H` | attention head 수 |
| `N` | sequence length |
| `D` | head dimension |
| `Q, K, V` | `[B, H, N, D]` |
| `O` | attention 출력, `[B, H, N, D]` |

처음에는 다음 범위만 구현한다.

- CUDA FP16 self-attention
- Q/K/V shape이 모두 `[B, H, N, D]`
- Q/K/V/O의 `[B, H, N, D]` stride를 kernel argument로 전달
- `16 <= D <= 128`
- `N`이 tile 크기의 배수가 아닌 경우까지 처리
- non-causal과 causal forward
- backward가 필요 없는 추론용 함수

다음 기능은 첫 구현에서 제외한다.

- backward와 `torch.autograd.Function`
- dropout, attention bias
- GQA/MQA, cross attention, variable length
- BF16/FP8
- TensorDescriptor/TMA
- autotune과 warp specialization

여기서 “일반적인 구현”은 모든 attention 변형을 지원한다는 뜻이 아니다. 특정 GPU 세대의 descriptor API에 의존하지 않는 기본 FlashAttention-style kernel을 뜻한다.

## 정답과 실행 방법

`check.py`의 reference는 아래 수식을 그대로 계산한다.

```text
S = Q @ K^T * sm_scale
P = softmax(S, dim=-1)
O = P @ V
```

causal이면 query 위치 `i`가 미래 key `j > i`를 보지 못하도록 `S[i, j] = -inf`를 적용한다.

Modal에서는 저장소 루트에서 실행한다.

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/fused_attention/check.py \
  --gpu B200
```

뼈대가 미완성이면 `NotImplementedError` 또는 정확도 assertion 실패가 정상이다. 구현 후에는 각 shape의 non-causal과 causal 결과에서 최대 절대 오차가 출력된다. 초기 허용 오차는 `atol=1e-2, rtol=0`이다.

`check.py`는 구현 단계의 경계를 잡을 수 있도록 다음 case를 포함한다.

| shape `[B, H, N, D]` | 확인할 내용 |
|---|---|
| `[1, 2, 17, 32]` | tile보다 짧은 sequence |
| `[1, 2, 128, 64]` | 기본 block과 causal/non-causal |
| `[2, 3, 129, 64]` | batch/head decode, sequence tail, padded row stride |
| `[1, 2, 129, 80]` | `BLOCK_D=128` padding과 이중 tail |
| `[1, 2, 1000, 128]` | 여러 K/V block의 누적 오차 |
| `[4, 16, 257, 64]` | 많은 query tile과 persistent scheduling |

확장 raw 단계는 위 shape를 contiguous로 검사한다. TMA 단계는 별도의 full-tile
case를 사용한다. `train`은 forward 외에 `dQ/dK/dV`도 검사하며 gradient 허용 오차는
`atol=3e-2, rtol=1e-2`다. reference는 Q/K/V를 FP32로 올려 계산한다.

처음에는 첫 case만 남기고, 구현 순서에 맞춰 뒤 case를 다시 활성화해도 된다.

## 왜 fused하는가

naive attention은 `S`와 `P`라는 `[B, H, N, N]` 중간 텐서를 만든다. 예를 들어 `B=4, H=32, N=4096`이면 행렬 하나만 해도 다음 크기다.

```text
FP16: 4 * 32 * 4096 * 4096 * 2 bytes = 4 GiB
FP32: 4 * 32 * 4096 * 4096 * 4 bytes = 8 GiB
```

fused attention은 Q 일부 행을 register/SRAM에 유지하고 K/V block을 차례로 읽는다. 각 score tile을 즉시 softmax 상태와 output accumulator에 반영한 뒤 버리므로 전체 score/probability 행렬을 HBM에 만들지 않는다.

```text
naive:
Q, K -> S 저장 -> softmax -> P 저장 -> P @ V -> O

fused:
Q tile 유지 -> K/V tile 순회 -> online softmax + PV 누적 -> O만 저장
```

수학적으로는 같은 softmax attention이다. 연산 순서와 부동소수점 반올림 때문에 PyTorch와 bitwise identical하지는 않다.

## 핵심: online softmax

한 query row의 모든 score를 동시에 보지 않고도 softmax를 계산할 수 있다. 지금까지 처리한 key block의 상태를 세 값으로 유지한다.

```text
m_i   : 지금까지 본 score의 row-wise maximum
l_i   : exp(score - m_i)의 row-wise sum
acc_i : exp(score - m_i) @ V의 누적값
```

새 score block `S_ij`를 만날 때 다음처럼 합친다.

```text
m_new = max(m_old, rowmax(S_ij))
alpha = exp(m_old - m_new)
P_ij  = exp(S_ij - m_new)

l_new   = alpha * l_old + rowsum(P_ij)
acc_new = alpha * acc_old + P_ij @ V_j
```

모든 key block을 처리한 뒤 `O_i = acc_i / l_i`다. 초깃값은 `m_i=-inf`, `l_i=0`, `acc_i=0`이다. maximum이 커지면 기존 누적값의 기준도 바뀌므로 `l_i`와 `acc` 모두 `alpha`를 곱해야 한다.

Triton에서는 보통 `tl.exp2`를 사용한다.

```text
exp(x) = 2^(x * log2(e))
qk_scale = sm_scale * 1.4426950408889634
```

첫 forward 구현은 backward용 log-sum-exp를 저장할 필요가 없다. 나중에 backward를 추가할 때 `M = m_i + log2(l_i)`를 별도 buffer에 저장한다.

## program과 tile 배치

program 하나가 한 `(batch, head)`의 `BLOCK_M`개 query row를 담당한다.

```python
grid = (triton.cdiv(N, BLOCK_M), B * H)
```

grid에서는 `B * H`개의 head를 하나의 축으로 펴지만, 주소는 `N_HEADS`와 tensor별 batch/head stride를 사용해 계산한다.

```python
query_block = tl.program_id(0)
batch_head = tl.program_id(1)
batch_idx = batch_head // N_HEADS
head_idx = batch_head % N_HEADS
q_head_offset = batch_idx * stride_qb + head_idx * stride_qh
```

program 내부 tile은 다음 shape을 가진다.

| 값 | shape | 역할 |
|---|---|---|
| `q` | `[BLOCK_M, BLOCK_D]` | K/V loop 동안 유지 |
| `k` | `[BLOCK_N, BLOCK_D]` | 현재 key block |
| `qk` | `[BLOCK_M, BLOCK_N]` | 현재 score block |
| `v` | `[BLOCK_N, BLOCK_D]` | 현재 value block |
| `acc` | `[BLOCK_M, BLOCK_D]` | FP32 output numerator |
| `m_i, l_i` | `[BLOCK_M]` | FP32 softmax 상태 |

처음에는 튜닝하지 않고 고정한다.

```python
BLOCK_M = 64
BLOCK_N = 64
BLOCK_D = triton.next_power_of_2(D)
num_warps = 4
num_stages = 2
```

`D`가 power of two가 아니면 `BLOCK_D`까지 padding해서 load하고 `offs_d < D` mask를 적용한다.

## pointer와 stride로 주소 계산하기

Q/K/V는 shape이 같아도 storage layout이 같다는 보장이 없다. 따라서 일반적인 Triton raw-pointer kernel처럼 각 tensor의 네 축 stride를 별도로 받는다. Q tile은 다음 형태다.

```python
offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
offs_d = tl.arange(0, BLOCK_D)

q_ptrs = (
    q_ptr
    + q_head_offset
    + offs_m[:, None] * stride_qm
    + offs_d[None, :] * stride_qd
)
q = tl.load(
    q_ptrs,
    mask=(offs_m[:, None] < N) & (offs_d[None, :] < D),
    other=0.0,
)
```

K/V는 key loop의 `offs_n`을 사용한다.

```python
offs_n = start_n + tl.arange(0, BLOCK_N)

k_ptrs = (
    k_ptr
    + k_head_offset
    + offs_n[:, None] * stride_kn
    + offs_d[None, :] * stride_kd
)
k = tl.load(
    k_ptrs,
    mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
    other=0.0,
)
```

인자가 길어 보이지만 역할은 단순하다. `*_ptr`는 storage 시작 주소이고 `stride_*`는 해당 축에서 한 칸 움직일 때 건너뛸 원소 수다. 이렇게 두면 online softmax 구현과 memory layout 가정을 분리할 수 있다.

## tail mask와 causal mask

`N`이 tile 크기의 배수가 아니면 query와 key의 유효성을 구분한다.

```python
query_valid = offs_m < N
key_valid = offs_n < N
score_valid = key_valid[None, :]

if CAUSAL:
    score_valid &= offs_m[:, None] >= offs_n[None, :]

qk = tl.where(score_valid, qk, -float("inf"))
```

`query_valid`를 score mask에 넣어 invalid query row 전체를 `-inf`로 만들면 row maximum도 `-inf`가 되어 `-inf - -inf`에서 NaN이 생길 수 있다. invalid query row는 계산 중에는 그대로 두고, 마지막 output store에서 `offs_m < N` mask로 버리는 편이 단순하다.

K/V tail은 load mask로 0을 채우되, 해당 key의 score는 반드시 `-inf`로 제외해야 한다. 0으로 읽은 가짜 K가 softmax에 참여하면 결과가 달라진다.

## forward kernel 뼈대

`fused_attention.py`에는 program mapping, online-softmax 상태, K/V loop, wrapper 설정까지 들어 있다. 긴 완성 kernel을 README와 코드에 중복하지 않고 다음 TODO 경계만 제공한다.

| TODO | 구현 내용 |
|---:|---|
| 1 | Q tile을 loop 밖에서 한 번 load |
| 2 | 현재 K tile과 tail mask |
| 3 | `QKᵀ * sm_scale * log2(e)` |
| 4 | key tail과 optional causal score mask |
| 5 | `m_i/l_i/acc` online-softmax 갱신 |
| 6 | V tile load와 `P @ V` 누적 |
| 7 | normalize 후 O store |
| 8 | Python wrapper에서 kernel launch |

구현할 때 지킬 점:

- Q는 key loop 밖에서 한 번만 load한다.
- `m_i`, `l_i`, `acc`는 FP32로 둔다.
- Tensor Core를 사용하도록 `p`는 `tl.dot` 직전에 FP16으로 내린다.
- K/V load에는 token tail과 head-dimension tail mask를 모두 적용한다.
- 첫 버전은 causal 최적화를 하지 않고 모든 K block을 순회해도 된다.
- forward 전체는 한 번의 kernel launch다.

## Python wrapper 뼈대

`attention()`은 입력 검증, output 할당, meta-parameter 계산, kernel launch만 담당한다.

```python
def attention(q, k, v, causal: bool, sm_scale: float):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape == k.shape == v.shape
    batch, n_heads, n_ctx, head_dim = q.shape
    assert 16 <= head_dim <= 128

    out = torch.empty_like(q)
    block_m = 64
    block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(n_ctx, block_m), batch * n_heads)

    _attention_fwd[grid](
        q, k, v, out,
        n_ctx,
        sm_scale,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        N_HEADS=n_heads,
        HEAD_DIM=head_dim,
        CAUSAL=causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out
```

`causal`은 compile-time meta-parameter로 넘겨 causal/non-causal kernel이 각각 컴파일되게 한다.

## 권장 구현 순서

### 1. Q를 O로 복사

동일한 grid와 Q/O pointer 계산만 구현해 `out == q`를 확인한다. batch/head stride, token tail, `D` padding을 먼저 검증한다.

### 2. K/V block 하나만 계산

작은 `N <= BLOCK_N`에서 non-causal `QK^T -> softmax -> PV`를 계산한다. 아직 key loop를 여러 번 돌지 않는다.

### 3. 여러 K/V block을 합치기

`N > BLOCK_N`으로 바꾸고 `m_i`, `l_i`, `acc`의 online softmax 갱신을 구현한다. 이 단계가 핵심이다.

### 4. sequence tail 처리

`N=129`, `N=1000`처럼 block 배수가 아닌 입력을 추가한다. K/V load mask와 score `-inf` mask를 함께 확인한다.

### 5. causal mask 추가

처음에는 모든 K block을 순회하며 위치 비교 mask를 적용한다. 정확도가 맞은 뒤 현재 query tile보다 완전히 미래인 K block을 건너뛰는 최적화를 추가한다.

## 자주 생기는 오류

- `tl.dot(q, k)`를 호출함: K는 `[BLOCK_N, D]`이므로 `tl.trans(k)`가 필요하다.
- `tl.exp2`를 쓰면서 `sm_scale * log2(e)` 변환을 빼먹음.
- maximum이 바뀔 때 `acc`만 rescale하고 `l_i`를 rescale하지 않음.
- K tail을 0으로 load했지만 가짜 key score를 `-inf`로 제거하지 않음.
- invalid query row를 전부 `-inf` 처리해 NaN을 만듦.
- `acc`를 FP16으로 두어 sequence가 길수록 오차가 커짐.
- Q를 key loop 안에서 매번 다시 load함.
- causal 조건을 반대로 씀. 허용 조건은 `query_index >= key_index`다.
- output store mask에서 `offs_d < D`를 빼먹음.
- correctness 전에 autotune을 붙여 문제 원인을 구분하기 어렵게 만듦.

## 1차 완료 조건

- `check.py`의 non-causal test 통과
- `check.py`의 causal test 통과
- `N`이 `BLOCK_M/BLOCK_N`의 배수가 아닌 test 통과
- `D`가 power of two가 아닌 경우를 지원하거나 wrapper에서 명시적으로 거부
- 전체 `N x N` score/probability tensor를 만들지 않음
- forward가 한 Triton kernel launch로 실행됨

## 성능 측정

정확도를 모두 통과한 뒤 Torch SDPA와 같은 shape에서 비교한다.

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/fused_attention/benchmark.py \
  --gpu B200
```

`benchmark.py`는 먼저 `check_attention()`을 실행하고 실제 측정 shape도 검증한다.
`train`은 backward correctness까지 검사하되 시간은 forward만 측정한다.
그다음 causal/non-causal에서 다음을 측정한다.

- `torch.nn.functional.scaled_dot_product_attention`
- 직접 구현한 Triton `attention`

시간은 출력 할당과 descriptor 생성 등을 포함하는 wrapper 호출 기준이며,
autotune의 첫 탐색은 측정 전에 끝낸다. 순수 kernel 시간 비교와는 구분한다.

forward의 주요 연산량은 QKᵀ와 PV를 합쳐 대략 `4 * B * H * N² * D` FLOPs다. causal은 계산을 실제로 삼각 영역으로 줄인 최적화 kernel에 한해 약 절반으로 계산한다. 첫 causal 구현처럼 모든 K block을 계산하고 mask만 한다면 이 TFLOPS 숫자는 유효 연산 기준이므로 hardware utilization과 같지 않다.

성능을 개선할 때는 한 번에 하나씩 비교한다.

1. causal future block skip
2. `BLOCK_M/BLOCK_N`
3. `num_warps/num_stages`
4. autotune
5. TensorDescriptor/TMA

## 계산 범위와 작업 배치: causal 구간 분리와 persistent

`_raw_inner`의 STAGE에 따라 K/V loop 범위를 바꾼다. query tile 시작이 `start_m`일 때:

| STAGE | K/V 범위 | score mask |
|---|---|---|
| 1 | `[0, start_m)` | key tail만 |
| 2 | `[start_m, min(start_m + BLOCK_M, N))` | key tail + causal |
| 3 | `[0, N)` | key tail만 (non-causal) |

causal이면 1→2 순서로 같은 `m_i/l_i/acc`를 넘긴다. 구간 사이에서 상태를 초기화하면
과거 구간의 결과가 사라진다. `BLOCK_M % BLOCK_N == 0`을 유지해야 경계에서 K/V
tile이 겹치지 않는다. 기본형과 달리 완전히 미래인 block은 계산하지 않는다.

`_raw_fwd`는 `(head, query tile)`을 linear tile ID로 펼친다. `split`은 tile 수만큼
program을 띄우고, `persistent`는 `min(total_tiles, SM 수 * programs_per_sm)`개만
띄운 뒤 program 수를 보폭으로 순회한다. `m_i/l_i/acc`는 **매 query tile마다**
초기화한다. 먼저 같은 raw 본문으로 grid만 바꿔 검증하고 시간을 비교한다.

## 데이터 이동과 overlap: TMA와 warp specialization

`_launch_tma`에 descriptor 생성과 launch가 준비되어 있다. Q/K/V/O를
`[B*H*N, D]`로 해석하고 query 좌표에는 `head * N + start_m`, key/value 좌표에는
`head * N + start_n`을 사용한다. block shape은 Q/O `[BLOCK_M,D]`, K/V `[BLOCK_N,D]`다.

이 2D flatten 방식에서는 descriptor가 head 경계를 모른다. N tail에서 load하면
다음 head의 데이터까지 읽을 수 있으므로 첫 실습은 N을 128의 배수로 제한한다.
N tail을 확장하려면 per-head descriptor 또는 명시적인 score/store 경계 처리를
설계해야 한다. raw 버전의 tail 지원이 자동으로 옮겨지는 것이 아니다.

TMA TODO에서는 load/store를 교체하고 같은 online softmax 수식을 넣는다.
WS TODO에서는 같은 TMA kernel의 K/V loop에 `warp_specialize=True`를 요청한다.
자동 WS는 compiler와 loop/layout의 지원 범위에 영향을 받으므로, GPU 조건을
만족해도 모든 설정이 컴파일되거나 빨라진다는 보장은 없다.
현재 실습은 고정 `BLOCK_M=BLOCK_N=64`, `num_warps=4`, `num_stages=2`에서 시작한다.
[`tl.range` 공식 API](https://triton-lang.org/main/python-api/generated/triton.language.range.html)

## 설정 탐색: autotune

`RAW_CONFIGS`와 `_autotuned_raw_fwd`가 준비되어 있다. raw split 계산을 완성한 뒤
각 후보를 고정 설정으로 먼저 검증한다. `_launch_raw`의 고정 config를 후보에 맞춰
바꾸고 `--impl split`로 검사한 후 원래 설정으로 되돌리면 된다. 그다음 autotune
wrapper의 raise를 삭제하고 `--impl autotune`을 실행한다. 잘못된 후보가 빠른
측정값 때문에 선택되는 일을 피하기 위한 순서다.

이 단계는 raw kernel을 튜닝한다. TMA autotune으로 확장하려면 후보마다 descriptor의
`block_shape`도 갱신하는 config pre-hook이 추가로 필요하다. block meta만 바꾸고
descriptor shape를 고정해 두면 load/store tile shape가 어긋난다.

## 학습을 위한 gradient 계산: backward

먼저 raw forward의 `SAVE_LSE` 분기를 완성한다. 저장할 값은 FP32 `[B,H,N]`의
`LSE = m_i + log2(l_i)`다. 이후 full probability 행렬을 저장하는 대신 tile마다
다음 식으로 재계산한다. 여기서 `S = QKᵀ * sm_scale`이다.

```text
P     = exp2(S * log2(e) - LSE)
Delta = rowsum(O * dO)
dP    = dO @ Vᵀ
dS    = P * (dP - Delta)
dQ    = (dS @ K) * sm_scale
dK    = (dSᵀ @ Q) * sm_scale
dV    = Pᵀ @ dO
```

`dS`는 자연 지수 softmax의 scaled score에 대한 미분이다. 따라서 gradient 식에
`log2(e)`를 다시 곱하지 않는다. causal 미래 위치와 tail의 P는 0으로 둔다.

`_bwd_preprocess`는 Delta를 만들고, `_bwd_dq`는 query tile을 소유하며 K/V를
순회한다. `_bwd_dkdv`는 key tile을 소유하며 query를 순회한다. 후자는 dK/dV를
atomic 없이 한 번 저장할 수 있지만 P/dS를 재계산하는 비용이 든다.

`_AttentionAutograd`에는 저장할 tensor와 3개 커널 launch가 연결되어 있다.
기본 `attention()`은 inference 실습이고, gradient를 붙일 때는 `attention_train()`을
사용한다. backward는 forward와 별도의 launch이며 double backward는 이 실습 범위에 없다.

## 후속 과제

BF16, FP8, cross attention, GQA/MQA, dropout, variable length는 아직 뼈대에 포함하지
않았다. FP8은 dtype 변경뿐 아니라 Q/K/V의 quantization scale, P를 dot에 넣을 때의
정밀도, V layout, reference와 허용 오차를 함께 설계하는 별도 실습으로 진행한다.

## 참고 자료

- [Triton 공식 Fused Attention 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Triton Matrix Multiplication 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
