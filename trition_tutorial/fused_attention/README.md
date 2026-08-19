# 일반적인 Triton Fused Attention 구현 가이드

이 디렉터리의 첫 목표는 최신 하드웨어 전용 최적화를 따라 하는 것이 아니다. 먼저 다음 계산을 `하나의 Triton forward kernel`에 담는다.

```text
QK^T -> scale / causal mask -> online softmax -> PV -> O
```

전체 `N x N` score와 probability를 HBM에 저장하지 않는 것이 핵심이다. 첫 구현은 raw pointer와 `tl.load`/`tl.store`만 사용한다. TensorDescriptor, TMA, autotune, warp specialization, FP8, backward는 정확한 기본 구현을 완성한 뒤에 확장한다.

## 어디서 시작할까

Fused Attention에 필요한 직접적인 선수 지식은 tiled matmul의 pointer/K loop와 fused softmax다. Persistent Matmul, Group GEMM, Block-scaled Matmul은 필수 선수 과목이 아니다.

- 기본 matmul의 `tl.dot`, pointer, tail mask가 익숙하면 이 가이드를 계속 진행한다.
- 그 부분이 아직 낯설면 [Persistent Matmul 가이드의 기본 GEMM 단계](../persistent_matmul/)까지만 먼저 구현하고 돌아온다.
- Group GEMM과 Block-scaled Matmul은 각각 scheduling과 저정밀 하드웨어를 다루는 별도 트랙이다.

## 파일 구성

```text
fused_attention/
├── fused_attention.py  # Triton kernel과 attention() wrapper
├── check.py            # PyTorch reference와 정확도 검사
├── benchmark.py        # Torch SDPA와 runtime/TFLOPS 비교
└── README.md           # 구현 가이드
```

`fused_attention.py`의 공개 함수는 다음 시그니처를 유지한다.

```python
def attention(q, k, v, causal: bool, sm_scale: float):
    ...
```

`check.py`가 이 함수를 PyTorch reference와 비교한다.

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
- 마지막 차원이 contiguous
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
| `[1, 2, 128, 64]` | 기본 block과 causal/non-causal |
| `[2, 3, 129, 64]` | batch/head decode와 sequence tail |
| `[1, 2, 129, 80]` | `BLOCK_D=128` padding과 이중 tail |

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

program id는 다음처럼 해석한다.

```python
pid_m = tl.program_id(0)
pid_bh = tl.program_id(1)

batch = pid_bh // H
head = pid_bh % H
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

## raw pointer로 주소 계산하기

descriptor 대신 tensor의 stride를 kernel argument로 넘긴다. Q의 pointer tile은 다음 형태다.

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_d = tl.arange(0, BLOCK_D)

q_ptrs = (
    Q
    + batch * stride_qb
    + head * stride_qh
    + offs_m[:, None] * stride_qn
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
    K
    + batch * stride_kb
    + head * stride_kh
    + offs_n[:, None] * stride_kn
    + offs_d[None, :] * stride_kd
)
k = tl.load(
    k_ptrs,
    mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
    other=0.0,
)
```

Q/K/V의 stride가 같다고 가정하지 말고 각각 전달한다.

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

아래 뼈대를 `fused_attention.py`에 옮기고 TODO를 순서대로 채운다.

```python
@triton.jit
def _attention_fwd(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    H, N,
    sm_scale,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // H
    head = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # TODO 1: Q pointer와 mask를 만들고 Q tile load
    q = ...

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    for start_n in tl.range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # TODO 2: K tile load
        k = ...

        # TODO 3: score 계산
        qk = tl.dot(q, tl.trans(k)) * qk_scale

        # TODO 4: key tail과 optional causal mask
        score_valid = ...
        qk = tl.where(score_valid, qk, -float("inf"))

        # TODO 5: online softmax
        m_new = ...
        alpha = ...
        p = ...
        l_new = ...

        # TODO 6: V tile load 후 output numerator 누적
        v = ...
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)

        m_i = m_new
        l_i = l_new

    out = acc / l_i[:, None]

    # TODO 7: O pointer와 query/head-dim tail mask
    tl.store(..., out, mask=...)
```

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
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1

    batch, n_heads, n_ctx, head_dim = q.shape
    assert 16 <= head_dim <= 128

    out = torch.empty_like(q)
    block_m = 64
    block_n = 64
    block_d = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(n_ctx, block_m), batch * n_heads)

    _attention_fwd[grid](
        q, k, v, out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        n_heads,
        n_ctx,
        sm_scale,
        D=head_dim,
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

동일한 grid와 Q/O pointer 계산만 구현해 `out == q`를 확인한다. batch/head decode, stride, token tail, `D` padding을 먼저 검증한다.

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

`benchmark.py`는 잘못된 kernel을 빈 output만 반환하는 빠른 kernel로 측정하지 않도록 먼저 `check_attention()`을 실행한다. 그다음 causal/non-causal에서 다음을 측정한다.

- `torch.nn.functional.scaled_dot_product_attention`
- 직접 구현한 Triton `attention`

forward의 주요 연산량은 QKᵀ와 PV를 합쳐 대략 `4 * B * H * N² * D` FLOPs다. causal은 계산을 실제로 삼각 영역으로 줄인 최적화 kernel에 한해 약 절반으로 계산한다. 첫 causal 구현처럼 모든 K block을 계산하고 mask만 한다면 이 TFLOPS 숫자는 유효 연산 기준이므로 hardware utilization과 같지 않다.

성능을 개선할 때는 한 번에 하나씩 비교한다.

1. causal future block skip
2. `BLOCK_M/BLOCK_N`
3. `num_warps/num_stages`
4. autotune
5. TensorDescriptor/TMA

## 그다음 확장 순서

1. BF16과 더 다양한 head dimension
2. `N_Q != N_KV`인 cross attention
3. backward용 log-sum-exp 저장과 `dQ/dK/dV` kernel
4. `BLOCK_M`, `BLOCK_N`, `num_warps` autotune
5. causal off-band/diagonal 영역 분리
6. TensorDescriptor/TMA로 memory access 교체
7. warp specialization과 FP8

TensorDescriptor는 attention 알고리즘 자체가 아니라 tile을 읽고 쓰는 방법이다. raw-pointer 구현의 `tl.load`/`tl.store` 부분을 descriptor load/store로 바꾸는 최적화 단계로 이해하면 된다. backward도 forward와 같은 launch에 담는 것이 아니라 보통 별도의 preprocess와 backward kernel로 구성한다.

## 참고 자료

- [Triton 공식 Fused Attention 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Triton Matrix Multiplication 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
