# Triton Fused Attention 구현 가이드

이 문서는 Triton 공식 [Fused Attention 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)을 직접 구현하기 위한 학습 순서다. 공식 코드는 FlashAttention-2를 forward, backward, causal mask, FP8, Tensor Descriptor, autotune까지 한 파일에 담고 있어서 처음부터 그대로 옮기면 알고리즘과 최적화 코드를 구분하기 어렵다.

여기서는 다음 순서로 구현한다.

1. PyTorch reference로 수식과 layout을 고정한다.
2. FP16 non-causal forward만 구현한다.
3. online softmax가 block을 합치는 과정을 검증한다.
4. causal mask를 off-band와 diagonal block으로 나눈다.
5. `torch.autograd.Function`과 backward를 붙인다.
6. 마지막에 Tensor Descriptor, autotune, warp specialization, FP8을 추가한다.

첫 목표는 공식 예제의 모든 기능을 복사하는 것이 아니라, `QK^T`와 attention probability를 HBM에 저장하지 않는 정확한 forward kernel을 만드는 것이다.

## 1. 구현 범위와 표기

입력 layout은 공식 예제와 같은 `[B, H, N, D]`로 둔다.

| 기호 | 의미 |
|---|---|
| `B` | batch 크기 |
| `H` | attention head 수 |
| `N` | sequence length (`N_CTX`) |
| `D` | head dimension (`HEAD_DIM`) |
| `Q, K, V` | `[B, H, N, D]` |
| `S` | scaled score, `QK^T * sm_scale` |
| `P` | `softmax(S)` |
| `O` | attention output, `PV` |

수식은 다음과 같다.

```text
S = Q @ K^T * sm_scale
P = softmax(S, dim=-1)
O = P @ V
```

causal attention에서는 query 위치 `i`가 미래의 key `j > i`를 보지 못하게 한다.

```text
S[i, j] = -inf  if j > i
```

학습 단계에서는 아래 제약부터 시작하는 것이 좋다.

- CUDA GPU
- `dtype=torch.float16`
- `D in {64, 128}`
- `N`은 `BLOCK_M`, `BLOCK_N`의 배수
- Q/K/V는 contiguous이고 stride가 동일
- dropout, bias, GQA/MQA, variable length는 제외

## 2. 왜 fused attention이 필요한가

naive attention은 `S`와 `P`라는 `[B, H, N, N]` 중간 텐서를 만든다. 연산량뿐 아니라 이 텐서를 HBM에 쓰고 다시 읽는 비용이 크다.

예를 들어 `B=4`, `H=32`, `N=4096`이면 attention matrix 하나의 크기는 다음과 같다.

```text
FP16: 4 * 32 * 4096 * 4096 * 2 bytes = 4 GiB
FP32: 4 * 32 * 4096 * 4096 * 4 bytes = 8 GiB
```

FlashAttention은 `Q`의 일부 행과 `K/V`의 일부 열만 tile로 가져온다. score tile을 계산한 즉시 softmax 통계와 output accumulator에 반영한 뒤 버린다. 따라서 전체 `N x N` score/probability 행렬을 HBM에 materialize하지 않는다.

핵심 차이는 다음과 같다.

```text
naive:
Q, K -> S를 HBM에 저장 -> P를 HBM에 저장 -> V와 matmul

fused:
Q tile을 유지 -> K/V tile을 순회 -> online softmax와 PV를 즉시 누적 -> O만 저장
```

이 알고리즘은 근사 attention이 아니다. 연산 순서와 부동소수점 반올림은 달라질 수 있지만 수학적으로 같은 softmax attention을 계산한다.

## 3. 먼저 PyTorch reference를 만든다

Triton kernel보다 reference와 test를 먼저 작성한다. 이 함수가 이후 모든 단계의 정답이다.

```python
def attention_reference(q, k, v, causal: bool, sm_scale: float):
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if causal:
        n = q.shape[-2]
        mask = torch.ones((n, n), device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)
```

초기 smoke test는 작게 잡는다.

```python
B, H, N, D = 1, 2, 128, 64
sm_scale = D**-0.5
```

검증 순서는 다음과 같다.

1. non-causal forward
2. causal forward
3. 서로 다른 `B`, `H`
4. `D=64`, `D=128`
5. `N=128`, `1024`, `4096`
6. `dQ`, `dK`, `dV`

FP16 forward/gradient의 시작 tolerance는 공식 예제와 같은 `atol=1e-2, rtol=0`으로 둘 수 있다. 실패하면 tolerance부터 늘리지 말고 어느 row와 어느 causal 경계에서 처음 달라지는지 확인한다.

## 4. Online softmax

한 query row의 score 전체를 한 번에 갖고 있지 않아도 softmax를 정확하게 계산할 수 있다. 지금까지 처리한 key block의 상태를 세 값으로 유지한다.

```text
m_i   : 지금까지 본 score의 row-wise maximum
l_i   : exp(score - m_i)의 row-wise sum
acc_i : exp(score - m_i) @ V의 누적값
```

새 score block `S_ij`를 만났을 때 다음처럼 갱신한다.

```text
m_new = max(m_old, rowmax(S_ij))
alpha = exp(m_old - m_new)
P_ij  = exp(S_ij - m_new)

l_new   = alpha * l_old + rowsum(P_ij)
acc_new = alpha * acc_old + P_ij @ V_j
```

모든 key block을 처리한 뒤:

```text
O_i = acc_i / l_i
```

`m`이 커질 때 기존 누적값의 기준도 바뀌므로 `l_old`와 `acc_old` 모두 `alpha`를 곱해야 한다. 둘 중 하나라도 빠뜨리면 작은 입력에서는 우연히 맞아 보이더라도 block maximum이 달라지는 순간 결과가 깨진다.

### 왜 `exp2`를 사용하는가

공식 kernel은 `tl.math.exp2`를 사용한다. 자연지수 softmax를 밑 2 지수로 바꾸려면 score에 `log2(e) = 1 / ln(2)`를 곱한다.

```text
exp(x) = 2^(x * log2(e))
qk_scale = sm_scale * 1.44269504
```

따라서 kernel 안에서 관리하는 `m_i`도 밑 2 영역의 값이다. forward 마지막에 backward용으로 저장하는 값은:

```text
M_i = m_i + log2(l_i)
```

이며, 이는 각 row의 log-sum-exp를 밑 2로 표현한 값이다.

## 5. Forward kernel의 tile 구조

각 Triton program은 한 `(batch, head)`의 `BLOCK_M`개 query row를 담당한다.

```text
grid axis 0: ceil_div(N, BLOCK_M)
grid axis 1: B * H
```

program 내부 shape은 다음과 같다.

| 값 | tile shape | 역할 |
|---|---|---|
| `q` | `[BLOCK_M, D]` | program 동안 유지 |
| `k` | `[BLOCK_N, D]` | K loop마다 load |
| `qk` | `[BLOCK_M, BLOCK_N]` | 현재 score tile |
| `v` | `[BLOCK_N, D]` | K loop마다 load |
| `acc` | `[BLOCK_M, D]`, FP32 | output numerator |
| `m_i`, `l_i` | `[BLOCK_M]`, FP32 | online softmax 상태 |

program id는 다음처럼 해석한다.

```python
start_m = tl.program_id(0)
off_hz = tl.program_id(1)
off_z = off_hz // H
off_h = off_hz % H
```

flatten된 `[B, H, N, D]`에서 해당 head의 token 시작 위치는:

```text
offset_y = (off_z * H + off_h) * N
```

### Forward 구현 뼈대

아래는 복사해서 완성하는 코드가 아니라 구현 순서를 보여주는 의사 코드다.

```python
start_m = tl.program_id(0)
off_hz = tl.program_id(1)

q = load Q[start_m * BLOCK_M : ..., :]
m_i = full([BLOCK_M], -inf, fp32)
l_i = zeros([BLOCK_M], fp32)
acc = zeros([BLOCK_M, D], fp32)

for start_n in range(0, N, BLOCK_N):
    k = load K[start_n : start_n + BLOCK_N, :]
    qk = dot(q, trans(k)) * qk_scale

    m_new = maximum(m_i, max(qk, axis=1))
    p = exp2(qk - m_new[:, None])
    alpha = exp2(m_i - m_new)

    v = load V[start_n : start_n + BLOCK_N, :]
    acc = acc * alpha[:, None] + dot(p.to(fp16), v)
    l_i = l_i * alpha + sum(p, axis=1)
    m_i = m_new

store O, acc / l_i[:, None]
store M, m_i + log2(l_i)
```

구현할 때 확인할 점:

- `q`는 K/V loop 밖에서 한 번만 load한다.
- `acc`, `m_i`, `l_i`는 FP32로 누적한다.
- `p`는 `tl.dot(p, v)` 직전에 FP16으로 내린다.
- `l_i`, `m_i` 갱신은 register pressure를 낮추기 위해 loop 뒤쪽에 둔다.
- 첫 구현에서는 `BLOCK_M=64`, `BLOCK_N=64`, `num_warps=4`로 고정한다.
- autotune은 correctness가 완성된 뒤에 붙인다.

## 6. Causal attention은 두 영역으로 나눈다

causal mask를 모든 score tile에 적용할 필요는 없다. query tile `[start_m, start_m + BLOCK_M)`를 기준으로 key 영역을 나누면 된다.

```text
off-band: [0, start_m * BLOCK_M)
           전부 과거 token이므로 mask 불필요

on-band:  [start_m * BLOCK_M, (start_m + 1) * BLOCK_M)
           diagonal과 겹치므로 triangular mask 필요

right:     그 이후
           전부 미래 token이므로 계산하지 않음
```

on-band에서만 다음 mask를 적용한다.

```python
mask = query_offsets[:, None] >= key_offsets[None, :]
qk = tl.where(mask, qk, -1.0e6)
```

공식 코드의 `STAGE`는 이 실행 범위를 bit mask로 표현한다.

| wrapper의 `STAGE` | 의미 |
|---:|---|
| `1` | non-causal: 전체 K/V 범위 |
| `3` | causal: off-band와 on-band 모두 실행 |

공식 `_attn_fwd`에서 첫 inner call에 `4 - STAGE`를 넘기는 부분은 읽기 어렵다. 결과만 풀면 non-causal일 때 inner stage 3이 전체 `[0, N)`을 돌고, causal일 때 inner stage 1이 mask 없는 왼쪽 영역을 돈다. 이어서 stage 2가 diagonal 영역만 mask해서 처리한다.

첫 구현에서는 숫자 trick을 바로 따라 하기보다 `CAUSAL: tl.constexpr` 분기와 의미가 드러나는 helper 두 개로 작성한 뒤, correctness를 확인하고 공식 구조로 합치는 편이 이해하기 쉽다.

## 7. Tensor Descriptor와 TMA

공식 최신 예제는 Q/K/V/O 접근에 Tensor Descriptor를 사용한다. kernel 내부에서는 대략 다음 형태다.

```python
desc_q = tl.make_tensor_descriptor(
    q_ptr,
    shape=[B * H * N, D],
    strides=[D, 1],
    block_shape=[BLOCK_M, D],
)
q = desc_q.load([query_row_offset, 0])
```

NVIDIA TMA를 지원하는 GPU에서는 descriptor load/store가 TMA hardware를 사용할 수 있다. Descriptor에는 다음 제약이 있다.

- base pointer는 16-byte aligned여야 한다.
- 마지막 dimension은 contiguous여야 한다.
- leading stride는 byte 기준 16의 배수여야 한다.
- 현재 공식 문서 기준 2~5차원 tensor를 지원한다.
- TMA descriptor용 global allocation callback을 `triton.set_allocator`로 등록해야 한다.

그러나 descriptor는 알고리즘의 본질이 아니다. 학습할 때는 다음 두 단계로 나누는 것을 권장한다.

1. 일반 pointer 또는 block pointer로 online-softmax forward를 완성한다.
2. load/store 부분만 Tensor Descriptor로 교체한다.

이렇게 해야 잘못된 결과가 주소 계산 때문인지 online softmax 때문인지 분리할 수 있다.

공식 `main` 튜토리얼은 `triton.tools.tensor_descriptor.TensorDescriptor`와 `tl.make_tensor_descriptor`를 함께 사용한다. 이 API는 Triton 버전에 민감하므로 구현을 시작하기 전에 Modal image 안에서 확인한다.

```python
import torch, triton
from triton.tools.tensor_descriptor import TensorDescriptor

print(torch.__version__)
print(triton.__version__)
print(torch.cuda.get_device_capability())
```

이 저장소의 Modal image는 현재 Torch와 Triton 버전을 고정하고 있으므로, 공식 `main` 코드가 그대로 import되지 않으면 알고리즘을 바꾸지 말고 descriptor API만 설치된 버전에 맞춘다.

## 8. Python wrapper와 autograd 연결

forward wrapper가 책임질 일은 다음과 같다.

1. Q/K/V shape과 stride 검사
2. output `O` 할당
3. backward에서 사용할 row-wise log-sum-exp `M` 할당
4. `causal`을 compile-time stage로 변환
5. grid 계산
6. kernel launch
7. `q, k, v, o, M`을 `ctx`에 저장

```python
class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        ...

    @staticmethod
    def backward(ctx, do):
        ...


attention = Attention.apply
```

forward grid는 meta parameter인 `BLOCK_M`에 의존하므로 callable로 둔다.

```python
def grid(meta):
    return (triton.cdiv(N, meta["BLOCK_M"]), B * H, 1)
```

## 9. Backward 수식

먼저 수식을 PyTorch tensor 연산으로 유도하고 kernel mapping을 본다. upstream gradient를 `dO`라고 하면:

```text
dV = P^T @ dO
dP = dO @ V^T
D_i = sum_j(P_ij * dP_ij)
dS = P * (dP - D[:, None])
dQ = sm_scale * dS @ K
dK = sm_scale * dS^T @ Q
```

`D_i`는 다음 항등식으로 attention matrix 없이 계산할 수 있다.

```text
D_i = sum_d(O_id * dO_id)
```

그래서 공식 backward는 먼저 작은 preprocess kernel을 실행한다.

```python
delta = tl.sum(o * do, axis=1)
```

그 뒤 main backward kernel은 두 방향의 scan을 수행한다.

- K/V tile을 유지하고 Q/dO tile을 순회하여 `dK`, `dV` 누적
- Q/dO tile을 유지하고 K/V tile을 순회하여 `dQ` 누적

forward에서 저장한 `M`을 이용하면 `P`를 HBM에 저장하지 않고 score tile마다 다시 만들 수 있다.

```text
P_ij = exp2(Q_i K_j^T * sm_scale * log2(e) - M_i)
```

즉 backward는 메모리를 아끼는 대신 `QK^T`를 재계산한다. 공식 benchmark가 backward FLOPs에 recomputation 비용을 포함하는 이유다.

### Backward 구현 순서

한 번에 전체 backward를 작성하지 말고 다음 순서로 진행한다.

1. `_attn_bwd_preprocess`: `delta = rowsum(O * dO)`
2. non-causal `dV`
3. non-causal `dK`
4. non-causal `dQ`
5. 세 gradient를 reference와 비교
6. causal diagonal block mask 추가
7. causal의 mask 없는 영역 scan 추가

gradient마다 단독 test를 두면 transpose나 scale 오류를 빨리 찾을 수 있다.

## 10. Autotune은 마지막에 붙인다

공식 예제는 다음 공간을 탐색한다.

```text
BLOCK_M in {64, 128}
BLOCK_N in {32, 64, 128}
num_stages in {2, 3, 4}
num_warps in {4, 8}
```

autotune key에는 code generation이나 최적 config를 바꿀 입력을 넣는다.

```text
N_CTX, HEAD_DIM, FP8_OUTPUT, warp_specialize
```

잘못된 config는 benchmark 전에 제거한다.

- `BLOCK_M > N_CTX` 제거
- causal에서 `BLOCK_M < BLOCK_N` 제거
- architecture별 비효율적 조합 제거

`@triton.autotune`은 후보마다 kernel을 여러 번 실행한다. output을 누적 수정하는 kernel이라면 `reset_to_zero`나 pre-hook이 필요하다. forward처럼 매번 output을 완전히 덮어쓰더라도 test와 benchmark에서 autotune 비용을 실제 kernel 시간과 분리해야 한다.

개발 중에는 config 하나만 사용하고, correctness matrix가 모두 통과한 뒤 autotune을 켠다.

## 11. Warp specialization과 FP8

이 둘은 기본 forward/backward가 완성된 뒤의 확장 과제다.

### Warp specialization

공식 예제는 Hopper/Blackwell 여부, causal 여부, head dimension에 따라 warp specialization을 제한적으로 활성화하고 Blackwell에서는 `maxnreg`도 조절한다. 이는 단순 boolean 최적화가 아니라 architecture와 tile shape에 따른 register/producer-consumer scheduling 문제다.

확인할 항목:

- `kernel.n_regs`, `kernel.n_spills`
- shared memory 사용량
- 선택된 autotune config
- warp specialization 전후 kernel time
- H100과 B200의 결과 차이

### FP8

공식 예제의 FP8 forward는 V layout을 별도로 다루며 backward는 지원하지 않는다. 첫 구현에 FP8을 섞으면 dtype 오차와 layout 오류를 algorithm 오류와 구분하기 어렵다.

권장 순서:

1. FP16 forward/backward 완료
2. FP8 forward reference를 FP32로 계산
3. Q/K 변환 검증
4. V transpose/layout 검증
5. FP8 전용 tolerance 적용

## 12. Correctness test matrix

최소한 다음 조합을 자동 test로 만든다.

| 구분 | 값 |
|---|---|
| `B` | 1, 4 |
| `H` | 2, 32 |
| `N` | 128, 1024, 4096 |
| `D` | 64, 128 |
| causal | False, True |
| mode | forward, backward |

매 test에서 확인한다.

```python
torch.testing.assert_close(triton_out, ref_out, atol=1e-2, rtol=0)
torch.testing.assert_close(triton_dq, ref_dq, atol=1e-2, rtol=0)
torch.testing.assert_close(triton_dk, ref_dk, atol=1e-2, rtol=0)
torch.testing.assert_close(triton_dv, ref_dv, atol=1e-2, rtol=0)
```

추가로 확인할 invariant:

- causal output의 첫 token은 `V[..., 0, :]`와 가까워야 한다.
- 모든 output과 gradient가 finite여야 한다.
- 같은 seed에서 실행할 때 결과가 재현되어야 한다.
- non-contiguous 입력을 거부할지 내부에서 contiguous로 만들지 wrapper 계약이 명확해야 한다.
- `N`이 block 배수가 아닐 때 mask를 구현하지 않았다면 assertion으로 명시해야 한다.

## 13. Benchmark 읽는 법

forward의 주요 matmul은 `QK^T`와 `PV` 두 개다.

```text
one matmul FLOPs = 2 * B * H * N^2 * D
forward FLOPs    = 2 * one matmul FLOPs
```

causal은 score matrix의 절반만 계산하므로 공식 benchmark는 FLOPs를 `0.5`배 한다. backward는 forward 대비 gradient matmul과 score recomputation을 포함해 공식 예제에서 `2.5`배를 사용한다.

```python
tflops = total_flops * 1e-12 / (ms * 1e-3)
```

비교 대상:

- 직접 만든 Triton kernel
- `torch.nn.functional.scaled_dot_product_attention`
- 설치되어 있다면 FlashAttention-2

benchmark 전에 반드시 correctness를 통과해야 한다. causal/non-causal, `D=64/128`, forward/backward를 별도 그래프로 기록한다. peak TFLOPS 하나만 보지 말고 `N` 증가에 따른 추세와 tile 경계의 불연속도 함께 본다.

이 저장소의 Modal runner에서 구현 파일을 실행하는 명령은 다음 형태다.

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/fused_attention/check.py \
  --gpu B200
```

## 14. 자주 틀리는 지점

### `alpha`를 `l_i`에만 적용함

maximum 기준이 변했으므로 `acc`에도 반드시 같은 `alpha`를 적용해야 한다.

### scale을 두 번 적용함

`qk_scale = sm_scale * log2(e)`를 썼다면 inner loop에서 `sm_scale`을 다시 곱하지 않는다.

### causal diagonal 조건이 반대임

허용 조건은 `query_index >= key_index`다. 작은 `N=4` 입력으로 mask를 출력해 확인한다.

### `tl.dot` operand shape가 뒤집힘

forward shape를 주석으로 고정한다.

```text
q  [BM, D]
kT [D, BN]
p  [BM, BN]
v  [BN, D]
```

### `M`에 `m_i`만 저장함

backward에서 probability를 재구성하려면 `m_i + log2(l_i)`가 필요하다.

### descriptor 문제와 알고리즘 문제를 동시에 디버깅함

pointer 기반 kernel을 먼저 맞춘 다음 descriptor로 교체한다.

### 처음부터 autotune을 켬

compile 조합이 많아지고 실패한 config의 원인을 찾기 어려워진다. 고정 config 하나로 시작한다.

### backward scale이 맞지 않음

공식 구현은 `exp2`와 미리 scale한 K를 사용하기 때문에 `dQ` 저장 전에 `ln(2)` 보정이 등장한다. 자연지수 수식과 밑 2 구현을 섞지 말고 각 tensor가 어느 영역의 값인지 적는다.

## 15. 권장 파일 구조와 구현 체크리스트

```text
trition_tutorial/fused_attention/
├── README.md                 # 이 문서
├── fused_attention.py        # Triton kernel과 autograd wrapper
├── check.py                  # reference 및 작은 correctness test
└── TIL.md                    # 구현 후 실제 GPU 측정 결과와 해석
```

구현 순서:

- [ ] PyTorch reference와 forward test 작성
- [ ] 고정 tile non-causal forward
- [ ] online softmax 상태 `m`, `l`, `acc` 검증
- [ ] causal off-band/on-band 분할
- [ ] backward용 `M = m + log2(l)` 저장
- [ ] `torch.autograd.Function` wrapper
- [ ] backward preprocess `delta`
- [ ] non-causal `dQ`, `dK`, `dV`
- [ ] causal backward
- [ ] 전체 correctness matrix
- [ ] Tensor Descriptor/TMA 적용
- [ ] autotune config와 pruning
- [ ] B200/H100 resource 및 성능 측정
- [ ] warp specialization 실험
- [ ] FP8 forward 실험

## 참고 자료

- [Triton 공식 Fused Attention 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Triton `make_tensor_descriptor` API](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html)
- [Triton `autotune` API](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
- [FlashAttention 논문](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 논문](https://arxiv.org/abs/2307.08691)
