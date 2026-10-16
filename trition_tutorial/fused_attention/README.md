# Triton FlashAttention forward 구현 가이드

FA1/FA2 forward의 tiling과 online softmax를 Triton 코드로 옮기는 실습이다.
FA1의 K/V 중심 루프와 FA2의 query tile 중심 루프를 각각 구현하고,
둘 다 다음 계산을 `하나의 Triton forward kernel`에 담는다.

```text
QK^T -> scale / causal mask -> online softmax -> PV -> O
```

전체 `N x N` score와 probability를 HBM에 저장하지 않는 것이 핵심이다. 첫 구현은 raw pointer와 `tl.load`/`tl.store`만 사용한다. 확장 단계도 커널 구조와 launch는 제공하지만 핵심 계산은 TODO로 남겨두었다. FP8은 후속 과제로 남긴다.

## 개념과 실습 코드

실습 개수나 순서를 채우는 것이 목표는 아니다. 공통 기반은 tile 단위 QKᵀ/PV,
online softmax, mask와 수치 안정성이다. 그 기반을 이해한 뒤 관심 있는 주제를
선택한다.

- Attention 계산: QKᵀ/PV, online softmax 상태 갱신, 마지막 한 번의 정규화.
- 데이터 이동: raw pointer와 TMA descriptor, load/계산 overlap과 warp specialization.
- 작업 배치와 튜닝: causal 계산 생략, persistent scheduling, tile/warp/stage 선택.

현재 범위는 forward다. 먼저 기본 커널에서 상태가 어떻게 유지되는지 확인하고,
TMA와 WS 등은 관심에 따라 선택한다. 기본 커널에도 query tile 단위 병렬화가
있으며, persistent는 그 tile들을 program에 배분하는 방식을 바꾸는 추가 실험이다.

| `--impl` | 학습할 내용 | 구현 위치 | 입력 범위 |
|---|---|---|---|
| `baseline` | stride, tail mask, online softmax | `attention_fa2.py` | FP16, 임의 stride, N tail, D=16~128 |
| `fa1` | K/V 바깥 루프, 정규화된 O와 통계의 반복 저장 | `attention_fa1.py` | contiguous FP16, N/D tail |
| `fa2` | Q tile별 병렬화, 정규화 전 출력 누적 | `attention_fa2.py` (`baseline`과 동일) | baseline과 동일 |
| `split` | causal 과거/대각 구간 분리 | `attention_variants.py`: SPLIT | contiguous, N/D tail |
| `persistent` | 제한된 program이 query tile 여러 개 처리 | 같은 파일: PERSISTENT | split과 동일 |
| `tma` | host TensorDescriptor load/store | 같은 파일: TMA | Hopper 이상, contiguous, N=128의 배수, D=64/128 |
| `tma-ws` | K/V loop의 자동 warp specialization | 같은 파일: WS | Blackwell, TMA와 동일 |
| `autotune` | BLOCK_M/N, warp/stage 탐색 | 같은 파일: RAW_CONFIGS | split과 동일 |

모든 변형은 같은 self-attention 수식을 계산한다. `split/persistent/autotune`은
raw forward 본문을 공유하고, `tma/tma-ws`는 descriptor 본문을 공유한다.
TMA·persistent·WS는 서로 다른 선택이다. 데이터 이동 방식, program의 작업 배치,
warp의 역할 분할을 각각 바꾼다. persistent가 항상 빠르다는 가정은 하지 않는다.

## FA1 vs FA2: 둘 다 최적화하며 비교하기

목표는 FA1을 일부러 느리게 작성해서 FA2가 이기는 결과를 만드는 것이 아니라,
두 알고리즘을 이해하고 각 구조에서 얻을 수 있는 성능을 찾아보는 것이다.

| 비교할 부분 | FA1 방식 | FA2 방식 |
|---|---|---|
| program 소유 범위 | head 하나 | query tile 하나 |
| 내부 루프 | K/V 바깥, Q 안쪽 | K/V 순회 |
| 출력 상태 | 매 block 정규화, global buffer에 저장 | FP32 acc 유지, 마지막 한 번 정규화 |
| program 수 | B × H | B × H × ceil(N / BLOCK_M) |
| 튜닝 변수 | BLOCK_M/N, num_warps, num_stages | 동일한 후보 범위에서 독립 탐색 |

FA1도 전체 forward를 한 번의 kernel launch로 처리한다. K/V block마다 Python에서
kernel을 호출하면 launch overhead와 알고리즘 차이를 함께 재게 되므로 이 실습에서는
그렇게 구성하지 않는다. 양쪽 모두 causal의 완전히 미래인 영역은 건너뛰며, 대각 tile에는
mask를 적용한다. FA1은 현재 key block에서 기여가 전혀 없는 query row의 NaN 처리도 필요하다.

FA1의 정규화된 중간 O는 반복 FP16 반올림을 피하기 위해 **FP32 scratch**에 저장한다.
마지막 FP16 변환도 같은 kernel 안에서 처리하며 그 비용은 측정에 포함된다. 따라서
이 뼈대의 실제 메모리 이동량에는 FP32 상태 저장과 마지막 변환 비용이 들어간다.

`prepare_fa1()`과 `prepare_fa2()`는 buffer를 미리 할당한 뒤 재실행할 callable을
반환한다. 각 파일의 TODO를 채우고 prepare 함수의 raise를 삭제하면 아래를 실행할 수 있다.
FA2의 launch는 연결되어 있으므로 TODO 8은 argument 대응을 확인하는 실습이다.

```bash
# 먼저 같은 reference로 결과 확인
.venv/bin/python trition_tutorial/fused_attention/check.py --impl fa1 fa2

# 같은 tile/warp/stage로 구조 비교
.venv/bin/python trition_tutorial/fused_attention/compare_fa.py --preset fixed

# 작은 B×H: sequence 방향 병렬화의 영향
.venv/bin/python trition_tutorial/fused_attention/compare_fa.py \
  --preset quick --batch 1 --heads 2 --seq-lens 512 2048 4096

# 큰 B×H: FA1도 충분한 head를 갖는 조건
.venv/bin/python trition_tutorial/fused_attention/compare_fa.py \
  --preset full --batch 4 --heads 32 --seq-lens 2048 --head-dim 64

# Modal에서도 동일한 비교 가능
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/fused_attention/compare_fa.py \
  --script-args '--preset quick --batch 1 --heads 2 --seq-lens 2048'
```

`fixed`는 `(BM,BN,warps,stages)=(64,64,4,2)` 하나, `quick`은 6개,
`full`은 `BM/BN={32,64,128}`, `warps={4,8}`, `stages={1,2,3,4}`의 72개 후보다.
각 shape·causal 조건·구현별로 최고 설정을 따로 고른다. 모든 후보는 작은 sequence,
이중 tail과 실제 측정 shape에서 검증하고, 반복 호출로 FA1 상태 초기화도 확인한다.
정확도 실패·컴파일 실패·자원 초과 후보는 기록하고 제외한다. 예상하지 못한 runtime
오류는 중단한다. `full` 탐색은 여러 shape를 한꺼번에 넣기보다 관심 shape부터 실행한다.

측정은 사전 할당된 buffer를 CUDA Graph에서 반복 사용하는 GPU 시간이다.
컴파일·할당 비용은 제외하고, FA1의 커널 내부 상태 초기화·갱신 비용은 포함한다.
L2 flush 없이 같은 buffer를 재사용하므로 이를 HBM 대역폭 실측으로 해석하지 않는다.
기존 `benchmark.py`의 wrapper 포함 시간과도 구분한다.

검색 후에는 고정 설정과 각 구현의 최고 설정을 각각 FA1/FA2 순서를 번갈아 재측정한다.
기본 3회 측정의 중앙값과 원본 값을 저장한다. `FA1 시간 / FA2 시간`이 1보다 크면
FA2가 그 비율만큼 빠르다. 양쪽 결과가 모두 있을 때만 배율을 표시한다.
이는 **탐색한 후보 중 최고 성능**이며 모든 가능한 최적화를 끝냈다는 뜻은 아니다.

결과는 `tuning_results/fa_compare_<UTC시각>.csv/.json`에 저장한다.
Modal은 `OUT_DIR`를 통해 결과를 회수한다. CSV에는 실패 이유와 register/spill/shared
memory 정보가 포함되고, JSON에는 GPU·Torch·Triton 버전, 소스 hash, 고정/최고 설정
비교가 들어간다. 미구현이면 TODO만 출력하고 성능 파일은 생성하지 않는다.

그다음에는 causal 구간 분리, TMA, WS를 붙이며 각각의 효과를 분리해서 측정한다.
FA1의 루프를 Q 중심으로 바꾼 뒤 여전히 FA1이라고 이름 붙이면 구조 비교가 흐려지므로,
알고리즘의 구분을 유지하는 튜닝과 새 구조의 실험을 구별한다.
공식 CUDA 구현의 split-K/split-Q warp 배치를 이 Triton 뼈대가 그대로 재현하지는 않는다.
결과는 이 저장소의 FA1/FA2 **방식** 비교로 기록한다.
알고리즘 기준: [FA1 논문](https://arxiv.org/abs/2205.14135),
[FA2 논문](https://tridao.me/publications/flash2/flash2.pdf).

## FA2 forward 수식을 코드에 연결하기

`attention_fa2.py`가 출발점이다. 글에서 바깥 Q loop로 표현한 작업을 이 커널은
`tl.program_id(0)`으로 나눠 실행한다. 각 program은 query tile 하나를 맡고,
내부의 `start_n` loop가 K/V tile들을 순회한다.

| 글의 기호/개념 | 코드 | 확인할 점 |
|---|---|---|
| Q 행 블록 `i` | `query_block`, `offs_m` | program마다 다른 query row |
| K/V 열 블록 `j` | `start_n`, `offs_n` | 같은 Q tile로 모든 유효 K/V 처리 |
| `B_r`, `B_c`, `d` | `BLOCK_M`, `BLOCK_N`, `HEAD_DIM` | `BLOCK_D`는 dot을 위한 D padding |
| 행별 최댓값 `m_i` | `m_i` | 새 최대에 맞춰 누적 상태 rescale |
| 행별 분모 `ℓ_i` | `l_i` | FP32 누적 |
| 정규화 전 출력 `Õ_i` | `acc` | K/V loop 안에서 나누지 않음 |
| 최종 출력 `O_i` | `acc / l_i[:, None]` | loop 뒤에 한 번 정규화·store |

글의 자연 지수 `exp` 대신 이 뼈대는 `exp2`를 쓴다. 따라서 score에
`sm_scale * log2(e)`를 곱한다. 이때 `m_i`는 변환된 score의 최댓값이며,
`alpha = exp2(m_old - m_new)`로 이전 `l_i`와 `acc`를 함께 보정한다.
`p = exp2(qk - m_new)`는 아직 행 합이 1인 probability가 아니다.
이 `p`로 바로 PV를 누적하고 마지막에만 `l_i`로 나누는 부분이 핵심이다.

글에서 SRAM으로 묶어 설명한 상태는 실제 코드에서 register/shared memory와
compiler layout으로 구체화된다. `acc`는 Triton의 논리적인 FP32 tile이고,
성능 실험에서는 register 사용량과 spill도 확인한다.

현재 forward 출력은 O뿐이므로 LSE 저장은 필요 없다. 우선 TODO를 채워 causal과
non-causal, tail case를 통과시키면 글의 수식이 실제 GPU 계산으로 연결된다.

## 어디서 시작할까

Fused Attention에 필요한 직접적인 선수 지식은 tiled matmul의 pointer/K loop와 fused softmax다. Persistent Matmul, Group GEMM, Block-scaled Matmul은 필수 선수 과목이 아니다.

- 기본 matmul의 `tl.dot`, pointer, tail mask가 익숙하면 이 가이드를 계속 진행한다.
- 그 부분이 아직 낯설면 [Persistent Matmul 가이드의 기본 GEMM 단계](../persistent_matmul/)까지만 먼저 구현하고 돌아온다.
- Group GEMM과 Block-scaled Matmul은 각각 scheduling과 저정밀 하드웨어를 다루는 별도 트랙이다.

## 파일 구성

```text
fused_attention/
├── attention_fa1.py       # K/V 중심 forward 뼈대, prepare_fa1
├── attention_fa2.py     # Q tile 중심 forward 뼈대, prepare_fa2
├── attention_variants.py  # forward 최적화 실습과 launch
├── implementations.py     # --impl 이름, 하드웨어 조건
├── check.py               # FP32 reference와 forward 검사
├── benchmark.py           # 검증한 forward와 Torch SDPA 비교
├── compare_fa.py          # FA1/FA2 사전 할당 비교, 개별 튜닝, CSV/JSON
└── README.md              # 구현 가이드
```

`attention_fa2.py`의 공개 함수는 다음 시그니처를 유지한다.

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
.venv/bin/python trition_tutorial/fused_attention/check.py --impl baseline
.venv/bin/python trition_tutorial/fused_attention/check.py --impl split persistent
.venv/bin/python trition_tutorial/fused_attention/check.py --impl tma tma-ws
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
case를 사용한다. reference는 Q/K/V를 FP32로 올려 계산한다.

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

forward만 계산하므로 마지막에는 O를 저장하면 된다.

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

`attention_fa2.py`에는 program mapping, online-softmax 상태, K/V loop, wrapper 설정까지 들어 있다. 긴 완성 kernel을 README와 코드에 중복하지 않고 다음 TODO 경계만 제공한다.

| TODO | 구현 내용 |
|---:|---|
| 1 | Q tile을 loop 밖에서 한 번 load |
| 2 | 현재 K tile과 tail mask |
| 3 | `QKᵀ * sm_scale * log2(e)` |
| 4 | key tail과 optional causal score mask |
| 5 | `m_i/l_i/acc` online-softmax 갱신 |
| 6 | V tile load와 `P @ V` 누적 |
| 7 | normalize 후 O store |
| 8 | 준비된 Python launch의 stride/meta-parameter 대응 확인 |

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

## 후속 과제

BF16, FP8, cross attention, GQA/MQA, dropout, variable length는 아직 뼈대에 포함하지
않았다. FP8은 dtype 변경뿐 아니라 Q/K/V의 quantization scale, P를 dot에 넣을 때의
정밀도, V layout, reference와 허용 오차를 함께 설계하는 별도 실습으로 진행한다.

## 참고 자료

- [Triton 공식 Fused Attention 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Triton Matrix Multiplication 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
