# Triton Block-scaled Matrix Multiplication 구현 가이드

Block-scaled Matmul은 네 주제 중 가장 나중에 하는 것이 좋다. GEMM tiling 외에도 저정밀 표현, scale broadcasting, FP4 packing, scale layout, 세대별 Tensor Core 명령을 함께 이해해야 하기 때문이다.

공식 튜토리얼의 최종 kernel은 Blackwell의 `tl.dot_scaled`, TensorDescriptor, preshuffled scale을 사용한다. 이 자료는 계산 의미부터 확인할 수 있도록 두 층으로 나눈다.

1. 모든 CUDA GPU에서 이해할 수 있는 explicit-scale FP16 kernel
2. B200에서 FP8/FP4와 `tl.dot_scaled`를 사용하는 하드웨어 경로

## 파일과 학습용 API

```text
block_scaled_matmul/
├── block_scaled_matmul.py  # explicit-scale portable kernel 뼈대
├── check.py                # 명시적 dequantize reference
├── benchmark.py            # explicit reference와 fused runtime 비교
└── README.md
```

첫 API는 일부러 단순하게 정의한다.

```python
c = block_scaled_matmul(a, b, a_scale, b_scale, vec_size)
```

shape:

```text
A       [M, K]              FP16 payload
B       [K, N]              FP16 payload
A_scale [M, K / VEC_SIZE]   FP32
B_scale [K / VEC_SIZE, N]   FP32
C       [M, N]              FP32 또는 FP16
```

K는 첫 단계에서 `VEC_SIZE`의 배수로 제한한다. M/N은 tile 배수가 아니어도 된다.
payload, scale, output의 각 축 stride는 kernel argument로 전달한다. 따라서 pointer 식은 `shape`로 row stride를 추측하지 않고 실제 tensor layout을 그대로 따른다.

## block scaling의 의미

scale 하나가 K축의 연속된 `VEC_SIZE` 원소에 적용된다.

```text
sa(m, k) = A_scale[m, floor(k / VEC_SIZE)]
sb(k, n) = B_scale[floor(k / VEC_SIZE), n]

C[m, n] =
    sum_k (A[m, k] * sa(m, k))
        * (B[k, n] * sb(k, n))
```

`check.py` reference는 `repeat_interleave`로 scale을 원래 K 크기로 펼친 뒤 FP32 `torch.matmul`을 실행한다. 먼저 이 의미론과 Triton 결과를 맞추는 것이 목표다.

실제 MXFP8/MXFP4에서는 payload가 이미 scale을 전제로 양자화된 값이고 scale 자체도 정해진 저정밀 형식을 사용한다. 학습용 FP16/FP32 표현은 hardware format이 아니라 같은 broadcasting 수식을 눈에 보이게 만든 출발점이다.

## 1단계: explicit-scale tiled GEMM

`_block_scaled_matmul_kernel`은 일반 GEMM과 같은 output tiling을 사용한다.

```python
pid = tl.program_id(0)
num_m_tiles = tl.cdiv(M, BLOCK_M)
tile_m = pid % num_m_tiles
tile_n = pid // num_m_tiles
```

K loop마다 다음 tile을 만든다.

```text
a       [BLOCK_M, BLOCK_K]
b       [BLOCK_K, BLOCK_N]
scale_a [BLOCK_M, BLOCK_K]  # scale index를 K 방향 broadcast
scale_b [BLOCK_K, BLOCK_N]

acc += dot(a * scale_a, b * scale_b)
```

각 K 원소의 scale index는:

```python
scale_k = offs_k // VEC_SIZE
```

A scale pointer는 `offs_m[:, None] * stride_asm`과 `scale_k[None, :] * stride_ask`를 쓴다. B scale pointer는 `scale_k[:, None] * stride_bsk`와 `offs_n[None, :] * stride_bsn`을 쓴다. payload와 C도 같은 방식으로 자신의 stride를 사용한다.

시작 config:

```python
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
num_warps = 4
num_stages = 2
```

`BLOCK_K`는 `VEC_SIZE`의 배수로 둔다. A/B payload와 scale을 곱한 값은 `tl.dot` 전에 FP16으로 내릴지 FP32 입력으로 둘지 비교할 수 있지만, Tensor Core 경로를 학습하려면 FP16으로 변환하고 accumulator만 FP32로 둔다.

### mask

- A: `offs_m < M`과 `offs_k < K`
- B: `offs_k < K`과 `offs_n < N`
- A scale: A와 같은 유효 M/K
- B scale: B와 같은 유효 K/N
- C store: `offs_m < M`과 `offs_n < N`

K tail을 나중에 지원한다면 payload뿐 아니라 scale load도 함께 mask해야 한다.

## 검증

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/block_scaled_matmul/check.py \
  --gpu B200
```

`check.py`는 `VEC_SIZE=16, 32`와 M/N irregular shape를 검사한다. 현재 wrapper의 `NotImplementedError`를 launch 코드로 바꾸고 TODO 1~4를 완성한다.

오차가 크면 일반 GEMM을 먼저 확인하고 다음 순서로 중간값을 비교한다.

1. `A * expanded_scale_a`
2. `B * expanded_scale_b`
3. K block 하나의 `tl.dot`
4. 여러 K block의 accumulator

## 2단계: 실제 저정밀 payload

explicit-scale kernel이 맞은 뒤 FP8부터 진행한다. FP4는 두 원소가 한 byte에 packed되므로 주소 계산까지 동시에 바뀐다.

공식 format의 주요 차이:

| format | payload | scale vector | 비고 |
|---|---|---:|---|
| MXFP8 | E4M3 계열 | 보통 K축 32개 | 원소당 1 byte |
| MXFP4 | E2M1 계열 | K축 32개 | byte당 2원소 |
| NVFP4 | E2M1 계열 | K축 16개 | NVIDIA 전용 scale 규칙 |

정확한 dtype과 packing은 `triton.tools.mxfp`의 `MXFP4Tensor`, `MXScaleTensor`를 사용해 생성한다. 직접 bit packing부터 구현하지 않는다.

## 3단계: tl.dot_scaled

Blackwell의 block-scaled Tensor Core는 operand dequantization과 MMA를 하나의 명령 경로로 처리한다. Triton에서는 `tl.dot_scaled`로 표현한다.

```python
acc = tl.dot_scaled(
    a,
    scale_a,
    "e4m3",
    b.T,
    scale_b,
    "e4m3",
    acc,
)
```

이때 `scale_a/scale_b`는 단순히 펼친 FP32 행렬이 아니다. `tl.dot_scaled`가 요구하는 scale dtype과 논리 shape을 맞춰야 한다.

이 하드웨어 kernel은 첫 구현 파일에 빈 signature로 미리 넣지 않는다. explicit-scale kernel의 correctness와 benchmark를 끝낸 뒤 `_block_scaled_dot_kernel`을 새로 추가하고 다음 세 부분을 한 단계씩 구현한다.

1. descriptor에서 저정밀 operand와 packed scale load
2. scale을 `tl.dot_scaled`가 요구하는 2D 논리 shape으로 변환
3. `tl.dot_scaled` 누적과 descriptor store

공식 NVIDIA 경로는 A/B tile과 scale tile을 TensorDescriptor로 읽는다. B operand는 format에 따라 `[N, K]`로 저장한 뒤 kernel에서 논리적으로 transpose한다. 첫 API의 `B[K, N]`와 저장 layout이 다르다는 점에 주의한다.

## 4단계: scale preshuffle

Tensor Core의 빠른 K loop가 scale을 연속적으로 읽게 하려면 linear scale layout을 미리 재배치한다. 공식 NVIDIA layout은 scale을 개념적으로:

```text
(non-K / 128, scale-K / 4, 32, 4, 4)
```

형태로 pack한다. kernel에서 load한 뒤 transpose/reshape하여 `tl.dot_scaled`가 보는 2D 논리 shape:

```text
A scale: [BLOCK_M, BLOCK_K / VEC_SIZE]
B scale: [BLOCK_N, BLOCK_K / VEC_SIZE]
```

로 복원한다. 이 변환은 scale 값을 바꾸는 양자화가 아니라 memory access 순서를 바꾸는 layout 최적화다.

처음에는 host에서 pack/unpack round trip을 검사한다.

```text
linear scale -> preshuffle -> unpack -> linear scale
```

이 값이 정확히 같아진 다음 kernel에 연결한다.

## 5단계: 성능 실험

다음 구현을 분리해 측정한다.

1. PyTorch explicit dequantize + matmul
2. Triton explicit-scale fused kernel
3. FP8 `tl.dot_scaled`
4. FP4 `tl.dot_scaled`

1번은 전체 dequantized tensor를 materialize하므로 의미론 reference다. 2번은 이를 HBM에 만들지 않는 fusion 효과를 보여준다. 3/4번은 저정밀 memory traffic과 hardware scaled MMA 효과를 보여준다.

payload/scale 생성과 preshuffle 시간은 kernel-only benchmark에서 제외하고, end-to-end 수치로도 따로 기록한다.

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/block_scaled_matmul/benchmark.py \
  --gpu B200
```

## 하드웨어 경계

portable explicit-scale 단계는 일반 CUDA GPU에서 학습할 수 있다. 공식 block-scaled hardware 경로는 현재 NVIDIA compute capability 10/11 또는 AMD CDNA4를 대상으로 한다. 이 저장소의 Modal `B200`은 Blackwell 경로를 실험하기 적합하다.

지원 여부를 확인한 뒤 hardware kernel을 launch한다.

```python
major, minor = torch.cuda.get_device_capability()
assert major in (10, 11)
assert hasattr(tl, "dot_scaled")
```

## 자주 생기는 오류

- `VEC_SIZE`를 byte당 원소 수와 혼동함
- B scale의 shape/방향을 A와 똑같이 취급함
- FP4인데 K pointer를 원소 수만큼 전진시킴
- scale을 preshuffle했지만 kernel에서 논리 2D layout으로 복원하지 않음
- `tl.dot_scaled` reference를 FP16 matmul만으로 비교해 실제 quantized 값을 무시함
- descriptor block shape와 kernel의 BLOCK 크기가 다름
- hardware 미지원 GPU에서 최종 경로부터 실행함

## 완료 조건

- explicit-scale kernel이 `check.py`의 두 case 통과
- expanded scale tensor를 HBM에 materialize하지 않음
- FP8 quantized reference와 `tl.dot_scaled` 결과 일치
- scale pack/unpack round trip 통과
- FP4에서 byte addressing과 실제 K 원소 수를 구분
- reference/portable/hardware kernel의 시간을 같은 shape에서 기록

## 참고

- [Triton Block Scaled Matrix Multiplication 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
- [Triton Matrix Multiplication 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
