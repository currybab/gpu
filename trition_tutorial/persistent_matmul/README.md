# Triton Persistent Matmul 구현 가이드

기본 matmul의 pointer와 K loop가 아직 익숙하지 않다면 Persistent Matmul을 fused attention보다 먼저 해보는 편이 좋다. attention도 여러 `tl.dot`과 상태를 tile 단위로 관리하는 문제인데, persistent matmul에서는 softmax 없이 다음 두 가지에 집중할 수 있기 때문이다.

1. 하나의 output tile을 정확히 계산하는 tiled GEMM
2. 적은 수의 program이 여러 output tile을 반복 처리하는 scheduling

공식 튜토리얼은 일반 GEMM, persistent kernel, TMA, warp specialization, FP8, CLC까지 한 파일에 담는다. 여기서는 raw pointer FP16 구현부터 시작하고 하드웨어별 기능은 마지막에 붙인다.

다만 Persistent Matmul이 fused attention의 필수 선수 과목은 아니다. 기본 tiled matmul과 fused softmax를 이미 이해했다면 attention 트랙을 먼저 진행해도 된다.

## 파일과 목표

```text
persistent_matmul/
├── persistent_matmul.py  # 기본/persistent kernel 뼈대
├── check.py              # torch.matmul과 정확도 비교
├── benchmark.py          # runtime/TFLOPS 비교
├── tune.py               # persistent config 정확도/성능 sweep
└── README.md
```

공개 함수는 다음과 같다. swizzle 버전은 3단계 실습 뼈대다.

```python
matmul(a, b)             # 1 program = 1 C tile
persistent_matmul(a, b)  # 1 program = 여러 C tile
swizzle_matmul(a, b, group_size_m=8)
persistent_swizzle_matmul(a, b, group_size_m=8)
```

입력은 FP16 `A[M, K]`, `B[K, N]`이고 출력은 FP16 `C[M, N]`이다. M/N/K tail뿐 아니라 tensor의 실제 stride도 처음부터 kernel에 전달한다.

일반적인 Triton GEMM처럼 tensor 인자는 `a_ptr`, `b_ptr`, `c_ptr`로 이름을 붙이고, 각 축의 이동 폭은 별도 인자로 받는다.

```python
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
```

`a_ptr`의 타입을 Python에서 따로 선언하지 않아도 launch 때 넘긴 CUDA tensor로부터 pointer type이 정해진다. 반면 stride는 주소 계산 규칙이므로 명시적으로 넘겨야 한다.

## 먼저 알아야 할 tiled GEMM

행렬곱을 다음 block algorithm으로 계산한다.

```text
각 (tile_m, tile_n)를 병렬 실행:
    acc[BLOCK_M, BLOCK_N] = 0  # FP32
    for start_k in range(0, K, BLOCK_K):
        a = A[tile_m, start_k]
        b = B[start_k, tile_n]
        acc += dot(a, b)
    C[tile_m, tile_n] = acc
```

일반 kernel의 grid 크기는 전체 output tile 수다.

```python
num_m_tiles = triton.cdiv(M, BLOCK_M)
num_n_tiles = triton.cdiv(N, BLOCK_N)
grid = (num_m_tiles * num_n_tiles,)
```

GPU scheduler가 각 program을 한 번 실행하고, program 하나는 C tile 하나만 저장한 뒤 종료한다.

## persistent가 바꾸는 것

Persistent kernel은 GEMM 수식을 바꾸지 않는다. launch하는 program 수와 tile 배정만 바꾼다.

```text
일반:
grid = 전체 tile 수
program p -> tile p 하나

persistent:
grid = min(SM 수, 전체 tile 수)
program p -> tile p, p + grid, p + 2*grid, ...
```

kernel 안의 핵심 loop는 다음 모양이다.

```python
start_tile = tl.program_id(0)

for tile_id in tl.range(start_tile, num_tiles, NUM_PROGRAMS):
    tile_m, tile_n = linear_tile_to_mn(tile_id)
    # tile마다 acc=0부터 GEMM
    # 결과 저장 후 다음 tile로 이동
```

program이 GPU에 오래 남는다는 의미에서 persistent라고 부른다. launch/scheduling overhead를 줄이고 tile 사이의 pipeline을 발전시킬 여지가 생기지만, GPU의 동적 load balancing을 일부 포기한다. 따라서 항상 일반 GEMM보다 빠른 것은 아니다.

## 첫 tile ordering은 왜 row-major인가

여기서 row-major는 tensor의 memory layout이 아니라 output tile을 방문하는 순서를 뜻한다. 첫 구현에서는 linear tile id를 단순한 row-major 좌표로 바꾼다.

```python
tile_m = tile_id // num_n_tiles
tile_n = tile_id % num_n_tiles
```

L2 reuse를 개선하는 grouped ordering은 중요한 최적화지만 persistent scheduling의 핵심은 아니다. 기본/persistent 결과가 모두 맞은 뒤 별도 단계로 추가한다. 그래서 초기 `persistent_matmul.py`에는 mapping helper나 `GROUP_SIZE_M`을 미리 넣지 않았다.

## 1단계: 기본 GEMM 완성

`_matmul_kernel`의 TODO 1~3을 구현한다.

### pointer

```python
offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)

a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
```

K loop에서 A pointer는 `BLOCK_K * stride_ak`, B pointer는 `BLOCK_K * stride_bk`만큼 전진한다. 마지막 K tile은 `start_k + offs_k < K`로 mask하고 0을 채운다. C 주소도 `stride_cm`, `stride_cn`을 사용한다.

### accumulator와 store

`acc`는 FP32로 만들고 `tl.dot(a, b, acc)`로 갱신한다. 마지막에 FP16으로 변환하고 `offs_m < M`, `offs_n < N` mask로 저장한다.

wrapper는 다음 고정값부터 시작한다.

```python
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
num_warps = 4
num_stages = 2
```

## 2단계: persistent scheduling

`_persistent_matmul_kernel`의 TODO 4에 기본 tile GEMM 본문을 옮긴다. 중요한 차이는 pointer를 persistent loop 밖에서 한 번 만들어 계속 증가시키면 안 된다는 점이다. 다음 `tile_id`는 전혀 다른 M/N 위치일 수 있으므로 매 iteration에 `tile_m/tile_n`에서 pointer를 다시 계산한다.

launch는 실제 SM 수를 사용한다.

```python
num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
num_programs = min(num_sms, num_tiles)
grid = (num_programs,)
```

kernel meta-parameter `NUM_PROGRAMS`에는 실제 grid stride인 `num_programs`를 넘긴다.

## 검증

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/persistent_matmul/check.py \
  --gpu B200
```

`check.py`는 기본/persistent 구현을 각각 `torch.matmul`과 비교한다. 아직 구현하지 않은 함수는 `pending`으로 출력하므로 기본 GEMM부터 한 단계씩 진행할 수 있다.

검증 shape에는 tile 배수와 비배수가 모두 있다. 처음에는 첫 shape만 남겨 pointer를 확인하고, 그다음 tail case를 복구한다.

## 3단계: grouped ordering

이번 실습은 **program에 배정하는 output tile 좌표의 swizzling**이다. tensor의 stride나 shared memory layout은 그대로 둔다. 기존 두 kernel을 기준으로 파일 아래쪽에 다음 뼈대를 추가했다.

| 기준 kernel | 구현할 kernel | launch 함수 |
| --- | --- | --- |
| `_matmul_kernel` | `_swizzle_matmul_kernel` | `swizzle_matmul` |
| `_persistent_matmul_kernel` | `_persistent_swizzle_matmul_kernel` | `persistent_swizzle_matmul` |

pointer, mask, K loop, `tl.dot`, store와 launch 코드는 복사되어 있다. 두 새 kernel에서 **SWIZZLE 1~3**만 구현하면 된다. 현재 좌표 계산 두 줄은 row-major 임시값이며, wrapper의 `raise NotImplementedError`가 미구현 상태를 `pending`으로 알린다. 각 kernel의 mapping을 완성한 뒤 해당 wrapper의 `raise`를 삭제한다.

### 어떤 순서로 바꾸나

`GROUP_SIZE_M`은 원소 행 수가 아니라 **M축 tile 개수**다. M tile 여러 개를 그룹으로 묶고, 그룹 안에서는 M을 먼저 증가시키고 그다음 N을 증가시킨다. 같은 N tile의 B 데이터를 가까운 program들이 L2에서 재사용할 기회를 만드는 방식이다. 실제 GPU 실행 순서나 성능 향상을 보장하지는 않는다. [공식 튜토리얼의 L2 Cache Optimizations](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations)를 참고한다.

예를 들어 M tile 5개, N tile 3개, `GROUP_SIZE_M=2`라면 `(tile_m, tile_n)` 순서는 다음과 같다.

```text
row-major 처음 6개: (0,0) (0,1) (0,2) (1,0) (1,1) (1,2)
group 0:          (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)
group 1:          (2,0) (3,0) (2,1) (3,1) (2,2) (3,2)
group 2:          (4,0)       (4,1)       (4,2)
```

### TODO 구현 순서

1. **SWIZZLE 1 — 그룹 찾기.** 정상 크기 그룹의 tile 수 `num_tiles_in_group`을 구한다. `tile_id`가 몇 번째 그룹인지 `group_id`, 그 그룹의 첫 M tile이 어디인지 `first_tile_m`을 계산한다.
2. **SWIZZLE 2 — 마지막 그룹 처리.** `group_size_m`은 `GROUP_SIZE_M`과 남은 M tile 수 중 작은 값이다. Triton에서는 `tl.minimum`을 사용한다. 위 예제의 마지막 그룹에서는 2가 아니라 1이다.
3. **SWIZZLE 3 — 그룹 내부 좌표.** 그룹 내부 linear id를 구하고, 나머지로 M축 상대 위치, 몫으로 N축 위치를 만든다. 나눗셈 기준은 실제 `group_size_m`이다. M축 상대 위치에 `first_tile_m`을 더한다. 임시 row-major 두 줄을 이 좌표로 교체한다.

먼저 `_swizzle_matmul_kernel`에 구현한다. 이어 같은 mapping을 `_persistent_swizzle_matmul_kernel`의 **tile loop 안**에 넣는다. persistent에서는 `tl.program_id(0)`가 아니라 매 iteration의 **`tile_id`**를 변환해야 한다. `NUM_PROGRAMS` 간격의 loop와 grid는 그대로 유지한다.

### 구현 후 확인

```bash
uv run python trition_tutorial/persistent_matmul/check.py
uv run python trition_tutorial/persistent_matmul/benchmark.py
```

`check.py`에는 두 swizzle 버전의 `group_size_m=1, 8` 비교가 연결되어 있다. 1은 row-major와 같아야 하고, 8은 그룹보다 M tile이 적은 경우와 마지막 그룹이 짧은 경우에도 맞아야 한다. 기본 block 설정에서 `M=1153`은 M tile 10개로 마지막 그룹 크기가 2다. `M=4096, N=1024`는 여러 그룹과 persistent 반복 처리를 확인한다.

수치 비교에 앞서 위 작은 예제를 Python으로 직접 나열해 모든 좌표가 범위 안에 있고, 중복 없이 정확히 `num_m_tiles * num_n_tiles`개인지 확인해보자. row-major 임시값도 matmul 정답은 맞으므로, **정확도 통과만으로 swizzling 구현 여부를 판단할 수는 없다.**

성능은 `naive tiled ↔ swizzle tiled`, `naive persistent ↔ swizzle persistent`끼리 비교한다. shape, block 크기, warps, stages, programs_per_sm을 동일하게 두고 `group_size_m=1, 2, 4, 8, 16`만 바꿔본다. `tune.py`는 기존 naive persistent config sweep이므로 이 단계의 swizzle 비교는 `benchmark.py`에서 한다.

## 4단계: 성능 실험

정확도 통과 후 `triton.testing.do_bench`로 다음을 같은 shape에서 비교한다.

- `torch.matmul`
- 기본 `matmul`
- `persistent_matmul`
- `swizzle_matmul`
- `persistent_swizzle_matmul`

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/persistent_matmul/benchmark.py \
  --gpu B200
```

후보 config를 한 프로세스에서 비교할 때는 `tune.py`를 사용한다. 각 후보는 먼저 `torch.matmul`과 정확도를 비교하고, 통과한 경우에만 p20/median/p80 runtime과 TFLOPS를 측정한다. register, spill, shared memory도 config별 최초 launch에서 한 번 출력한다.

```bash
# 로컬 GPU
uv run python trition_tutorial/persistent_matmul/tune.py

# B200
uv run modal run modal_run.py \
  --script trition_tutorial/persistent_matmul/tune.py \
  --gpu B200
```

`CONFIGS`의 tuple 순서는 `(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)`다. 최적 config는 GPU와 shape에 종속되므로 RTX 5090 결과를 그대로 B200 기본값으로 사용하지 않는다.

측정 전 warmup하고 결과를 사용해 GPU 동기화를 보장한다. TFLOPS는 다음 식으로 계산한다.

```text
TFLOPS = 2 * M * N * K / (milliseconds * 1e9)
```

변수는 한 번에 하나만 바꾼다.

1. `NUM_PROGRAMS`: 실제 SM의 1/2, 1배, 2배
2. `BLOCK_M/BLOCK_N/BLOCK_K`
3. `GROUP_SIZE_M`
4. `num_warps`, `num_stages`

작은 K에서는 scheduling overhead 감소가 보일 수 있지만, tile 수가 SM에 고르게 나뉘지 않으면 persistent가 느려질 수 있다.

## 5단계: cross-tile pipelining

`persistent_matmul.py` 아래쪽의 `cross_tile_pipelining_kernel`과 호출 함수 `cross_tile_pipelining_matmul`을 사용한다. `_persistent_swizzle_matmul_kernel`의 grouped mapping, stride, mask, GEMM 본문과 launch 설정을 복사한 실습 뼈대다.

기존 K-loop pipelining은 같은 C 타일 안에서 다음 K block의 load와 현재 block의 dot을 겹친다. Cross-tile pipelining은 한 persistent program이 맡는 **다음 C 타일**의 준비를 현재 타일 처리와 겹치는 것이 목표다. 다음 작업의 linear id는 `tile_id + NUM_PROGRAMS`이며, C의 바로 옆 타일이라는 뜻은 아니다.

이번에는 compiler-assisted 방식부터 실습한다. 바깥 `tl.range`의 `flatten=True`는 중첩 loop를 펼치고 파이프라이닝하도록 컴파일러에 요청한다. 옵션을 지정한 것만으로 실제 cross-tile overlap이나 성능 향상이 보장되는 것은 아니다. [tl.range의 flatten 설명](https://triton-lang.org/main/python-api/generated/triton.language.range.html)을 참고한다.

### TODO 진행 순서

1. **CROSS 1 — loop flattening 요청.** 새 커널 바깥 tile loop의 `flatten=False`를 `True`로 바꾼다. 첫 비교에서는 `NUM_STAGES`, block 크기, warps, group 크기, program 수를 유지해 flattening의 영향을 살펴본다. 기존 커널에도 바깥/안쪽 loop의 `num_stages`가 있으므로, 기존 버전을 무조건 “파이프라이닝 없음”이라고 부르지 않는다.
2. **CROSS 2 — 타일별 상태 확인.** 각 타일의 pointer를 현재 좌표에서 만들고 accumulator를 0으로 초기화하는 위치를 확인한다. 이 방식에서는 소스의 `acc` 초기화와 store 위치를 그대로 유지한다. 수동으로 다음 타일 pointer나 별도 accumulator를 추가하는 단계는 아니다.
3. **CROSS 3 — 경계 정확도 검증.** 한 program이 여러 C 타일을 처리할 때도 매 타일의 K 누적이 끝난 뒤 그 타일에 한 번 저장되어야 한다. M/N/K tail과 비연속 stride도 확인한다.
4. **CROSS 4 — 실행 연결.** wrapper의 `raise NotImplementedError`를 삭제하고 `check.py`를 실행한다. `benchmark.py`에는 `cross-tile persistent`가 연결되어 있다. 구현 전에는 두 스크립트에서 `pending`으로 표시된다.

```bash
uv run python trition_tutorial/persistent_matmul/check.py
uv run python trition_tutorial/persistent_matmul/benchmark.py
```

`check.py`는 group=1/8을 각각 확인한다. 기본 설정에서 `4097×1025×97, padding=7`은 output tile이 561개여서 RTX 5090의 340개 program 중 일부가 두 번째 타일을 처리하고, M/N/K tail과 stride까지 함께 검증한다. 다른 GPU에서는 실제 `NUM_PROGRAMS`와 전체 tile 수를 비교해 반복 처리가 있는지 확인한다.

### 효과 확인

비교 기준은 같은 설정의 `persistent_swizzle_matmul`이다. M/N을 고정하고 K를 `128, 512, 4096`으로 바꿔, 타일 경계 비용이 상대적으로 큰 짧은 K에서 이득이 나타나는지 확인한다. `benchmark.py`는 현재 고정 shape 하나를 측정하므로 상단 M/N/K를 바꿔 실행한다.

- 정확도 통과는 계산이 맞다는 뜻이며 cross-tile overlap이 생겼다는 증거는 아니다.
- 실행 시간을 반복 비교하고 regs/thread, spills, shared/CTA도 함께 기록한다. 여러 iteration을 겹치면 살아 있는 데이터가 늘어 자원 사용량이 증가할 수 있다.
- 컴파일 결과의 TTGIR/PTX/SASS를 비교해 loop 구조와 다음 타일 load의 배치를 살펴본다. 새 커널 launch 결과 `k.asm`에서 확인할 수 있다. ncu의 WarpStateStats/SourceCounters로 barrier와 데이터 대기 변화도 보되, 집계 지표만으로 overlap을 확정하지 않는다.
- flattening이 적용되지 않거나 빨라지지 않아도 원인을 기록한다. 수동 flattened loop, TMA, warp specialization은 이 실험 이후의 확장이다.

## 이후 공식 튜토리얼 경로

기본 raw-pointer persistent kernel 뒤에 다음 순서로 확장한다.

1. `@triton.autotune`
2. TensorDescriptor/TMA load와 store
3. `tl.range(..., warp_specialize=True)`
4. FP8 input/output
5. Blackwell Cluster Launch Control

TMA나 warp specialization이 persistent의 정의는 아니다. 핵심은 고정된 program 집합이 여러 tile을 처리하는 scheduling이다.

## 자주 생기는 오류

- persistent loop 밖에서 `acc`를 초기화해 서로 다른 C tile 값이 섞임
- 다음 tile에서도 이전 A/B pointer를 계속 사용함
- grid stride와 kernel의 `NUM_PROGRAMS`가 다름
- K tail load에는 mask했지만 C의 M/N tail store에는 mask하지 않음
- `num_tiles < num_sms`인데 불필요한 program까지 launch함
- performance 비교 전에 correctness와 warmup을 확인하지 않음

## 완료 조건

- 기본 GEMM이 세 검증 shape를 통과
- persistent GEMM이 같은 결과를 통과
- grid가 전체 tile 수가 아닌 고정 program 수임
- 한 persistent program이 둘 이상의 tile을 처리하는 shape 확인
- 기본/persistent/Torch runtime을 같은 조건에서 기록

## 참고

- [Triton Matrix Multiplication 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton Persistent Matmul 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)

## 6단계: TMA tiled matmul 첫 실습

기존 pointer 기반 tiled GEMM을 `TensorDescriptor`의 load/store로 바꾼다.
TMA는 타일 전송 하드웨어 기능이고, 여기서 TensorDescriptor는 전체 tensor의
shape·stride와 전송 tile 크기를 기술하는 호스트 인터페이스다.

이번 범위는 **1 program = 1 C tile**, FP16 입력/출력, FP32 누적,
`warp_specialize=False`다. 타일 좌표 계산과 launch는 준비되어 있다.

### 구현 순서

`persistent_matmul.py` 하단 TMA 실습의 TODO 1~4를 채운다.

1. Python wrapper에서 A/B/C의 `TensorDescriptor`를 만든다.
2. K loop에서 두 descriptor의 `.load([행 시작, 열 시작])`를 호출한다.
3. A tile과 B tile을 `tl.dot`으로 FP32 accumulator에 누적한다.
4. FP16으로 변환한 결과를 C descriptor의 `.store(...)`로 저장한다.

완성한 뒤 wrapper의 `raise NotImplementedError`를 삭제한다. 뼈대 상태에서
`check_tma.py`는 `pending`을 출력한다. 이는 정확도 통과가 아니다.

### shape을 먼저 확인하기

| 대상 | 전체 shape | descriptor block_shape |
| --- | --- | --- |
| A | `[M, K]` | `[64, 32]` |
| B | `[K, N]` | `[32, 64]` |
| C | `[M, N]` | `[64, 64]` |

기존 함수와 동일하게 contiguous B[K,N]을 `tma_matmul(a, b)`에 넘긴다.
B descriptor의 shape은 `[K, N]`, block_shape은 `[BLOCK_K, BLOCK_N]`이다.
`b_desc.load([start_k, start_n])`으로 읽고 `tl.dot(a_tile, b_tile, acc)`로
누적한다. 입력 준비와 dot 모두 전치가 필요 없으며 결과는 `a @ b`다.

`shape`는 전체 행렬, `block_shape`는 한 번에 읽거나 쓰는 tile 크기다.
좌표와 stride는 원소 단위다. 기존의 원소별 pointer/mask 대신 tile의 시작 좌표를
전달한다. 기본 zero padding으로 범위 밖 load는 0이 되며 범위 밖 store는 버려진다.

이번 wrapper는 contiguous FP16으로 제한한다. 바깥 축 stride가 16-byte 정렬이어야
하므로 K/N을 8의 배수로 받는다. M은 임의의 양수이며 K/N도 **tile 크기의 배수일 필요는 없다**.
따라서 `(257, 200, 104)`는 세 축의 tail 처리를 확인한다.

### 실행

저장소 루트에서:

```bash
uv run python trition_tutorial/persistent_matmul/check_tma.py
```

Modal에서 실행하려면:

```bash
uv run modal run modal_run.py --script trition_tutorial/persistent_matmul/check_tma.py --gpu B200
```

정확도 통과 후 기존 pointer 버전의 `tl.load/tl.store`와 descriptor load/store를
비교해보자. 전송 경로는 컴파일된 kernel의 `asm["ptx"]`에서
`cp.async.bulk.tensor` 계열 명령을 찾아 확인할 수 있다.
성능 비교 시 block/warps/stages와
타일 방문 순서를 맞춰야 한다. 현재 기존 pointer kernel과 설정은 다를 수 있다.

다음 실습은 이 본문을 persistent tile loop에 옮기는 것이다. 이후
`EPILOGUE_SUBTILE`, warp specialization 순으로 확장한다.

참고: [공식 Persistent Matmul 튜토리얼](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)의 TMA tiled 부분을 학습 범위의 기준으로 삼았다.

## 7단계: TMA persistent matmul

같은 파일 하단의 `_tma_persistent_matmul_kernel`과 `tma_persistent_matmul`을
사용한다. 입력은 6단계와 같은 A[M,K], contiguous B[K,N]다.
Descriptor 생성과 launch는 준비되어 있고 **TODO TMA P1~P4**가 구현할 부분이다.

일반 TMA tiled에서는 program 하나가 C tile 하나를 맡았다. 여기서는
현재 튜닝 설정은 `grid = min(4 * SM 수, 전체 tile 수)`로 launch하고 program p가
`p, p + NUM_PROGRAMS, p + 2 * NUM_PROGRAMS, ...` 타일을 처리한다.

1. **P1:** persistent loop 안에서 현재 타일의 FP32 accumulator를 0으로 만든다.
2. **P2:** TMA tiled의 descriptor load와 dot을 K loop 안에 옮긴다.
3. **P3:** K 누적이 끝나면 현재 C 타일을 FP16으로 저장한다.
4. **P4:** wrapper의 `raise NotImplementedError`를 삭제한다.

Descriptor는 wrapper에서 한 번 만들고 재사용한다. 반면 시작 좌표와 accumulator는
각 output tile마다 갱신해야 한다. `NUM_PROGRAMS`는 실제 grid 크기와 같다.

```bash
uv run python trition_tutorial/persistent_matmul/check_tma.py
```

검증 스크립트는 tiled/persistent를 각각 검사한다. 하나가 미구현이어도 다른 버전을
검사하며, 마지막 shape은 GPU의 SM 수로부터 `8 * SM + 1`개 output tile을 만들어
여러 타일 처리, 불균등한 마지막 반복, M/N/K tail을 함께 확인한다.

첫 뼈대는 row-major 타일 순서와 `warp_specialize=False`를 사용한다.
Persistent scheduling 자체가 타일 간 전송·연산 겹침이나 성능 향상을 보장하지는 않는다.
먼저 정확도를 통과한 뒤 outer loop flattening과 pipeline 설정을 비교하고,
이후 `EPILOGUE_SUBTILE`, warp specialization으로 확장한다.


### TMA 성능 비교

`benchmark.py`에 `TMA tiled`, `TMA persistent`가 연결되어 있다. 각 구현은
같은 입력에 대한 `torch.matmul` 정확도 검사를 통과한 뒤 ms/TFLOPS를 출력한다.
아직 `NotImplementedError`가 남아 있는 구현은 `pending`으로 표시하고 건너뛴다.
정확도 실패는 측정하지 않고 오류로 중단한다.

```bash
uv run python trition_tutorial/persistent_matmul/benchmark.py
```

모든 구현에 같은 contiguous B[K,N]을 전달하며 전치 복사는 수행하지 않는다.
측정은 기존과 같이 output 할당과 descriptor 생성을
포함하는 wrapper를 `triton.testing.do_bench`로 호출한다.
기본 shape은 `(16384, 16384, 4096)`이며 작은 실험은 `main`의 기본값을 조정한다.

현재 목록의 pointer 버전과 TMA 버전은 block 크기, tile 순서, program 수 등이
다르므로 시간 차이를 TMA 전송만의 효과로 해석하면 안 된다. 먼저 TMA 두 버전의
정확도와 성능을 확인하고, 전송 방식만 비교하려면 나머지 설정을 맞춘다.

## 8단계: Warp specialization 실습

`persistent_matmul.py` 하단의 `_tma_warp_specialized_matmul_kernel`과
`tma_warp_specialized_matmul`을 완성한다. Descriptor 생성, persistent scheduling,
launch와 메타데이터 출력은 준비되어 있다. 첫 설정은 기존 TMA와 같은
`BLOCK_M/N/K=128/128/64`, `num_warps=4`, `num_stages=2`, 최대 `4 * SM` programs다.
입력은 contiguous FP16 `A[M,K]`, `B[K,N]`이며 K/N은 8의 배수로 제한한다.

Warp specialization은 전송과 계산 등의 작업을 서로 다른 warp 역할로 나누는
최적화다. 이 실습에서는 `tl.range`의 자동 분할을 요청하며, warp ID 분기나
barrier를 직접 작성하지 않는다. 먼저 Modal B200에서 진행하는 것을 권장한다.
RTX 5090은 B200과 연산 명령과 자원 제약이 다르므로 별도로 검증한다.
GPU 세대 검사만으로 특정 Triton 버전의 컴파일 성공이나 성능 향상을 보장하지 않는다.

### TODO 순서

1. **WS 1:** outer persistent loop의 `warp_specialize=False`를
   `warp_specialize=WARP_SPECIALIZE`로 바꾼다. `flatten=True`는 유지한다.
2. **WS 2:** inner K loop에 A/B descriptor load와 FP32 `tl.dot` 누적을 작성한다.
   B는 `[K,N]`이므로 이 실습에서는 전치하지 않는다.
3. **WS 3:** K loop 뒤에서 FP16 결과를 `[store_m, store_n]`에 저장한다.
   store 좌표는 공식 예제의 pipelining 우회 방식처럼 별도 counter로 계산해 두었다.
4. **WS 4:** wrapper의 `NotImplementedError`를 삭제한다.

완성 전에는 `check_tma.py`와 `benchmark.py`의 두 WS 항목이 `pending`을 출력한다.
이는 정확도 통과가 아니다. 완성 후 같은 커널의 `warp_specialize=False`와 `True`를
각각 검증한다. small/full tile, M/N/K tail, program당 여러 output tile 처리를 포함한다.

```bash
uv run python trition_tutorial/persistent_matmul/check_tma.py
uv run python trition_tutorial/persistent_matmul/benchmark.py

uv run modal run modal_run.py --script trition_tutorial/persistent_matmul/check_tma.py --gpu B200
uv run modal run modal_run.py --script trition_tutorial/persistent_matmul/benchmark.py --gpu B200
```

성능 비교는 **WS practice (False) 대 WS practice (True)**를 기준으로 한다.
두 경로는 동일한 입력, 타일 크기, program 수와 stage 설정을 사용한다.
기존 TMA persistent와는 store counter 구성도 다르므로 WS 효과만 비교하는 기준으로
쓰지 않는다. 출력된 registers/spills/shared memory와 실행 시간을 함께 기록한다.
`True`를 전달했다는 사실이나 barrier 개수만으로 실제 작업 겹침을 입증할 수는 없다.
필요하면 컴파일된 IR/PTX와 profiler로 분할 결과를 확인한다.

참고: [공식 Persistent Matmul 예제](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html),
[`tl.range` API](https://triton-lang.org/main/python-api/generated/triton.language.range.html).

## 9단계: Epilogue subtiling 실습

`persistent_matmul.py` 하단의 `_tma_epilogue_matmul_kernel`과
`tma_epilogue_matmul`에 전체/분할 저장 경로를 구현했다. K-loop와 FP32 누적은 기존 WS
커널과 같고, `epilogue_subtile=False`는 전체 타일 저장, `True`는 두 조각으로 나눠
저장하는 경로다. 두 경로 모두 정확도 검증을 완료했다. 비교할 때는 먼저 `warp_specialize=False`로
고정해 저장 방식만 비교한다. B는 기존 contiguous `[K,N]`, 타일 순서는 row-major다.

### 구현 순서 기록

1. **EPI 1 — C descriptor:** `True`일 때 `c_desc.block_shape`의 N축 크기를
   `block_n//2`로 바꾼다. 전체 C shape/strides, A/B descriptor와 grid는 그대로 둔다.
2. **EPI 2 — accumulator 분리:** `tl.reshape`, `tl.permute`, `tl.split`을 이용해
   `[BM,BN]`을 왼쪽/오른쪽 `[BM,BN//2]`로 나눈다. 코드의 shape 힌트를 따라간다.
   `tl.split`은 마지막 크기 2인 축을 분리하므로 permute가 필요하다.
3. **EPI 3 — 두 번 저장:** 두 조각을 각각 FP16으로 바꿔 저장한다.
   오른쪽 시작 열은 `store_n + BLOCK_N//2`다. tile 간격 자체는 `BLOCK_N`이다.
4. **EPI 4 — 실행:** wrapper의 미구현 `if/raise`를 제거하고 두 경로를 검증했다.

### 작은 설정에서 정확도부터 확인

```bash
uv run python trition_tutorial/persistent_matmul/check_epilogue.py

uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/persistent_matmul/check_epilogue.py
```

기본 설정은 `128/128/64`, warps=4, stages=2, programs/SM=1이다.
검사에는 full tile, 오른쪽 절반 전체가 범위 밖인 경우, 오른쪽 일부만 유효한 경우,
M/N/K tail과 program당 여러 C tile을 처리하는 경우가 포함된다.
shape 검사와 비교 허용 오차는 기존 TMA 실습과 같다. pending이나 resource_limited는
정확도 통과가 아니며, 출력이 틀리면 즉시 실패한다.

### 고정 설정 비교와 shared memory 제한 확인

정확도 통과 후 `--benchmark`로 False/True 시간을 비교한다. 같은 커널에서 flag만
바꾸며, 기본 성능 측정 shape은 4096³이다.

```bash
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/persistent_matmul/check_epilogue.py \
  --script-args '--benchmark'

# 기존 persistent가 shared memory 부족으로 실패했던 설정
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/persistent_matmul/check_epilogue.py \
  --script-args '--block-n 256 --num-stages 4 --benchmark'
```

두 번째 실행의 목표는 **False의 자원 부족과 True의 실행 가능 여부**를 확인하는 것이다.
False가 컴파일되지 않으면 같은 설정의 속도 향상 배율을 계산할 수 없다.
먼저 둘 다 실행되는 설정에서 비교하고, 이후 True 경로의 타일/stage를 재튜닝한다.
WS를 함께 비교할 때는 위 명령에 `--warp-specialize`를 추가한다.

출력의 `shared/CTA`, registers/spills와 시간을 함께 기록한다.
FP16 128×256 결과는 64 KiB, 절반인 128×128은 32 KiB다. 저장용 shared memory를
줄이는 것이 목표이며 전체 accumulator나 shared memory 총량을 절반으로 만드는
기능은 아니다. 실제 절약량은 컴파일 메타데이터로 확인한다.

이 스크립트의 시간은 기존 `benchmark.py`처럼 wrapper를 `do_bench`로 측정한다.
출력/descriptor를 미리 준비하는 `tune_tma.py`의 CUDA Graph 결과와 직접 비교하지 않는다.
설정을 바꾼 뒤에는 먼저 이 실습 검사로 정확도를 확인한다.
참고: [공식 epilogue subtiling 구현](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html).

### Subtiling × stages × WS 조합 비교

EPI 구현과 정확도 검사를 끝낸 뒤에는 아래 명령으로 같은 타일에서 조합별 효과를 본다.

```bash
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/persistent_matmul/benchmark_epilogue.py
```

기본값은 4096³, `BM/BN/BK=128/256/64`, warps=4, programs/SM=1이다.
stages 2/3/4/5 × WS False/True × subtiling False/True의 16개 조합을 검사한다.
컴파일된 조합은 NaN으로 초기화한 출력의 정확도를 확인하고, 출력/descriptor를
미리 준비한 CUDA Graph를 순서를 섞어 100ms씩 3회 측정한다.
JSON/CSV/Markdown 결과는 `modal_out/B200/epilogue_*`에 저장한다.

먼저 같은 stages/WS에서 subtiling만 비교한다. 그다음 각각의 경로에서 사용할 수 있는
최고 stage 설정의 성능을 비교한다. 메모리 절약으로 더 높은 stage가 가능해지는 효과와
동일 설정에서 저장 방식을 바꾼 효과를 구분한다. WS는 subtiling의 필수 조건이 아니며,
stage 수를 늘렸다고 항상 빨라지는 것도 아니다.

`--block-n 128`, `--num-warps 8`, `--programs-per-sm 2` 등으로 조건을 바꿀 수 있다.
이 도구는 kernel을 직접 호출하고 각 조합의 정확도를 확인한 뒤 측정한다.

### 구현 전체를 같은 GPU에서 재측정

```bash
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/persistent_matmul/benchmark_all.py
```

4096³·FP16을 기본으로 raw tiled/persistent, swizzle, cross-tile, TMA, WS, epilogue,
공식 예제 6종과 PyTorch를 한 GPU 할당에서 비교한다. 계산은 서로 간섭하지 않도록
순서를 섞어 순차 실행하며, 100ms × 3회 CUDA Graph 측정의 중앙값을 사용한다.
`--script-args '--size 4096 --rounds 5'`처럼 반복 수를 바꿀 수 있다.

우리 커널은 현재 wrapper 기본값, 최초 quick 선택값, 확대 탐색 선택값을 구분한다.
Epilogue는 기본값과 stages 2/3/4/5 × WS × subtiling 조합을 포함한다.
이것은 기존 설정들의 통합 재측정이며 모든 구현을 새로 최적화하는 실험은 아니다.
공식 예제는 고정된 원본 소스의 autotune을 이번 실행에서 수행해 설정을 선택한다.
공식 CLC/device-side descriptor 변형은 지금까지의 학습 범위에 포함하지 않았다.

`all_gemm_*.md/.csv/.json`에 각 행의 구현, 설정 출처, 정확도/컴파일 상태,
B 배치, flatten/WS/subtiling, grouped ordering, 타일 크기, 요청·실제 warp 수,
launch/loop stage, programs/SM·실제 grid와 시간·TFLOPS·shared memory를 기록한다.
우리 입력은 B[K,N], 공식 입력은 사전 전치한 B[N,K]이며, 두 PyTorch 기준값을
따로 측정한다. B 전치 복사 비용은 GEMM에 포함하지 않고 별도 기록한다.

### 큰 TMA 타일의 제한된 비교

```bash
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/persistent_matmul/check_large_tma_tiles.py
```

4096³·FP16, warps=8, persistent programs/SM=1을 고정하고
`128/256/64`, `256/256/32`, `256/256/64` × stages 3/4 ×
TMA tiled/persistent/WS/WS+subtiling의 24개 조합만 비교한다.
통과한 조합은 50ms씩 3회 CUDA Graph로 측정하며 `large_tma_*.json/.csv/.md`로 저장한다.
이 범위에서의 실패를 다른 stage/warp 설정에서도 실행 불가능하다는 뜻으로 해석하지 않는다.

### Raw pointer·swizzle·cross-tile 확대 튜닝

```bash
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/persistent_matmul/tune_raw.py \
  --script-args '--preset full --size 4096 --top-k 5 --max-seconds 1500'
```

다섯 raw 구현을 `BM/BN={64,128,256}`, `BK={32,64,128}`, warps={4,8},
stages={2,3,4}로 탐색한다. Persistent 구현은 programs/SM={1,2,4}도 비교한다.
첫 탐색은 grouped 구현의 GROUP_SIZE_M=8을 고정해 총 1,782개 후보를 검사한다.
그룹 크기는 구현별 상위 5개 후보에서 1/4/16/32로 추가 탐색하므로 전체 그룹 조합을
전수 탐색했다는 의미는 아니다.

Raw tiled/swizzle tiled의 기존 K-loop stages=4는 기본값을 유지한 `NUM_STAGES=4`
인자로 바꿨다. 후보 탐색에서는 launch와 명시적 loop stage에 같은 값을 사용한다.
기본값 비교는 기존대로 tiled의 launch=3 / K-loop=4, persistent의 launch=4 /
명시적 loop=4를 유지한다. Cross-tile은 outer loop에 stage 값을 지정한다.

모든 후보는 정확도 검사를 통과한 뒤 CUDA Graph로 측정한다. 컴파일/자원 제한은
기록하고 건너뛰며, 정확도 실패나 CUDA 실행 오류는 중단한다. 탐색 후 기본값과 상위
후보, PyTorch 및 기존 TMA 참고 설정들을 같은 GPU에서 순서를 섞어 100ms씩 3회
측정한다. 기본값이 더 빠르면 기본값을 선택한다. `--max-seconds`는 탐색 단계에서
후보 사이에 확인하는 예산이며, 시작한 컴파일과 최종 비교의 완료 시간은 별도다.

`raw_tuning_*.json`, `*_trials.csv`, `*_comparison.csv`, `.md`가 결과다.
완료 여부와 후보 범위를 확인한 뒤 해석한다. 작은 실행 점검에는 `--preset smoke`
를 사용한다. 이 도구도 kernel 직접 호출이므로 결과가 기존 wrapper 기본값에
자동 적용되지는 않는다.

## Shape별 TMA 파라미터 탐색 (Modal B200)

`tune_tma.py`는 작성한 kernel을 직접 호출해 `TMA tiled`, `TMA persistent`,
`WS off`, `WS on`을 각각 튜닝하고 `torch.mm`과 비교한다. 기존 wrapper의 고정
기본값은 바꾸지 않는다. shape 표기는 **M×N×K**이며 입력은 FP16 A[M,K], B[K,N]이다.

먼저 작은 smoke 실행으로 컴파일과 정확도를 확인한다.

```bash
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/persistent_matmul/tune_tma.py \
  --script-args '--preset smoke --max-seconds 300'
```

기본 탐색은 아래처럼 실행한다. 1024/2048/4096/8192 정사각형과
`8192×2048×4096`, `2048×8192×4096`, `16384×16384×4096`을 검사한다.

```bash
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/persistent_matmul/tune_tma.py \
  --script-args '--preset quick --max-seconds 1200'
```

| Preset | BM/BN/BK 후보 | warps | stages | persistent programs/SM |
|---|---|---|---|---|
| smoke | 64/64/32, 128/128/64 | 4 | 2 | 1, 4 |
| quick | 64/64/32, 128/64/64, 128/128/64, 128/256/64, 128/128/128, 128/256/128 | 4, 8 | 2, 3, 4 | 1, 2, 4 |
| full | BM=64/128 × BN=64/128/256 × BK=32/64/128 | 4, 8 | 2, 3, 4 | 1, 2, 4 |

`quick`은 shape당 tiled 36개, 나머지는 각각 108개 후보를 검사한다.
초기 quick 탐색은 stage=3과 큰 BK=128 후보를 포함하지 않았으므로 그 결과를
커널의 최적 성능으로 해석하지 않는다. 현재 quick은 이 후보들을 포함한다.
Tiled의 programs/SM은 CSV에서 0으로 표시하며 전체 tile 수로 launch한다.
Persistent의 실제 grid는 `min(SM 수 * programs_per_sm, tile 수)`다.
작은 shape에서는 서로 다른 programs/SM 후보가 같은 grid가 될 수 있다.

관심 shape만 더 촘촘히 탐색하거나 특정 구현만 골라 실행할 수 있다.

```bash
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/persistent_matmul/tune_tma.py \
  --script-args '--preset full --shapes 4096x4096x4096 8192x8192x8192 --variants ws_off ws_on --max-seconds 1200'

# 같은 도구로 로컬 GPU에서 실행
uv run python trition_tutorial/persistent_matmul/tune_tma.py \
  --preset quick --shapes 4096x4096x4096
```

### 측정 기준과 결과 읽기

- 모든 후보는 출력에 NaN을 채운 뒤 `torch.matmul` reference와 정확도를 비교한다.
  컴파일/자원 부족과 정확도 실패는 기록하고 순위에서 제외한다. CUDA 실행 오류는
  context가 손상됐을 수 있으므로 탐색을 중단한다.
- 출력 할당, descriptor 생성, JIT 컴파일과 Python launch 비용을 제외한
  **CUDA Graph 반복 실행 시간**을 측정한다. 동일한 입력을 재사용하므로 작은
  행렬은 L2 cache 효과가 크다. 기존 wrapper 기반 `benchmark.py`와 측정 조건이 다르다.
- FP16 입력, FP32 누적을 비교하며 PyTorch의 FP16 reduced-precision reduction은 끈다.
  TFLOPS는 `2*M*N*K / (median_ms * 1e9)`다.
- 후보 순서는 고정 seed로 섞는다. 기본 20ms 측정 후 구현별 상위 3개를 100ms씩
  3회 재측정하고 각 통계의 중앙값으로 최종 순위를 정한다 (`--final-repeats`).
  PyTorch도 탐색이 끝난 뒤 다시 측정한다. p20/median/p80, 반복별 중앙값과
  register/spill/shared memory도 저장한다.
- `--max-seconds`는 후보 사이에서 확인하는 시간 예산이다. 이미 시작한 컴파일/측정은
  끝까지 진행한다. 예산이 끝나면 부분 결과를 저장하고 `complete=false`로 표시한다.
  `screening_only`는 상위 후보 재측정까지 끝나지 않은 잠정 결과다.
- **최고 TFLOPS는 탐색한 shape/config 범위 안의 최고값**이다. 각 구현의 최적값 비교와
  WS 자체의 효과는 구분한다. WS 효과를 확인하려면 `all.csv`에서 shape와 모든 설정이
  같은 `ws_off`/`ws_on` 행을 비교한다.

Modal 결과는 `modal_out/B200/`, 로컬 결과는 `tuning_results/`에 저장한다.
파일명에 UTC 시각을 붙여 실행 간 덮어쓰기를 피한다.

| 파일 | 내용 |
|---|---|
| `*_all.csv` | 모든 시도와 실패 이유, 설정, 시간, 자원 사용량 |
| `*_best.csv` | shape/구현별 최고 설정과 PyTorch 대비 배율 |
| `*_summary.json` | GPU·Torch·Triton·kernel hash·탐색 범위·완료 여부와 최고 설정 |
| `*_report.md` | 비교 표와 구현별 최대 TFLOPS를 낸 shape |
| `*_best.png` | shape별 TFLOPS와 PyTorch 대비 배율 그래프 |

`full`은 후보 수가 많으므로 관심 shape로 범위를 좁혀 사용하는 편이 좋다.
측정 구현: [`triton.testing.do_bench_cudagraph`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench_cudagraph.html).

### 공식 Persistent Matmul 예제와 비교

```bash
uv run modal run modal_run.py --gpu B200 \
  --script trition_tutorial/persistent_matmul/benchmark_official.py \
  --script-args '--size 4096'
```

`benchmark_official.py`는 공식 09 예제를 고정된 Git commit과 SHA256으로 내려받아
원본 kernel과 autotune 후보를 사용한다. Raw tiled/persistent와 TMA tiled/persistent의
WS off/on을 비교한다. CLC와 device-side descriptor 변형은 이 비교에서 제외한다.
Autotune이 고른 설정을 고정하고 출력/descriptor를 미리 할당한 뒤 CUDA Graph로
100ms씩 3회 측정한 중앙값을 기록한다. 공식 CLI의 Proton 측정을 그대로 실행하는
것은 아니며, 이 저장소의 GPU 시간 측정 방식에 맞춘 비교다.

공식 예제는 B를 미리 contiguous `[N,K]`로 저장한다. 공식 커널의 PyTorch 대비
배율은 `torch.mm(a, b_transposed.T)`를 기준으로 계산한다. 기존 커널도 같은 GPU에서
이전 4096³ 최고 설정으로 재측정하며, 이쪽은 원래 contiguous `[K,N]` PyTorch를
기준으로 삼는다. `--size`를 바꿔도 기존 커널 설정은 다시 튜닝하지 않는다.
전치 복사 비용은 GEMM 시간에 포함하지 않고 `transpose_copy_ms`로 별도 기록한다.
모든 선택된 구현을 동일한 FP16 데이터의 reference와 검증한 뒤 측정한다.

결과 JSON/CSV와 실행한 공식 Python 원본은 `modal_out/B200/`에 저장한다.

`--tuned-configs`에 `tune_tma.Config` 필드를 담은 JSON을 전달하면 최초 quick 설정과
새 설정을 같은 GPU/입력에서 번갈아 3회 재측정한다. 예를 들어
`{"ws_on":{"block_m":128,"block_n":256,"block_k":64,"num_warps":4,"num_stages":3,"programs_per_sm":1}}`
형식이다. 이는 설정 전달 예시이며 최고 설정이라는 의미는 아니다.
