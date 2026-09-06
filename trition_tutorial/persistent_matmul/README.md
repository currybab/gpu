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
