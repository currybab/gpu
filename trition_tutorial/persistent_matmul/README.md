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
└── README.md
```

공개 함수는 두 개다.

```python
matmul(a, b)             # 1 program = 1 C tile
persistent_matmul(a, b)  # 1 program = 여러 C tile
```

입력은 row-major FP16 `A[M, K]`, `B[K, N]`이고 출력은 FP16 `C[M, N]`이다. 처음부터 M/N/K tail을 지원한다.

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
grid = min(NUM_SMS, 전체 tile 수)
program p -> tile p, p + grid, p + 2*grid, ...
```

kernel 안의 핵심 loop는 다음 모양이다.

```python
start_tile = tl.program_id(0)

for tile_id in tl.range(start_tile, num_tiles, NUM_SMS, flatten=True):
    tile_m, tile_n = linear_tile_to_mn(tile_id)
    # tile마다 acc=0부터 GEMM
    # 결과 저장 후 다음 tile로 이동
```

program이 GPU에 오래 남는다는 의미에서 persistent라고 부른다. launch/scheduling overhead를 줄이고 tile 사이의 pipeline을 발전시킬 여지가 생기지만, GPU의 동적 load balancing을 일부 포기한다. 따라서 항상 일반 GEMM보다 빠른 것은 아니다.

## grouped ordering

linear tile id를 단순 row-major로 풀면 가까운 program들이 A/B tile을 재사용하기 어렵다. `GROUP_SIZE_M`개의 M tile을 한 묶음으로 두고 묶음 안에서 N 방향으로 진행하면 L2 reuse가 좋아질 수 있다.

`persistent_matmul.py`의 `_linear_tile_to_mn`은 일반 kernel과 persistent kernel이 같은 mapping을 공유하게 한다. 먼저 `GROUP_SIZE_M=1`로 검증하고, 이후 8로 바꿔 성능을 비교한다.

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

K loop에서 A pointer는 `BLOCK_K * stride_ak`, B pointer는 `BLOCK_K * stride_bk`만큼 전진한다. 마지막 K tile은 `start_k + offs_k < K`로 mask하고 0을 채운다.

### accumulator와 store

`acc`는 FP32로 만들고 `tl.dot(a, b, acc)`로 갱신한다. 마지막에 FP16으로 변환하고 `offs_m < M`, `offs_n < N` mask로 저장한다.

wrapper는 다음 고정값부터 시작한다.

```python
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
GROUP_SIZE_M = 1
num_warps = 4
num_stages = 2
```

정확도가 맞으면 `GROUP_SIZE_M=8`로 바꾼다.

## 2단계: persistent scheduling

`_persistent_matmul_kernel`의 TODO 4에 기본 tile GEMM 본문을 옮긴다. 중요한 차이는 pointer를 persistent loop 밖에서 한 번 만들어 계속 증가시키면 안 된다는 점이다. 다음 `tile_id`는 전혀 다른 M/N 위치일 수 있으므로 매 iteration에 `tile_m/tile_n`에서 pointer를 다시 계산한다.

launch는 실제 SM 수를 사용한다.

```python
num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
num_programs = min(num_sms, num_tiles)
grid = (num_programs,)
```

kernel meta-parameter `NUM_SMS`에는 실제 grid stride인 `num_programs`를 넘긴다. 이름이 NUM_SMS여도 작은 문제에서는 grid가 SM 수보다 작을 수 있다.

## 검증

```bash
uv run modal run modal_run.py +  --script trition_tutorial/persistent_matmul/check.py +  --gpu B200
```

`check.py`는 기본/persistent 구현을 각각 `torch.matmul`과 비교한다. 아직 구현하지 않은 함수는 `pending`으로 출력하므로 기본 GEMM부터 한 단계씩 진행할 수 있다.

검증 shape에는 tile 배수와 비배수가 모두 있다. 처음에는 첫 shape만 남겨 pointer를 확인하고, 그다음 tail case를 복구한다.

## 3단계: 성능 실험

정확도 통과 후 `triton.testing.do_bench`로 다음을 같은 shape에서 비교한다.

- `torch.matmul`
- 기본 `matmul`
- `persistent_matmul`

```bash
uv run modal run modal_run.py \
  --script trition_tutorial/persistent_matmul/benchmark.py \
  --gpu B200
```

측정 전 warmup하고 결과를 사용해 GPU 동기화를 보장한다. TFLOPS는 다음 식으로 계산한다.

```text
TFLOPS = 2 * M * N * K / (milliseconds * 1e9)
```

변수는 한 번에 하나만 바꾼다.

1. `NUM_SMS`: 실제 SM의 1/2, 1배
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
- grid stride와 kernel의 `NUM_SMS`가 다름
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
