# RTX 5090에서는 비슷하고 B200에서는 차이가 벌어지는 이유

비교 대상은 FP32 row-wise softmax다. 벤치마크 shape은 `M=4096`, `N=256..12672`이고, 유효 대역폭은 입력 1회 read와 출력 1회 write를 기준으로 다음처럼 계산했다.

```text
effective bandwidth = 2 * x.numel() * x.element_size() / kernel_time
```

## 측정 결과

| GPU | Triton | PyTorch | 이론 메모리 대역폭 대비 |
|---|---:|---:|---:|
| RTX 5090 | 약 1,480 GB/s | 약 1,521 GB/s | 82.6% / 84.9% |
| B200 | 약 5,308 GB/s | 약 2,423 GB/s | 66.3% / 30.3% |
| B200, PyTorch 고성능 구간 | - | 약 3,797 GB/s | 47.5% |

5090에서 B200으로 바뀌었을 때의 증가율은 다음과 같다.

- Triton: `5308 / 1480 = 3.59x`
- PyTorch 저성능 구간: `2423 / 1521 = 1.59x`
- PyTorch 고성능 구간: `3797 / 1521 = 2.50x`

5090에서는 두 커널 모두 1.792 TB/s인 디바이스 메모리 대역폭의 83~85%에 도달하므로 구현 차이가 성능에 크게 나타나지 않는다. 반면 B200에서는 어느 쪽도 8 TB/s의 HBM 대역폭을 채우지 못한다. Softmax가 compute-bound 연산으로 바뀌었다기보다는, 커널의 병렬성·리덕션 방식·자원 사용량이 충분한 memory-level parallelism을 만들기 전에 병목이 생긴다고 보는 편이 정확하다.

## PyTorch는 N에 따라 서로 다른 커널을 고른다

Modal B200에서 `x.shape=(1823, 781)`을 profile했을 때 실제 실행된 커널은 다음과 같았다.

```text
softmax_warp_forward<float...>
CUDA time: 4.576 us
```

이 호출의 유효 대역폭은 약 2,489 GB/s로, 벤치마크에서 관찰한 PyTorch 저성능 구간과 일치한다.

PyTorch 2.13의 FP32 CUDA softmax 디스패치는 다음과 같다.

1. `N <= 2048`이면 `dispatch_softmax_forward`를 통해 `softmax_warp_forward`를 실행한다. 한 warp가 한 row를 담당하고 block당 128 threads를 사용한다.
2. `N > 2048`이면 block 기반 커널로 넘어간다.
3. `potential_reg_cnt = ceil(N / block_threads)`가 10보다 작으면 `cunn_SoftMaxForwardReg<potential_reg_cnt>`를 사용한다.
4. 그 외에는 alignment, ILP, shared-memory 조건에 따라 `cunn_SoftMaxForwardSmem` 또는 일반 `cunn_SoftMaxForward`를 사용한다.

소스: [SoftMax.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu), [PersistentSoftmax.cuh](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/PersistentSoftmax.cuh)

이 디스패치가 측정 결과의 불연속 구간과 맞아떨어진다.

| 경계 | B200 PyTorch 성능 변화 | 실행 경로 변화 |
|---|---:|---|
| `N=2048 -> 2176` | 약 2,506 -> 1,611 GB/s | warp 커널에서 `Reg<3>` 커널로 전환 |
| `N=4096 -> 4224` | 약 2,423 -> 1,544 GB/s | `Reg<4>`에서 `Reg<5>`로 전환 |
| `N=9216 -> 9344` | 약 2,233 -> 3,332 GB/s | `Reg<9>` 조건을 벗어나 shared-memory 경로로 전환 |

따라서 B200에서 보인 약 2.4 TB/s와 3.8 TB/s의 두 성능대는 단순 측정 흔들림이 아니라 shape 기반 커널 선택의 결과다. 특히 `N=9344`에서는 `ceil(9344 / 1024) = 10`이 되어 `potential_reg_cnt < 10` 조건을 처음 벗어나고, shared-memory 커널이 오히려 B200에서 더 높은 처리량을 낸다.

## Triton 커널의 자원 사용량

현재 Triton 커널은 `num_warps=8`, `num_stages=1`이고, `N`을 다음 2의 거듭제곱으로 올린 `BLOCK_SIZE`에 맞춰 JIT compile된다. B200에서 확인한 compile 결과는 다음과 같다.

| BLOCK_SIZE | registers/thread | spills | shared memory | register 기준 programs/SM |
|---:|---:|---:|---:|---:|
| 256 | 24 | 0 | 32 B | 10 |
| 512 | 26 | 0 | 32 B | 9 |
| 1024 | 26 | 0 | 32 B | 9 |
| 2048 | 28 | 0 | 32 B | 9 |
| 4096 | 34 | 0 | 32 B | 7 |
| 8192 | 64 | 0 | 32 B | 4 |
| 16384 | 122 | 0 | 32 B | 2 |

`programs/SM`은 레지스터 용량만으로 계산한 상한이며 실제 residency는 hardware의 warp/block 제한에도 걸린다. 그래도 `BLOCK_SIZE=8192 -> 16384`에서 registers/thread가 `64 -> 122`로 증가하고 레지스터 기준 동시 program 수가 `4 -> 2`로 줄어드는 변화는 성능 단차와 정확히 같은 위치에 있다.

```text
N=8192: Triton 약 5.2 TB/s
N=8320: Triton 약 4.1 TB/s
```

`N=8320`부터 `BLOCK_SIZE`가 16384로 커지면서 레지스터 압박이 급증한다. 반면 모든 `num_stages=1` 설정에서 spill은 0이고 shared memory도 32 bytes뿐이므로, 그 이전 구간의 병목은 spill이나 shared-memory 용량이 아니다.

Triton 구현은 각 program이 여러 row를 persistent하게 처리하고, row의 값을 레지스터에 유지한 채 max, exp, sum, normalize를 수행한다. Shape별 `BLOCK_SIZE`로 JIT specialization하면서도 중간 결과를 global memory에 쓰지 않는 점이 B200의 대역폭을 PyTorch의 warp/register 경로보다 더 잘 활용하게 한다.

## `num_stages`의 효과

`tl.range(..., num_stages=4)`는 일부 중간 크기에서 이득이 있었지만 전체적으로 항상 빠르지는 않았다.

| N | `num_stages=1` | `num_stages=4` | 변화 |
|---:|---:|---:|---:|
| 3072 | 3,799 GB/s | 4,104 GB/s | +8.0% |
| 8320 | 4,119 GB/s | 3,099 GB/s | -24.8% |
| 12672 | 5,255 GB/s | 4,250 GB/s | -19.1% |

stage를 늘리면 다음 loop iteration의 작업을 겹칠 여지가 생기지만 동시에 live value도 늘어난다. 이 커널에서는 작은 일부 구간의 ILP 이득보다 큰 `BLOCK_SIZE`에서의 자원 압박이 더 크게 나타났고, B200 전체 범위의 기본값으로는 `num_stages=1`이 더 안정적이었다.

## 결론

5090에서는 두 구현이 이미 메모리 대역폭 상한에 가까워 차이가 가려진다. B200에서는 더 높은 HBM 대역폭을 채우려면 훨씬 많은 동시 memory request가 필요한데, PyTorch의 `softmax_warp_forward`와 `cunn_SoftMaxForwardReg<N>` 경로는 이를 충분히 만들지 못한다. Shape에 따라 shared-memory 경로로 전환되면 성능이 약 3.8 TB/s까지 올라가는 것도 이 해석과 일치한다.

Triton 커널은 shape에 맞게 JIT specialization된 8-warp persistent program을 사용하고, `BLOCK_SIZE <= 8192`에서는 spill 없이 더 높은 병렬성을 유지해 약 5.3 TB/s에 도달했다. 다만 `BLOCK_SIZE=16384`에서는 레지스터 사용량 증가로 동시 program 수가 절반이 되면서 약 4.1 TB/s로 떨어진다.
