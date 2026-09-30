# B200 4096³ 튜닝 재검증 — 2026-09-30

최초 quick 탐색에는 stage=3과 큰 BK=128 후보가 빠져 있었다.
커널 본문을 변경하지 않고 full preset의 1,080개 후보를 검사했다.
944개가 정확도를 통과했고, 136개는 컴파일/자원 제한으로 제외됐다.
최종적으로 이전 설정, full 탐색의 선택 설정, 공식 예제와 PyTorch를
하나의 B200 실행에서 순서를 섞어 100ms씩 3회 측정했다.

환경: PyTorch 2.13.0+cu130, Triton 3.7.1, FP16, M=N=K=4096.
CUDA Graph 반복 실행으로 출력 할당/JIT 비용을 제외하며 입력을 재사용한다.
PyTorch의 FP16 reduced-precision reduction은 비활성화했다.

## 같은 GPU에서 재측정한 결과

| 커널 | 최초 quick 설정 | full 선택 설정 | 처리량 변화 |
|---|---:|---:|---:|
| TMA tiled | 124.17 μs | 126.23 μs | 0.984× |
| TMA persistent | 158.19 μs | 144.35 μs | 1.096× |
| WS off | 158.12 μs | 144.53 μs | 1.094× |
| WS on | 142.22 μs | 132.12 μs | 1.076× |

따라서 tiled는 기존 설정을 유지하는 편이 낫다. full 탐색 안에서는
BK=32와 BK=64가 0.3% 이내로 가까웠으나, 독립 반복 비교에서는 BK=64가 더 빨랐다.
작은 측정 차이로 고른 승자를 확정적인 최적값으로 취급하지 않는다.

## 재측정 후 사용할 후보

| 커널 | BM/BN/BK | 요청 warps | stages | programs/SM | TFLOPS | PyTorch 대비 |
|---|---|---:|---:|---:|---:|---:|
| TMA tiled | 128/256/64 | 4 | 4 | 전체 tile grid | 1,106.9 | 89.1% |
| TMA persistent | 128/256/64 | 4 | 3 | 2 | 952.1 | 76.6% |
| WS off | 128/256/64 | 4 | 3 | 2 | 951.0 | 76.5% |
| WS on | 128/128/128 | 8 | 3 | 1 | 1,040.3 | 83.7% |

PyTorch 기준은 contiguous B[K,N]에서 110.63 μs / 1,242.3 TFLOPS다.
이 표는 해당 shape와 환경에서 검증한 후보이며 모든 shape의 기본값이 아니다.
공개 wrapper의 고정 기본값을 자동 변경하지 않는다. 튜닝 도구는 kernel을 직접
호출하므로 wrapper 기반 benchmark에 표의 설정이 자동 적용되는 것은 아니다.

## 공식 예제와 남은 차이

공식 TMA persistent + WS는 119.66 μs / 1,148.6 TFLOPS였다.
공식 배치의 PyTorch는 112.41 μs / 1,222.6 TFLOPS이므로 처리량 비율은 93.9%다.
공식 예제는 B[N,K] 사전 전치, GROUP_SIZE_M=8, epilogue subtiling을 사용한다.
전치 복사 67.84 μs는 GEMM 시간에 포함하지 않았다.

누락된 튜닝 후보는 persistent/WS 성능 저하의 일부를 설명하지만 전체 차이를
설명하지는 않는다. 나머지 차이의 원인을 특정 최적화에 귀속하려면 입력 배치,
grouped ordering과 epilogue subtiling을 한 가지씩 바꾸는 별도 실험이 필요하다.

## 도구 수정과 원시 결과

- quick 후보에 stage=3과 큰 BK=128을 추가했다.
- 최종 후보를 기본 100ms × 3회 측정한다.
- PyTorch 기준값을 탐색 후 다시 측정한다.
- `benchmark_official.py --tuned-configs`로 이전/새 설정을 같은 실행에서 비교한다.

원시 파일은 로컬 `modal_out/B200/`에 있다.

- 확대 탐색: `tma_20260930T074959903027Z_summary.json`, `*_all.csv`
- 동일 GPU 최종 비교: `official_20260930T075713566457Z.json`, `.csv`
- 공식 소스 commit: `2e5ace0605bd5005da5c907a25de14d9d1707a98`
