- [2026-07-25 vector add(triton & cuda)](./pmpp_v2/vectoradd_py/)
- [2026-08-04 fused softmax(triton & cuda)](./trition_tutorial/fused_softmax/)
- [Persistent Matmul 구현 가이드](./trition_tutorial/persistent_matmul/)
- [Group GEMM 구현 가이드](./trition_tutorial/group_gemm/)
- [Block-scaled Matmul 구현 가이드](./trition_tutorial/block_scaled_matmul/)
- [fused attention 구현 가이드](./trition_tutorial/fused_attention/)

## 권장 학습 순서

공통 기초는 `vector add → fused softmax → 기본 tiled matmul`이다. 그 뒤에는 목표에 따라 나눈다.

- Attention 트랙: `기본 matmul + fused softmax → fused attention`
- GEMM scheduling 트랙: `기본 matmul → persistent matmul → group GEMM`
- 저정밀 하드웨어 트랙: `기본 matmul → block-scaled matmul`

Group GEMM과 Block-scaled Matmul은 fused attention의 선수 과목이 아니다. attention 구현이 현재 목표라면 fused attention을 계속하고, Triton의 scheduling 최적화를 체계적으로 익히고 싶다면 Persistent Matmul을 먼저 진행한다. Block-scaled Matmul은 FP8/FP4와 Blackwell 전용 개념이 추가되므로 가장 나중에 보는 편이 좋다.

## Modal에서 실행하기

로컬 GPU 대신 Modal의 GPU에서 같은 스크립트를 돌린다. 최초 1회 `uv run modal setup` 필요.

```sh
uv run modal run modal_run.py                 # fused_softmax, B200
uv run modal run modal_run.py --diagnostics   # Triton 레지스터/스필 + Torch CUDA 커널 확인
uv run modal run modal_run.py --gpu B200      # 5090과 같은 Blackwell 세대
uv run modal run modal_run.py --script pmpp_v2/vectoradd_py/submission_triton.py
uv run modal run modal_run.py --script trition_tutorial/persistent_matmul/check.py
uv run modal run modal_run.py --script trition_tutorial/group_gemm/check.py
uv run modal run modal_run.py --script trition_tutorial/block_scaled_matmul/check.py
```
