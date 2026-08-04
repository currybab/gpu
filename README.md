- [2026-07-25 vector add(triton & cuda)](./pmpp_v2/vectoradd_py/)
- [2026-07-25 fused softmax(triton & cuda)](./trition-tutorial/fused-softmax/)

## Modal에서 실행하기

로컬 GPU 대신 Modal의 GPU에서 같은 스크립트를 돌린다. 최초 1회 `uv run modal setup` 필요.

```sh
uv run modal run modal_run.py                 # fused_softmax, H100
uv run modal run modal_run.py --gpu B200      # 5090과 같은 Blackwell 세대
uv run modal run modal_run.py --script pmpp_v2/vectoradd_py/submission_triton.py
```
