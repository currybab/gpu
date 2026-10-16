"""실습 이름과 실행 조건. import 시 GPU 조회/커널 실행은 하지 않는다."""

import torch

from attention_fa2 import attention
from attention_fa1 import attention_fa1
from attention_variants import (
    attention_autotuned,
    attention_persistent,
    attention_split,
    attention_tma,
    attention_tma_ws,
)


IMPLEMENTATIONS = {
    "baseline": attention,
    "fa1": attention_fa1,
    "fa2": attention,
    "split": attention_split,
    "persistent": attention_persistent,
    "tma": attention_tma,
    "tma-ws": attention_tma_ws,
    "autotune": attention_autotuned,
}


def unsupported_reason(name):
    if not torch.cuda.is_available():
        return "CUDA GPU가 필요합니다."
    if name in ("tma", "tma-ws"):
        if torch.version.cuda is None:
            return "TMA 실습은 NVIDIA CUDA 전용입니다."
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            return "TMA 실습은 Hopper 이상에서 실행하세요."
        if name == "tma-ws" and major < 10:
            return "자동 WS 실습은 Blackwell에서 실행하세요."
    return None
