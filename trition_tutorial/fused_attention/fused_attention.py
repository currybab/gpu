import torch
import triton
import triton.language as tl


def attention(q, k, v, causal: bool, sm_scale: float):
    """Triton fused-attention 구현 진입점."""
    raise NotImplementedError("Triton fused attention을 구현하세요.")
