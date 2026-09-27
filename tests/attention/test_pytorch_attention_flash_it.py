"""GPU integration test for PyTorch's FlashAttention backend."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.attention.pytorch_attention_v2 import PyTorchAttentionV2


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlashAttention requires a CUDA-capable GPU.",
)
def test_pytorch_attention_v2_runs_with_flash_attention() -> None:
    """Run GPT-style multi-head attention using only the flash backend."""
    attention = PyTorchAttentionV2(
        embedding_dim=64,
        num_heads=2,
    ).cuda().half()
    inputs = torch.randn(
        2,
        8,
        64,
        device="cuda",
        dtype=torch.float16,
    )

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        context_vectors = attention(inputs, is_causal=True)

    assert context_vectors.shape == inputs.shape
    assert torch.isfinite(context_vectors).all()
