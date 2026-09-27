"""Tests for attention implemented with PyTorch's optimized primitive."""

import math

import torch

from src.attention.pytorch_attention_v1 import PyTorchAttentionV1


def test_forward_returns_scaled_self_attention_context_vectors() -> None:
    """Use the input as queries, keys, and values."""
    attention = PyTorchAttentionV1()
    inputs = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    context_vectors = attention(inputs)

    scores = inputs @ inputs.mT / math.sqrt(inputs.shape[-1])
    expected_context_vectors = torch.softmax(scores, dim=-1) @ inputs

    assert torch.allclose(context_vectors, expected_context_vectors)
