"""Tests for multi-head attention backed by PyTorch SDPA."""

import torch
import torch.nn.functional as functional

from src.attention.pytorch_attention_v2 import PyTorchAttentionV2


def test_forward_combines_context_vectors_from_each_head() -> None:
    """Split projected vectors into heads and restore their original width."""
    attention = PyTorchAttentionV2(embedding_dim=4, num_heads=2)
    inputs = torch.tensor([[
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0, 2.0],
    ]])
    identity = torch.eye(4)

    with torch.no_grad():
        attention.attention.in_proj_weight.copy_(identity.repeat(3, 1))
        attention.attention.out_proj.weight.copy_(identity)

    context_vectors = attention(inputs)

    heads = inputs.view(1, 2, 2, 2).transpose(1, 2)
    expected_heads = functional.scaled_dot_product_attention(
        heads,
        heads,
        heads,
        dropout_p=0.0,
    )
    expected_context_vectors = expected_heads.transpose(1, 2).reshape(1, 2, 4)

    assert torch.allclose(context_vectors, expected_context_vectors)
