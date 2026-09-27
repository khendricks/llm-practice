"""Tests for self-attention with linear query, key, and value layers."""

import math

import torch

from src.attention.self_attention_v2 import SelfAttentionV2


def test_forward_returns_scaled_linear_projected_context_vectors() -> None:
    """Use separate linear projections before calculating attention."""
    attention = SelfAttentionV2(
        input_dimension=2,
        output_dimension=2,
        qkv_bias=False,
    )
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with torch.no_grad():
        attention.query_weights.weight.copy_(torch.eye(2))
        attention.key_weights.weight.copy_(torch.eye(2))
        attention.value_weights.weight.copy_(
            torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        )

    context_vectors = attention(inputs)

    queries = attention.query_weights(inputs)
    keys = attention.key_weights(inputs)
    values = attention.value_weights(inputs)
    attention_scores = queries @ keys.mT / math.sqrt(keys.shape[-1])
    expected_context_vectors = torch.softmax(attention_scores, dim=-1) @ values

    assert torch.allclose(context_vectors, expected_context_vectors)
