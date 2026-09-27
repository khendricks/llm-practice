"""Tests for self-attention with learned projections."""

import math

import torch

from src.attention.self_attention_v1 import SelfAttentionV1


def test_forward_returns_scaled_projected_context_vectors() -> None:
    """Project inputs before calculating scaled attention context vectors."""
    attention = SelfAttentionV1(input_dimension=2, output_dimension=2)
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with torch.no_grad():
        attention.query_weights.copy_(torch.eye(2))
        attention.key_weights.copy_(torch.eye(2))
        attention.value_weights.copy_(
            torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        )

    context_vectors = attention(inputs)

    queries = inputs @ attention.query_weights
    keys = inputs @ attention.key_weights
    values = inputs @ attention.value_weights
    attention_scores = queries @ keys.mT / math.sqrt(keys.shape[-1])
    expected_context_vectors = torch.softmax(attention_scores, dim=-1) @ values

    assert torch.allclose(context_vectors, expected_context_vectors)
