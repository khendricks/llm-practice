"""Tests for multi-head causal self-attention."""

import torch

from src.attention.multi_head_attention_v1 import MultiHeadAttentionV1


def test_forward_concatenates_each_causal_attention_head() -> None:
    """Each head contributes one causal context vector to the output."""
    attention = MultiHeadAttentionV1(
        input_dimension=2,
        output_dimension=1,
        num_heads=2,
        qkv_bias=False,
        dropout_rate=0.0,
    )
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with torch.no_grad():
        for head, value_weights in zip(
            attention.heads,
            (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])),
        ):
            head.query_weights.weight.copy_(torch.tensor([[1.0, 0.0]]))
            head.key_weights.weight.copy_(torch.tensor([[1.0, 0.0]]))
            head.value_weights.weight.copy_(value_weights)

    context_vectors = attention(inputs)

    assert context_vectors.shape == (2, 2)
    assert torch.allclose(context_vectors[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(context_vectors[1], torch.tensor([0.5, 0.5]))
