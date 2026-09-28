"""Tests for causal self-attention."""

import torch

from src.attention.casual_attention_v1 import CausalAttentionV1


def test_forward_masks_future_tokens() -> None:
    """Each context vector uses its token and tokens before it only."""
    attention = CausalAttentionV1(
        input_dimension=2,
        output_dimension=2,
        qkv_bias=False,
    )
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with torch.no_grad():
        attention.query_weights.weight.copy_(torch.eye(2))
        attention.key_weights.weight.copy_(torch.eye(2))
        attention.value_weights.weight.copy_(torch.eye(2))

    context_vectors = attention(inputs)
    changed_inputs = inputs.clone()
    changed_inputs[1] = torch.tensor([100.0, 100.0])

    assert torch.allclose(context_vectors[0], inputs[0])
    assert torch.allclose(attention(changed_inputs)[0], inputs[0])
