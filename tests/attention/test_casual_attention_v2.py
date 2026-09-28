"""Tests for causal self-attention with dropout."""

import torch

from src.attention.casual_attention_v2 import CausalAttentionV2


def test_dropout_applies_only_while_training() -> None:
    """Dropout removes attention weights during training, not evaluation."""
    attention = CausalAttentionV2(
        input_dimension=2,
        output_dimension=2,
        qkv_bias=False,
        dropout_rate=1.0,
    )
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with torch.no_grad():
        attention.query_weights.weight.copy_(torch.eye(2))
        attention.key_weights.weight.copy_(torch.eye(2))
        attention.value_weights.weight.copy_(torch.eye(2))

    attention.train()

    assert torch.allclose(attention(inputs), torch.zeros_like(inputs))

    attention.eval()

    assert torch.allclose(attention(inputs)[0], inputs[0])
