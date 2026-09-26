
import torch

from src.attention.simple_attention_v1 import SimpleAttentionV1

def test_compute_returns_attention_scores() -> None:
    """Compute token-to-token scores, weights, and context vectors."""
    simple_attention = SimpleAttentionV1()
    inputs = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    attention_scores, attention_weights, context_vectors = (
        simple_attention.compute(inputs)
    )

    assert torch.equal(
        attention_scores,
        torch.tensor([
            [14.0, 32.0],
            [32.0, 77.0],
        ]),
    )
    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(2))
    assert context_vectors.shape == inputs.shape
