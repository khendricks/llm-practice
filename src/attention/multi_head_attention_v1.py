"""Build multi-head causal context vectors from independent attention heads.

Each head has its own learned query, key, and value projections. The heads
attend to the same input tokens independently, then their context vectors are
joined along the feature dimension. As with ``CausalAttentionV2``, a token
can attend to itself and earlier tokens, but never to future tokens.
"""

import torch
import torch.nn as nn

from src.attention.casual_attention_v2 import CausalAttentionV2


class MultiHeadAttentionV1(nn.Module):
    """Apply several independent causal-attention heads in parallel.

    The output feature dimension is ``output_dimension * num_heads`` because
    this wrapper concatenates, rather than averages, the output from every
    head.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        num_heads: int,
        qkv_bias: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize independent causal-attention heads.

        Args:
            input_dimension: Number of features in each input token vector.
            output_dimension: Features produced by each individual head.
            num_heads: Number of independent attention heads to apply.
            qkv_bias: Whether each head's projections include learnable bias.
            dropout_rate: Fraction of attention weights dropped while
                training. Dropout is disabled during evaluation.

        Raises:
            ValueError: If a dimension or the head count is not positive.
        """
        super().__init__()

        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive.")
        if output_dimension <= 0:
            raise ValueError("output_dimension must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        self.heads = nn.ModuleList(
            [
                CausalAttentionV2(
                    input_dimension=input_dimension,
                    output_dimension=output_dimension,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return concatenated causal context vectors from every head.

        Args:
            inputs: Token vectors shaped ``(sequence_length, input_dimension)``
                or ``(batch_size, sequence_length, input_dimension)``.

        Returns:
            Context vectors shaped like ``inputs`` except their final
            dimension is ``output_dimension * num_heads``.
        """
        context_vectors_by_head = [head(inputs) for head in self.heads]
        return torch.cat(context_vectors_by_head, dim=-1)
