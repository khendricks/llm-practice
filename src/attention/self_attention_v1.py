"""Create one context vector per token with learned self-attention.

The module accepts token embeddings with shape ``(tokens, input_dimension)``
or ``(batch, tokens, input_dimension)`` and returns context vectors whose
last dimension is ``output_dimension``. It first projects every input through
separate learned query, key, and value matrices, then uses query--key
similarity to mix the value vectors.

Walkthrough:
    For three tokens with two-dimensional embeddings, this illustrative
    example labels input rows with words; those labels are not token IDs.

        inputs                   (3 tokens x 2 input dimensions)
        The  [ 1.0, 0.0 ]
        cat  [ 0.0, 2.0 ]
        sat  [ 2.0, 1.0 ]

    Use three distinct fixed projection matrices for this example. A trained
    model learns these values; these matrices only make the walkthrough
    reproducible.

        query_weights = [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ]
        key_weights   = [ [ 1.0, 0.0 ], [ 1.0, 1.0 ] ]
        value_weights = [ [ 1.0, 1.0 ], [ 0.0, 1.0 ] ]

    1. Each input is projected into a distinct query, key, and value vector.

        queries             keys                values
        [ 1.0, 0.0 ]        [ 1.0, 0.0 ]        [ 1.0, 1.0 ]
        [ 0.0, 2.0 ]        [ 2.0, 2.0 ]        [ 0.0, 2.0 ]
        [ 2.0, 1.0 ]        [ 3.0, 1.0 ]        [ 2.0, 3.0 ]

    2. Each query is compared with every key. The module divides the result
       by ``sqrt(key_dimension)``; here, that is ``sqrt(2)``. This limits how
       quickly larger projected dimensions make softmax overly sharp.

        attention_scores = queries @ keys.mT
        [ 1.0, 2.0, 3.0 ]
        [ 0.0, 4.0, 2.0 ]
        [ 2.0, 6.0, 7.0 ]

        scaled_attention_scores = attention_scores / sqrt(2)
        [ 0.71, 1.41, 2.12 ]
        [ 0.00, 2.83, 1.41 ]
        [ 1.41, 4.24, 4.95 ]

    3. Softmax changes each scaled-score row into attention weights that sum
       to one. A row belongs to one query and weights all input-token values.

        attention_weights
        [ 0.140, 0.284, 0.576 ]
        [ 0.045, 0.768, 0.187 ]
        [ 0.019, 0.324, 0.657 ]

    4. Each weight row combines the value rows into one output context vector.

        context_vectors = attention_weights @ values
        [ 1.292, 2.436 ]
        [ 0.419, 2.141 ]
        [ 1.333, 2.638 ]
"""

import torch
import torch.nn as nn


class SelfAttentionV1(nn.Module):
    """Apply scaled dot-product self-attention with learned projections."""

    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        """Initialize the learned query, key, and value projections.

        Args:
            input_dimension: Number of features in each input token vector.
            output_dimension: Number of features in each projected vector.
        """
        super().__init__()
        self.query_weights: nn.Parameter = nn.Parameter(
            torch.rand(input_dimension, output_dimension)
        )
        self.key_weights: nn.Parameter = nn.Parameter(
            torch.rand(input_dimension, output_dimension)
        )
        self.value_weights: nn.Parameter = nn.Parameter(
            torch.rand(input_dimension, output_dimension)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a context vector for each input token.

        Args:
            inputs: Token vectors shaped ``(sequence_length, input_dimension)``
                or ``(batch_size, sequence_length, input_dimension)``.

        Returns:
            Context vectors shaped like ``inputs``, except their final
            dimension is ``output_dimension``.
        """
        queries: torch.Tensor = inputs @ self.query_weights
        keys: torch.Tensor = inputs @ self.key_weights
        values: torch.Tensor = inputs @ self.value_weights

        attention_scores: torch.Tensor = queries @ keys.mT
        scaled_attention_scores: torch.Tensor = (
            attention_scores / keys.shape[-1] ** 0.5
        )
        attention_weights: torch.Tensor = torch.softmax(
            scaled_attention_scores,
            dim=-1,
        )

        return attention_weights @ values
