"""Build one context vector for each token with learned self-attention.

The module accepts embeddings shaped ``(tokens, input_dimension)`` or
``(batch, tokens, input_dimension)``. It returns context vectors with
``output_dimension`` features.

Example walkthrough:
    Use two tokens with two features. These values are illustrative.

        inputs
        [ 1.0, 0.0 ]
        [ 0.0, 1.0 ]

    Use fixed linear-layer weights with no bias. Training learns the real
    values at runtime.

        query_weights.weight = [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ]
        key_weights.weight   = [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ]
        value_weights.weight = [ [ 2.0, 0.0 ], [ 0.0, 3.0 ] ]

    1. The three linear layers make queries, keys, and values.

        queries             keys                values
        [ 1.0, 0.0 ]        [ 1.0, 0.0 ]        [ 2.0, 0.0 ]
        [ 0.0, 1.0 ]        [ 0.0, 1.0 ]        [ 0.0, 3.0 ]

    2. Each query is compared with each key, then divided by ``sqrt(2)``.

        attention_scores          scaled_attention_scores
        [ 1.0, 0.0 ]             [ 0.707, 0.000 ]
        [ 0.0, 1.0 ]             [ 0.000, 0.707 ]

    3. Softmax changes each score row into weights that add to one.

        attention_weights
        [ 0.670, 0.330 ]
        [ 0.330, 0.670 ]

    4. Each weight row mixes the value rows into a context vector.

        context_vectors = attention_weights @ values
        [ 1.340, 0.991 ]
        [ 0.660, 2.009 ]

Difference from ``SelfAttentionV1``:
    V1 uses raw parameter matrices. V2 uses ``nn.Linear`` layers and can add
    a bias to the query, key, and value projections.
"""

import torch
import torch.nn as nn


class SelfAttentionV2(nn.Module):
    """Apply scaled dot-product attention through three linear projections."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize learned query, key, and value linear layers.

        Args:
            input_dimension: Number of features in each input token vector.
            output_dimension: Number of features in each projected vector.
            qkv_bias: Whether each projection includes a learnable bias.
        """
        super().__init__()
        self.query_weights = nn.Linear(
            input_dimension,
            output_dimension,
            bias=qkv_bias,
        )
        self.key_weights = nn.Linear(
            input_dimension,
            output_dimension,
            bias=qkv_bias,
        )
        self.value_weights = nn.Linear(
            input_dimension,
            output_dimension,
            bias=qkv_bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a context vector for every input token.

        Args:
            inputs: Token vectors shaped ``(sequence_length, input_dimension)``
                or ``(batch_size, sequence_length, input_dimension)``.

        Returns:
            Context vectors with the input shape except for their final
            ``output_dimension``.
        """
        queries = self.query_weights(inputs)
        keys = self.key_weights(inputs)
        values = self.value_weights(inputs)

        attention_scores = queries @ keys.mT
        scaled_attention_scores = (
            attention_scores / keys.shape[-1] ** 0.5
        )
        attention_weights = torch.softmax(
            scaled_attention_scores,
            dim=-1,
        )

        return attention_weights @ values
