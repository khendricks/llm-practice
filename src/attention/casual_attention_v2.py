"""Build causal context vectors with attention-weight dropout.

The module accepts embeddings shaped ``(tokens, input_dimension)`` or
``(batch, tokens, input_dimension)``. It returns one context vector per
token, with ``output_dimension`` features. Each token can use its own value
and values before it, but never values from later tokens.

Example walkthrough:
    Use two tokens with two features. These varied fixed values are
    illustrative.

        inputs
        [ 1.0, 2.0 ]  token 0
        [ 3.0, 1.0 ]  token 1

    Use fixed linear-layer weights with no bias. Training learns the real
    values at runtime.

        query_weights.weight = [ [ 1.0, 0.0 ], [ 0.0, 1.0 ] ]
        key_weights.weight   = [ [ 1.0, 0.0 ], [ 0.0, 0.5 ] ]
        value_weights.weight = [ [ 2.0, 1.0 ], [ 0.0, 3.0 ] ]

    The data flows through the module in this order:

        inputs -> Q, K, V projections -> scores -> causal mask
               -> softmax weights -> dropout -> context vectors

    1. The three linear layers create a query, key, and value for each
       token.

        queries             keys                values
        [ 1.0, 2.0 ]        [ 1.0, 1.0 ]        [ 4.0, 6.0 ]
        [ 3.0, 1.0 ]        [ 3.0, 0.5 ]        [ 7.0, 3.0 ]

    2. Each query is compared with each key and divided by ``sqrt(2)``.

        scaled_attention_scores
        [ 2.121, 2.828 ]
        [ 2.828, 6.718 ]

    3. The upper-triangle mask replaces future-token scores with ``-inf``.
       Softmax then makes their weights exactly zero.

        masked_attention_scores       attention_weights
        [ 2.121,  -inf ]              [ 1.000, 0.000 ]
        [ 2.828, 6.718 ]              [ 0.020, 0.980 ]

    4. In training, dropout randomly removes some nonzero attention weights.
       A surviving weight is scaled up to keep the average output stable. In
       evaluation, dropout does nothing. With dropout disabled, the final
       context vectors are:

        context_vectors = attention_weights @ values
        [ 4.000, 6.000 ]
        [ 6.940, 3.060 ]

Difference from ``CausalAttentionV1``:
    V1 always uses every allowed attention weight. V2 applies dropout after
    softmax while training, then uses the remaining weights to mix values.
"""

import torch
import torch.nn as nn

from src.attention.self_attention_v2 import SelfAttentionV2


class CausalAttentionV2(SelfAttentionV2):
    """Apply causal self-attention with dropout on attention weights."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        qkv_bias: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize learned projections and attention-weight dropout.

        Args:
            input_dimension: Number of features in each input token vector.
            output_dimension: Number of features in each projected vector.
            qkv_bias: Whether each projection includes a learnable bias.
            dropout_rate: Fraction of attention weights dropped during
                training. Dropout is disabled during evaluation.
        """
        super().__init__(input_dimension, output_dimension, qkv_bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return causal context vectors with training-time dropout.

        Args:
            inputs: Token vectors shaped ``(sequence_length, input_dimension)``
                or ``(batch_size, sequence_length, input_dimension)``.

        Returns:
            Context vectors with the input shape except for their final
            ``output_dimension``.
        """
        # Turn token embeddings into query, key, and value vectors.
        queries = self.query_weights(inputs)
        keys = self.key_weights(inputs)
        values = self.value_weights(inputs)

        # Compare each query with every key, then scale the scores.
        attention_scores = queries @ keys.mT
        scaled_attention_scores = (
            attention_scores / keys.shape[-1] ** 0.5
        )

        # Hide keys from future positions before scores become probabilities.
        future_token_mask = torch.triu(
            torch.ones(
                inputs.shape[-2],
                inputs.shape[-2],
                device=inputs.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        masked_attention_scores = scaled_attention_scores.masked_fill(
            future_token_mask,
            float("-inf"),
        )
        attention_weights = torch.softmax(masked_attention_scores, dim=-1)

        # Randomly remove weights only during training to regularize attention.
        attention_weights = self.dropout(attention_weights)

        # Use the remaining weights to combine the value vectors.
        return attention_weights @ values
