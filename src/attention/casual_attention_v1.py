"""Build causal context vectors with masked self-attention.

The module accepts embeddings shaped ``(tokens, input_dimension)`` or
``(batch, tokens, input_dimension)``. It returns one context vector per
token, with ``output_dimension`` features. A token can use its own value and
the values before it, but never a later token's value.

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
               -> softmax weights -> context vectors

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

    4. Each weight row mixes only the allowed value rows into the output.

        context_vectors = attention_weights @ values
        [ 4.000, 6.000 ]
        [ 6.940, 3.060 ]

Difference from ``SelfAttentionV2``:
    This class uses the same learned projections and scaled scores. It adds
    the causal mask before softmax, preventing information from the future.
"""

import torch

from src.attention.self_attention_v2 import SelfAttentionV2


class CausalAttentionV1(SelfAttentionV2):
    """Apply self-attention without allowing tokens to see the future."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return context vectors using only the current and prior tokens.

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

        return attention_weights @ values
