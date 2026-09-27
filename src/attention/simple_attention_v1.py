"""Calculate self-attention directly from input embedding vectors.

The module accepts embeddings shaped ``(sequence_length, embedding_dim)``
or ``(batch_size, sequence_length, embedding_dim)``. It returns three values:
the pairwise attention scores, their normalized attention weights, and one
context vector per input token. Unlike learned self-attention, the input
vectors serve directly as the queries, keys, and values.

Walkthrough:
    For this illustrative two-token sequence, each vector is orthogonal to
    the other, so each token's self-comparison is its largest score.

        inputs
        [ 1.0, 0.0 ]
        [ 0.0, 1.0 ]

    1. ``inputs @ inputs.mT`` compares every token vector with every other
       vector. Transposing only the final two dimensions preserves a leading
       batch dimension when one is present.

        attention_scores
        [ 1.0, 0.0 ]
        [ 0.0, 1.0 ]

    2. Softmax over the final dimension changes each score row into weights
       that sum to one. Each row belongs to a token and distributes attention
       across all tokens in its sequence.

        attention_weights
        [ 0.731, 0.269 ]
        [ 0.269, 0.731 ]

    3. Matrix multiplication uses each weight row to combine the original
       embeddings, producing a context vector for every token.

        all_context_vectors = attention_weights @ inputs
        [ 0.731, 0.269 ]
        [ 0.269, 0.731 ]
"""

import torch


class SimpleAttentionV1:
    """Compute self-attention without learned query, key, or value layers."""

    def compute(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scores, weights, and one context vector for each token.

        ``inputs`` contains one embedding vector for each input token. The
        dot product compares every vector with every other vector. Softmax
        converts each row of comparison scores into weights, so each token
        can make a weighted combination of all of the input vectors.

        Args:
            inputs: Embeddings shaped ``(sequence_length, embedding_dim)``
                or ``(batch_size, sequence_length, embedding_dim)``.

        Returns:
            A tuple containing attention scores, attention weights, and
            context vectors. Their shapes are respectively
            ``(..., sequence_length, sequence_length)``,
            ``(..., sequence_length, sequence_length)``, and
            ``(..., sequence_length, embedding_dim)``.
        """
        attention_scores = inputs @ inputs.mT
        attention_weights = torch.softmax(attention_scores, dim=-1)
        all_context_vectors = attention_weights @ inputs

        return attention_scores, attention_weights, all_context_vectors
