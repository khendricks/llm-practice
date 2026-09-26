"""Calculate simple self-attention from input embedding vectors.

Input embeddings represent each token as a group of numbers. Self-attention
lets each token compare its embedding with every token embedding in the same
sequence, including itself. A larger comparison score means two embeddings
are more similar according to their dot product.

For one sequence, the input shape is ``(sequence_length, embedding_dim)``.
For example, two tokens with three-value embeddings have this shape:

    inputs: [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]]

Comparing every token with every other token creates an attention-score
matrix with shape ``(sequence_length, sequence_length)``. Applying softmax
turns each row into attention weights that add up to one. Those weights are
then used to combine the input embeddings into one context vector per token.

For example, two simple embeddings produce these approximate outputs:

    inputs:          [[1.0, 0.0],
                      [0.0, 1.0]]

    attention scores: [[1.0, 0.0],
                       [0.0, 1.0]]

    attention weights: [[0.731, 0.269],
                        [0.269, 0.731]]

    context vectors: [[0.731, 0.269],
                      [0.269, 0.731]]

Each token gives the largest weight to itself because its embedding has the
highest dot-product score with itself.

Later, ``InputEmbeddingV1`` will provide batches with shape
``(batch_size, sequence_length, embedding_dim)``. This implementation works
with both shapes by transposing only the final two dimensions.
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
