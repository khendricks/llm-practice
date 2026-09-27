"""Calculate multi-head self-attention with PyTorch's built-in module.

Multi-head attention lets a token consider a sentence from several learned
perspectives at once. For example, one head might learn to focus on nearby
words while another learns to connect a pronoun with the noun it refers to.

``nn.MultiheadAttention`` performs the detailed work internally: it learns
query, key, value, and output projections; splits the vectors into heads;
calculates attention for every head; and combines the head outputs. This class
passes the same input as query, key, and value, which makes it self-attention.
During training, PyTorch adjusts the internal projection weights.
"""

import torch
import torch.nn as nn


class PyTorchAttentionV2(nn.Module):
    """Apply PyTorch's multi-head self-attention to input token vectors."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        """Create PyTorch's multi-head attention layer.

        Args:
            embedding_dim: Width of each input and output token vector.
            num_heads: Number of equally sized attention heads.

        Raises:
            ValueError: If the embedding width cannot be split into heads.
        """
        super().__init__()
        if num_heads <= 0 or embedding_dim % num_heads != 0:
            message = "embedding_dim must be divisible by a positive num_heads"
            raise ValueError(message)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            bias=False,
            batch_first=True,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Return one combined context vector for every input token.

        Args:
            inputs: Token vectors shaped
                ``(batch_size, sequence_length, embedding_dim)``.
            is_causal: When true, prevent attention to later positions.

        Returns:
            Context vectors with the same shape as ``inputs``.

        Raises:
            ValueError: If inputs are not a batch of token vectors.
        """
        if inputs.ndim != 3:
            message = "inputs must have batch, sequence, and embedding dimensions"
            raise ValueError(message)

        attention_mask = None
        if is_causal:
            sequence_length = inputs.shape[1]
            attention_mask = torch.ones(
                (sequence_length, sequence_length),
                device=inputs.device,
                dtype=torch.bool,
            ).triu(1)

        context_vectors, _ = self.attention(
            inputs,
            inputs,
            inputs,
            attn_mask=attention_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        return context_vectors
