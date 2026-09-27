"""Calculate self-attention with PyTorch's optimized implementation.

``scaled_dot_product_attention`` uses the input vectors as queries, keys,
and values here. This keeps the class comparable to ``SimpleAttentionV1``.
PyTorch selects an efficient backend, such as a fused attention kernel, when
the device, tensor shapes, and data type support one.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class PyTorchAttentionV1(nn.Module):
    """Apply PyTorch scaled dot-product self-attention to input vectors."""

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Return a context vector for each input token.

        Args:
            inputs: Token vectors shaped ``(sequence_length, embedding_dim)``
                or ``(batch_size, sequence_length, embedding_dim)``.
            is_causal: When true, prevent tokens from attending to later
                positions. GPT-style language models use this setting.

        Returns:
            Context vectors with the same shape as ``inputs``.
        """
        return functional.scaled_dot_product_attention(
            inputs,
            inputs,
            inputs,
            dropout_p=0.0,
            is_causal=is_causal,
        )
