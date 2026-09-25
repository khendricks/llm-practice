"""Turn token IDs into learned vectors a GPT model can work with.

The dataset and data loader prepare batches of token IDs. A token ID is an
integer that identifies a piece of text in the tokenizer's vocabulary, but
the number itself does not describe the token's meaning. For example, token
ID 10 is not inherently more similar to ID 11 than it is to ID 500.

Token and position embedding layers keep learnable vectors. The token table
has one vector for every token ID. The position table has one vector for
every sequence position. Their vectors are added so the model knows both
which token it received and where it occurred.

    input token IDs:    [[1, 2, 3],
                         [4, 5, 6]]

    token embeddings:   [[[...], [...], [...]],
                         [[...], [...], [...]]]

The input shape is ``(batch_size, sequence_length)``. The output adds an
``embedding_dim`` dimension, producing
``(batch_size, sequence_length, embedding_dim)``. Later GPT layers use
these vectors rather than the original integer token IDs.
"""

import torch
import torch.nn as nn

from src.config import MAX_SEQUENCE_LENGTH


class InputEmbeddingV1(nn.Module):
    """Combine learned token and absolute-position vectors."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        """Create token and position tables with matching vector widths."""
        super().__init__()
        self.token_embedding_layer = nn.Embedding(
            vocab_size,
            embedding_dim,
        )
        self.position_embedding_layer = nn.Embedding(
            MAX_SEQUENCE_LENGTH,
            embedding_dim,
        )

    # PyTorch calls ``forward`` when we write ``embedding(input_ids)``.
    # Use that module-call form instead of calling ``forward`` directly.
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token embeddings augmented by their absolute positions."""
        sequence_length = input_ids.shape[-1]
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
        )
        token_embeddings = self.token_embedding_layer(input_ids)
        position_embeddings = self.position_embedding_layer(position_ids)
        return token_embeddings + position_embeddings
