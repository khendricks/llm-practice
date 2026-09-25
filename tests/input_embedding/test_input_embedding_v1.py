"""Tests for the first input-embedding module."""

import torch
import torch.nn as nn

from src.config import MAX_SEQUENCE_LENGTH
from src.input_embedding.input_embedding_v1 import InputEmbeddingV1


def test_input_embedding_v1_is_a_torch_module() -> None:
    """Provide a module shell for the input-embedding stage."""
    embedding = InputEmbeddingV1(vocab_size=50, embedding_dim=4)

    assert isinstance(embedding, nn.Module)


def test_input_embedding_v1_returns_a_vector_for_each_token() -> None:
    """Expand a batch of token IDs with an embedding dimension."""
    # The embedding table has 50 rows: one possible vector for IDs 0--49.
    # Each row has four values because ``embedding_dim`` is 4.
    embedding = InputEmbeddingV1(vocab_size=50, embedding_dim=4)

    # This batch has two examples, each containing three token IDs.
    # Only table rows 1--6 are used during this particular lookup.
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    # Each ID is replaced with its four-value vector from the table.
    token_embeddings = embedding(input_ids)

    # Preserve the two examples and three IDs per example, then add the
    # four-value embedding vector for each ID: (2, 3) becomes (2, 3, 4).
    assert token_embeddings.shape == (2, 3, 4)


def test_input_embedding_v1_uses_the_shared_sequence_length() -> None:
    """Reserve one positional vector for every configured position."""
    embedding = InputEmbeddingV1(vocab_size=50, embedding_dim=4)

    assert embedding.position_embedding_layer.num_embeddings == (
        MAX_SEQUENCE_LENGTH
    )


def test_input_embedding_v1_adds_absolute_position_vectors() -> None:
    """Distinguish equal token IDs according to their input positions."""
    embedding = InputEmbeddingV1(vocab_size=2, embedding_dim=2)
    with torch.no_grad():
        embedding.token_embedding_layer.weight.zero_()
        embedding.position_embedding_layer.weight.zero_()
        embedding.position_embedding_layer.weight[0] = torch.tensor([1.0, 2.0])
        embedding.position_embedding_layer.weight[1] = torch.tensor([3.0, 4.0])

    embeddings = embedding(torch.tensor([[0, 0]]))

    assert torch.equal(
        embeddings,
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
    )
