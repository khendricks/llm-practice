"""Integration tests for the first input-embedding module."""

from src.dataloaders.dataloader_v1 import DataLoaderV1
from src.input_embedding.input_embedding_v1 import InputEmbeddingV1


def test_input_embedding_v1_accepts_an_input_batch_from_the_dataloader(
) -> None:
    """Connect token IDs from the data loader to the embedding layer."""
    dataloader = DataLoaderV1(
        "hello world, hello world!",
        batch_size=2,
        max_length=3,
        stride=1,
        shuffle=False,
    )
    input_ids, target_ids = next(iter(dataloader))
    embedding_dim = 4
    embedding = InputEmbeddingV1(
        vocab_size=50_257,
        embedding_dim=embedding_dim,
    )

    token_embeddings = embedding(input_ids)

    assert target_ids.shape == input_ids.shape
    assert token_embeddings.shape == (*input_ids.shape, embedding_dim)
