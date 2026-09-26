"""Integration tests for the first simple-attention module."""

from src.attention.simple_attention_v1 import SimpleAttentionV1
from src.dataloaders.dataloader_v1 import DataLoaderV1
from src.input_embedding.input_embedding_v1 import InputEmbeddingV1


def test_simple_attention_v1_accepts_embeddings_from_the_input_layer(
) -> None:
    """Connect data-loader token IDs through embeddings into attention."""
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
    attention = SimpleAttentionV1()

    token_embeddings = embedding(input_ids)
    attention_scores, attention_weights, context_vectors = attention.compute(
        token_embeddings,
    )

    assert target_ids.shape == input_ids.shape
    assert attention_scores.shape == (*input_ids.shape, input_ids.shape[-1])
    assert attention_weights.shape == attention_scores.shape
    assert context_vectors.shape == (*input_ids.shape, embedding_dim)
