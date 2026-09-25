"""Tests for the GPT data loader."""

from src.dataloaders.dataloader_v1 import DataLoaderV1
from src.config import MAX_SEQUENCE_LENGTH


def test_dataloader_returns_batched_input_target_pairs() -> None:
    """Group dataset examples into equally shaped input-target batches."""
    dataloader = DataLoaderV1(
        "hello world, hello world!",
        batch_size=2,
        max_length=3,
        stride=1,
        shuffle=False,
    )

    input_batch, target_batch = next(iter(dataloader))

    assert len(dataloader) == 1
    assert input_batch.shape == (2, 3)
    assert target_batch.shape == (2, 3)
    assert input_batch.tolist() == [
        [31373, 995, 11],
        [995, 11, 23748],
    ]
    assert target_batch.tolist() == [
        [995, 11, 23748],
        [11, 23748, 995],
    ]


def test_dataloader_keeps_incomplete_batch_when_requested() -> None:
    """Keep the final short batch when ``drop_last`` is disabled."""
    dataloader = DataLoaderV1(
        "hello world, hello world!",
        batch_size=2,
        max_length=3,
        stride=1,
        shuffle=False,
        drop_last=False,
    )

    batches = list(dataloader)

    assert len(batches) == 2
    assert batches[0][0].shape == (2, 3)
    assert batches[1][0].shape == (1, 3)


def test_dataloader_uses_the_shared_sequence_length_by_default() -> None:
    """Keep the default training window aligned with model positions."""
    dataloader = DataLoaderV1(
        "hello " * (MAX_SEQUENCE_LENGTH + 2),
        batch_size=1,
        shuffle=False,
    )

    input_batch, _ = next(iter(dataloader))

    assert input_batch.shape == (1, MAX_SEQUENCE_LENGTH)
