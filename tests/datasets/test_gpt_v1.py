"""Tests for the GPT next-token dataset."""

from src.datasets.gpt_v1 import GPTDatasetV1
from src.tokenizers.byte_pair_v1 import BytePairTokenizerV1


class StubTokenizer:
    """Return a predefined token sequence for dataset tests."""

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids

    def encode(self, text: str) -> list[int]:
        """Return the configured tokens regardless of the source text."""
        return self.token_ids


def test_create_dataset() -> None:
    """Create input-target pairs shifted by one token."""
    tokenizer = BytePairTokenizerV1()
    dataset = GPTDatasetV1(
        "hello world, hello world!",
        tokenizer,
        max_length=4,
        stride=1,
    )

    assert len(dataset) == 2
    assert dataset[0][0].tolist() == [31373, 995, 11, 23748]
    assert dataset[0][1].tolist() == [995, 11, 23748, 995]


def test_create_dataset_uses_stride_for_overlapping_sequences() -> None:
    """Start each sequence at the configured stride interval."""
    tokenizer = StubTokenizer([0, 1, 2, 3, 4, 5])
    dataset = GPTDatasetV1(
        "ignored",
        tokenizer,
        max_length=3,
        stride=2,
    )

    assert len(dataset) == 2
    assert dataset[0][0].tolist() == [0, 1, 2]
    assert dataset[0][1].tolist() == [1, 2, 3]
    assert dataset[1][0].tolist() == [2, 3, 4]
    assert dataset[1][1].tolist() == [3, 4, 5]


def test_create_dataset_returns_no_examples_when_text_is_too_short() -> None:
    """Require enough tokens for an input and a shifted target."""
    tokenizer = StubTokenizer([0, 1, 2, 3])
    dataset = GPTDatasetV1(
        "ignored",
        tokenizer,
        max_length=4,
        stride=1,
    )

    assert len(dataset) == 0
