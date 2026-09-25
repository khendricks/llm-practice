"""Group GPT training examples into batches.

``DatasetV1`` creates one small practice example at a time. A PyTorch
``DataLoader`` collects several of those examples into a batch, so the
model can practice on several sequences in one training step.

For example, a dataset might contain these individual examples:

    example 0:  input [A, B, C]  target [B, C, D]
    example 1:  input [B, C, D]  target [C, D, E]

With a batch size of two, this data loader combines them like this:

    input batch:   [[A, B, C],
                    [B, C, D]]

    target batch:  [[B, C, D],
                    [C, D, E]]

The first dimension is the number of examples in the batch. The second
dimension is the number of tokens in each example.
"""

from torch.utils.data import DataLoader

from src.config import MAX_SEQUENCE_LENGTH
from src.datasets.dataset_v1 import DatasetV1
from src.tokenizers.byte_pair_v1 import BytePairTokenizerV1


class DataLoaderV1(DataLoader):
    """Load batches of input-target token sequences created from text."""

    def __init__(
        self,
        text: str,
        batch_size: int,
        max_length: int = MAX_SEQUENCE_LENGTH,
        stride: int = 128,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ) -> None:
        """Create the underlying dataset and configure its batch loader."""
        self.tokenizer = BytePairTokenizerV1()
        self.dataset = DatasetV1(
            text,
            self.tokenizer,
            max_length,
            stride,
        )
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
