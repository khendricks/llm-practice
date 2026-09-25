"""Prepare next-token practice examples for teaching a GPT.

Before a model can learn from text, the text is converted into token IDs:
numbers that represent pieces of text. This module takes one long list of
those numbers and cuts it into many small practice examples.

In each example, the model sees the ``input`` tokens and tries to guess
the ``target`` tokens. The targets are the same sequence moved forward by
one position, so each answer is the next token after an input token.

    token stream:  [A, B, C, D, E, F]

    example 0:    input  [A, B, C]
                  target [B, C, D]

    example 1:    input  [B, C, D]
                  target [C, D, E]

A PyTorch ``Dataset`` stores these examples and lets PyTorch retrieve one
at a time. A ``DataLoader`` can later collect several examples into a
batch, which lets the model practice several at once.
"""

import torch
from torch.utils.data import Dataset

from src.tokenizers.byte_pair_v1 import BytePairTokenizerV1


class DatasetV1(Dataset):
    """Create overlapping input and target token sequences from text."""

    def __init__(
        self,
        text: str,
        tokenizer: BytePairTokenizerV1,
        max_length: int,
        stride: int,
    ) -> None:
        """Tokenize *text* into fixed-length, overlapping examples."""
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)

        for index in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[index : index + max_length]
            target_chunk = token_ids[index + 1 : index + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """Return the number of input-target sequence pairs."""
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the input and next-token target sequence at *index*."""
        return self.input_ids[index], self.target_ids[index]
