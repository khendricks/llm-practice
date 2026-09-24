"""
Byte-pair encoding tokenizer, backed by tiktoken.

BytePairTokenizerV1 wraps OpenAI's tiktoken library rather than
implementing byte-pair encoding from scratch, using its pretrained
"gpt2" encoding: a fixed merge table and vocabulary trained on GPT-2's
corpus, so encoding/decoding here doesn't depend on any of this
project's own text or vocab-building code.
"""

import tiktoken


class BytePairTokenizerV1:
    """Encode and decode text using tiktoken's pretrained GPT-2 encoding."""

    def __init__(self):
        """Load tiktoken's pretrained "gpt2" encoding."""
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list[int]:
        """Convert a str into a list of integer tokens."""
        return self.tokenizer.encode(
            text, allowed_special={"<|endoftext|>"}
        )

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer tokens back into a str."""
        return self.tokenizer.decode(ids)
