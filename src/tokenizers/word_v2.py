"""
Word-level tokenizer.

The WordTokenizerV2 class is derived from the examples in Sebastian Raschka's
"Build a Large Language Model (From Scratch)":
https://sebastianraschka.com/llms-from-scratch/
"""

import re


class WordTokenizerV2:
    """
    Encode and decode text using a fixed word-level vocabulary.

    - Vocabulary: a fixed dict mapping each unique token (word or punctuation
      mark) to an integer id, e.g. ``{"!": 0, ",": 1, "hello": 2, "world": 3}``.
      Tokens not in the vocabulary are mapped to ``"<|unk|>"``.
    - ``encode``: split text into tokens, replace unknown tokens with
      ``"<|unk|>"``, then look up each id.
    - ``decode``: look up each token, join with spaces, fix punctuation spacing.

    Example with vocab ``{"!": 0, ",": 1, "hello": 2, "world": 3}``::

        encode
        ------
        "hello, world!"
              |  re.split on punctuation / whitespace
              v
        ["hello", ",", "", " ", "world", "!", ""]
              |  strip, drop empty pieces
              v
        ["hello", ",", "world", "!"]
              |  str_to_int lookup
              v
        [2, 1, 3, 0]

        decode
        ------
        [2, 1, 3, 0]
              |  int_to_str lookup
              v
        ["hello", ",", "world", "!"]
              |  " ".join
              v
        "hello , world !"
              |  re.sub removes space before punctuation
              v
        "hello, world!"

    The vocabulary is built elsewhere and passed in, so the tokenizer never
    changes it. ``"<|unk|>"`` must itself be a token in the vocabulary, or
    encoding an unknown word will raise a ``KeyError``.
    """

    def __init__(self, vocab: dict[str, int]):
        """Store the vocabulary and build the reverse id-to-token mapping."""
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text: str) -> list[int]:
        """Convert a str into a list of integer tokens."""
        # Split on punctuation, "--", and whitespace, keeping the
        # delimiters (they're in a capturing group).
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        # Trim whitespace off each piece and drop any that are now
        # empty (the split leaves blanks between adjacent delimiters).
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        # Replace any token that isn't in the vocab with the
        # "<|unk|>" placeholder so the lookup below can't KeyError.
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]

        # Look up each token's integer id in the vocab.
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer tokens back into a str."""
        # Map each id back to its token string and join with spaces.
        text = " ".join([self.int_to_str[i] for i in ids])

        # Remove the space that ends up before punctuation, e.g.
        # "hello ." -> "hello.".
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text