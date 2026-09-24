"""
Word-level tokenizer.

The WordTokenizerV1 class is derived from the examples in Sebastian Raschka's
"Build a Large Language Model (From Scratch)":
https://sebastianraschka.com/llms-from-scratch/
"""

import re


class WordTokenizerV1:
    """
    Encode and decode text using a fixed word-level vocabulary.

    - Vocabulary: a fixed dict mapping each unique token (word or punctuation
      mark) to an integer id, e.g. ``{"!": 0, ",": 1, "hello": 2, "world": 3}``.
      Unknown tokens raise a ``KeyError``.
    - ``encode``: split text into tokens, then look up each id.
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
    changes it. Encoding a word that is not in the vocabulary raises a
    ``KeyError``.
    """

    def __init__(self, vocab: dict[str, int]):
        """Store the vocabulary and build the reverse id-to-token mapping."""
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    def encode(self, text: str) -> list[int]:
        """Convert a str into a list of integer tokens."""
        pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        tokens: list[str] = []
        for piece in pieces:
            piece = piece.strip()
            if piece:
                tokens.append(piece)

        ids: list[int] = []
        for token in tokens:
            ids.append(self.str_to_int[token])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert an integer token list back into a str."""
        tokens: list[str] = []
        for token_id in ids:
            tokens.append(self.int_to_str[token_id])

        text = " ".join(tokens)

        # Remove the space that join() put before punctuation.
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text