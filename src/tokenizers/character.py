class CharacterTokenizer:
    """
    Build a character-level vocabulary and encode/decode complete strings.

    This tokenizer treats every individual character as one token. For example,
    ``"hello"`` becomes five character tokens. Character-level tokenization is
    simple and can represent any character in its vocabulary, but it produces
    longer sequences than word- or subword-level tokenizers.

    Other common approaches tokenize whole words, or subwords (such as BPE,
    WordPiece, and Unigram/SentencePiece). Most modern language models use
    subword tokenizers, which balance vocabulary size against sequence length.
    """

    def __init__(self, text: str):
        """Create token mappings from the unique characters in text."""
        self.chars = sorted(list(set(text)))

        self.stoi = self._str_to_integer(self.chars)
        self.itos = self._integer_to_str(self.chars)

    def _str_to_integer(self, chars: list[str]) -> dict:
        """Map each character to its integer token."""
        result: dict = {}
        for i, ch in enumerate(chars):
            result[ch] = i

        return result

    def _integer_to_str(self, chars: list[str]) -> dict:
        """Map each integer token to its character."""
        result: dict = {}
        for i, ch in enumerate(chars):
            result[i] = ch

        return result

    def encode(self, text: str) -> list[int]:
        """Convert a str into a list of integer tokens."""
        result: list[int] = []
        for char in text:
            result.append(self.stoi[char])
        return result

    def decode(self, tokens: list[int]) -> str:
        """Convert an integer token list back into a str."""
        result: list[str] = []
        for token in tokens:
            result.append(self.itos[token])
        return "".join(result)

    def get_vocab_size(self) -> int:
        """Return the number of unique character tokens."""
        return len(self.chars)
