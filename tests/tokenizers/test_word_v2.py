import re

from src.tokenizers.word_v2 import WordTokenizerV2

def build_vocab(text: str) -> dict[str, int]:
    """Split text into words and punctuation, then map each unique token to an id."""
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [token.strip() for token in tokens if token.strip()]
    return {token: i for i, token in enumerate(sorted(set(tokens)))}

def test_encode_returns_integer_tokens():
    # sorted vocab: {"!": 0, ",": 1, "hello": 2, "world": 3}
    vocab = build_vocab("hello, world!")
    tokenizer = WordTokenizerV2(vocab)
    assert tokenizer.encode("hello, world!") == [2, 1, 3, 0]

def test_decode_returns_original_string():
    vocab = build_vocab("hello, world!")
    tokenizer = WordTokenizerV2(vocab)
    assert tokenizer.decode([2, 1, 3, 0]) == "hello, world!"

def test_encode_maps_unknown_token_to_unk_id():
    # sorted vocab: {"!": 0, ",": 1, "<|unk|>": 2, "hello": 3, "world": 4}
    vocab = build_vocab("hello, world! <|unk|>")
    tokenizer = WordTokenizerV2(vocab)
    assert tokenizer.encode("hello, there!") == [3, 1, 2, 0]
