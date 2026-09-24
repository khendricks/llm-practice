from src.tokenizers.character import CharacterTokenizer

def test_encode_returns_integer_tokens():
    tokenizer = CharacterTokenizer("hello")
    assert tokenizer.encode("hello") == [1, 0, 2, 2, 3]

def test_decode_returns_original_string():
    tokenizer = CharacterTokenizer("hello")
    assert tokenizer.decode([1, 0, 2, 2, 3]) == "hello"
