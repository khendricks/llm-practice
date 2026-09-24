from src.tokenizers.byte_pair_v1 import BytePairTokenizerV1

def test_encode_returns_integer_tokens():
    tokenizer = BytePairTokenizerV1()
    assert tokenizer.encode("hello, world!") == [31373, 11, 995, 0]

def test_decode_returns_original_string():
    tokenizer = BytePairTokenizerV1()
    assert tokenizer.decode([31373, 11, 995, 0]) == "hello, world!"
