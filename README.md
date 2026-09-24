# Practice LLM

A small character-level tokenizer and language-model learning project.

## Architecture

Currently this project has tokenizers only; the training/model pipeline
(the Transformer, attention, batching, `train.py`, `chat.py`) is being
rebuilt in phases and isn't wired up yet.

- **`CharacterTokenizer`** (`tokenizers/character.py`) — maps each
  character in a piece of text to an integer ID and back.
- **`WordTokenizerV1`** / **`WordTokenizerV2`**
  (`tokenizers/word_v1.py`, `tokenizers/word_v2.py`) — map whole words
  and punctuation marks to integer IDs and back, using a fixed
  vocabulary. `V2` additionally maps any word outside that vocabulary to
  an `"<|unk|>"` placeholder instead of raising an error.
- **`BytePairTokenizerV1`** (`tokenizers/byte_pair_v1.py`) — wraps
  `tiktoken`'s pretrained GPT-2 byte-pair encoding.

## Setup

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Python and the project dependencies:

```bash
uv python install 3.12
uv sync
```

`uv sync` creates or updates `.venv` from `pyproject.toml` and `uv.lock`, so you do not need to activate the environment manually.

## Tests

```bash
uv run python -m pytest -v
```
