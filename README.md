# Practice LLM

An evolving, from-scratch language-model learning project. It follows the
ideas and progression in Sebastian Raschka's
[*Build a Large Language Model (From Scratch)*](https://www.amazon.com/Sebastian-Raschka/e/B00J1DHHFS/ref=dp_byline_cont_book_1),
while deliberately adapting the implementation as I form my own opinions.

## Architecture so far

This is a working notebook of a model build, not a commitment to one final
architecture. Components are introduced in small, versioned steps, tested,
and may be replaced as the project develops.

The implemented path is currently:

```text
text -> tokenizer -> token IDs -> DatasetV1 -> DataLoaderV1
     -> InputEmbeddingV1 (token embeddings + absolute-position embeddings)
```

- **Tokenization experiments** (`src/tokenizers/`) include character- and
  word-level tokenizers plus `BytePairTokenizerV1`, a wrapper around
  `tiktoken`'s pretrained GPT-2 byte-pair encoding.
- **Next-token training data** (`src/datasets/dataset_v1.py`) creates
  overlapping input and target sequences, where each target sequence is
  shifted ahead by one token.
- **Batched examples** (`src/dataloaders/dataloader_v1.py`) collect those
  sequences for PyTorch training. The default sequence length is shared
  with the embedding layer through `src/config.py`.
- **Input embeddings** (`src/input_embedding/input_embedding_v1.py`) add a
  learned token vector to a learned absolute-position vector for every
  token in a batch.

Attention, Transformer blocks, training, and generation are future stages,
not fixed design decisions. Their interfaces and implementation choices will
evolve with the lessons from the book and this project's own experiments.

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
