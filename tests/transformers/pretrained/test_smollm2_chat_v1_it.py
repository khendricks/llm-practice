"""Integration test for a pretrained SmolLM2 chat model."""

from src.transformers.pretrained.smollm2_chat_v1 import SmolLM2ChatV1


def test_generate_reply_returns_assistant_text() -> None:
    """A user prompt produces non-empty assistant text."""
    chat = SmolLM2ChatV1()

    reply = chat.generate_reply("Name the capital of France.", max_new_tokens=16)

    assert isinstance(reply, str)
    assert reply.strip()
