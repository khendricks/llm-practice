"""Chat with the pretrained SmolLM2 instruction model.

The model weights and tokenizer are downloaded from Hugging Face the first
time the class is created. Later instances reuse the local Hugging Face cache.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


class SmolLM2ChatV1:
    """Generate deterministic chat replies with pretrained SmolLM2 weights."""

    def __init__(self) -> None:
        """Load the model and tokenizer from Hugging Face or its local cache."""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        self.model.eval()

    def generate_reply(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Return assistant text generated for one user prompt.

        Args:
            prompt: The user's chat message.
            max_new_tokens: Maximum number of assistant tokens to generate.

        """
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_length = inputs["input_ids"].shape[1]
        output_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        reply_ids = output_ids[0, input_length:]

        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
