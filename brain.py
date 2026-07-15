"""
inference.py
------------
Loads a trained InnieModel + Tokenizer and generates text.

Generation strategy: simple temperature-based sampling, one token at a
time, feeding each new token back in as part of the context window. This
is the same basic loop that larger autoregressive language models use --
INNIE AI just uses a much smaller network under the hood for now.
"""

import os
import numpy as np

from config import (
    CONTEXT_WINDOW,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    WEIGHTS_PATH,
    VOCAB_PATH,
)
from tokenizer import Tokenizer
from model import InnieModel


class InnieInference:
    """Wraps a trained model + tokenizer for easy text generation."""

    def __init__(self, weights_path: str = WEIGHTS_PATH, vocab_path: str = VOCAB_PATH):
        self.tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)

        if os.path.exists(vocab_path):
            self.tokenizer.load(vocab_path)
        else:
            raise FileNotFoundError(
                f"No vocab found at {vocab_path}. Run trainer.py first to train a model."
            )

        self.model = InnieModel(
            vocab_size=len(self.tokenizer),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
        )

        if os.path.exists(weights_path):
            self.model.load(weights_path)
        else:
            raise FileNotFoundError(
                f"No trained weights found at {weights_path}. Run trainer.py first."
            )

    def _sample(self, probs: np.ndarray, temperature: float) -> int:
        """Sample a token id from a probability distribution with temperature scaling."""
        if temperature <= 0:
            return int(np.argmax(probs))  # greedy decoding

        # Apply temperature: lower = more confident/deterministic, higher = more random
        logits = np.log(probs + 1e-9) / temperature
        scaled_probs = np.exp(logits) / np.sum(np.exp(logits))
        return int(np.random.choice(len(scaled_probs), p=scaled_probs))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 0.8,
    ) -> str:
        """
        Generate a continuation for `prompt`.

        Args:
            prompt: the input text to continue from
            max_new_tokens: how many tokens to generate
            temperature: sampling temperature (0 = greedy/deterministic)
        Returns:
            The generated text (prompt not included).
        """
        context_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        eos_id = self.tokenizer.token_to_id.get("<EOS>")

        generated_ids = []
        for _ in range(max_new_tokens):
            # Only look at the most recent CONTEXT_WINDOW tokens
            window = context_ids[-CONTEXT_WINDOW:]
            probs = self.model.forward(window)
            next_id = self._sample(probs, temperature)

            if next_id == eos_id:
                break

            generated_ids.append(next_id)
            context_ids.append(next_id)

        return self.tokenizer.decode(generated_ids)


if __name__ == "__main__":
    # Quick manual smoke test: `python inference.py`
    # (Requires trainer.py to have been run first.)
    engine = InnieInference()
    response = engine.generate("Hello INNIE", max_new_tokens=20)
    print("Generated:", response)
