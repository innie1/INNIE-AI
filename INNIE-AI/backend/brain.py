"""
brain.py
--------
The orchestrator that ties everything together: tokenizer, model,
inference, and memory. This is the single object api.py talks to --
it is the closest thing INNIE AI has to a "mind".

Responsibilities:
    1. Load (or lazily train) the model + tokenizer.
    2. Keep track of conversation memory.
    3. Turn a user message into a generated reply.

Keeping this orchestration logic in one place means the API layer stays
thin, and the model/tokenizer/memory internals can change independently.
"""

import os

from config import WEIGHTS_PATH, VOCAB_PATH
from memory import Memory


class Brain:
    """High-level interface: Brain.think(user_message) -> reply string."""

    def __init__(self):
        self.memory = Memory()
        self.engine = None  # lazily loaded InnieInference instance
        self._try_load_engine()

    def _try_load_engine(self) -> None:
        """Load a trained model if one exists; otherwise stay in fallback mode."""
        if os.path.exists(WEIGHTS_PATH) and os.path.exists(VOCAB_PATH):
            from inference import InnieInference  # imported lazily to avoid
            self.engine = InnieInference()          # circular-import surprises

    def is_trained(self) -> bool:
        return self.engine is not None

    def think(self, user_message: str, max_new_tokens: int = 30, temperature: float = 0.8) -> str:
        """
        Process one user message and return INNIE AI's reply.

        This also records the exchange into short-term memory so future
        turns in the same session have conversational context available.
        """
        self.memory.add_turn("user", user_message)

        if self.engine is None:
            reply = (
                "I haven't been trained yet, so I can't generate a real response. "
                "Run `python trainer.py` from the backend/ folder first, "
                "then restart the server."
            )
        else:
            reply = self.engine.generate(
                user_message,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            if not reply.strip():
                reply = "(I generated an empty response -- try training on more data.)"

        self.memory.add_turn("assistant", reply)
        return reply

    def remember_fact(self, fact: str) -> None:
        """Expose long-term memory writing to the API layer."""
        self.memory.remember(fact)

    def recall_facts(self) -> list[str]:
        return self.memory.recall_all()


if __name__ == "__main__":
    # Quick manual smoke test: `python brain.py`
    brain = Brain()
    print("Trained:", brain.is_trained())
    print(brain.think("Hello INNIE, who are you?"))
