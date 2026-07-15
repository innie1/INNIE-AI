"""
memory.py
---------
INNIE AI's memory system.

Two tiers, mirroring how the roadmap describes memory:

1. Short-term memory: an in-RAM sliding window of the current
   conversation, used to build context for the model.
2. Long-term memory: a JSON file on disk that persists facts/notes
   across restarts. This is intentionally simple (no vector search yet)
   so it is easy to inspect, edit by hand, and later swap for a real
   embeddings-based memory store without changing the public interface.
"""

import json
import os
import time

from config import SHORT_TERM_MEMORY_LIMIT, LONG_TERM_MEMORY_ENABLED, MEMORY_PATH


class Memory:
    """Manages short-term conversation history and long-term persisted facts."""

    def __init__(self, memory_path: str = MEMORY_PATH):
        self.memory_path = memory_path

        # Short-term: list of {"role": "user"/"assistant", "text": str, "time": float}
        self.short_term: list[dict] = []

        # Long-term: list of persisted strings (facts, notes, summaries)
        self.long_term: list[str] = []

        if LONG_TERM_MEMORY_ENABLED:
            self._load_long_term()

    # ------------------------------------------------------------------
    # Short-term memory
    # ------------------------------------------------------------------
    def add_turn(self, role: str, text: str) -> None:
        """Record one turn of the conversation (role is 'user' or 'assistant')."""
        self.short_term.append({"role": role, "text": text, "time": time.time()})

        # Keep only the most recent N turns to bound memory/context size
        if len(self.short_term) > SHORT_TERM_MEMORY_LIMIT:
            self.short_term = self.short_term[-SHORT_TERM_MEMORY_LIMIT:]

    def get_recent_context(self, n: int = 5) -> str:
        """Return the last `n` turns of conversation as a single text blob."""
        recent = self.short_term[-n:]
        lines = [f"{turn['role']}: {turn['text']}" for turn in recent]
        return "\n".join(lines)

    def clear_short_term(self) -> None:
        self.short_term = []

    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------
    def remember(self, fact: str) -> None:
        """Persist a fact/note to long-term memory."""
        if fact not in self.long_term:
            self.long_term.append(fact)
            self._save_long_term()

    def forget(self, fact: str) -> bool:
        """Remove a fact from long-term memory. Returns True if it was found."""
        if fact in self.long_term:
            self.long_term.remove(fact)
            self._save_long_term()
            return True
        return False

    def recall_all(self) -> list[str]:
        return list(self.long_term)

    def search(self, keyword: str) -> list[str]:
        """Very simple substring search over long-term memory."""
        keyword_lower = keyword.lower()
        return [f for f in self.long_term if keyword_lower in f.lower()]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_long_term(self) -> None:
        directory = os.path.dirname(self.memory_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.long_term, f, ensure_ascii=False, indent=2)

    def _load_long_term(self) -> None:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                self.long_term = json.load(f)


if __name__ == "__main__":
    # Quick manual smoke test: `python memory.py`
    mem = Memory(memory_path="memory_test.json")
    mem.add_turn("user", "Hello INNIE")
    mem.add_turn("assistant", "Hi Innocent, how can I help?")
    mem.remember("User's company is INNIE Group.")

    print("Recent context:\n", mem.get_recent_context())
    print("Long-term memory:", mem.recall_all())

    os.remove("memory_test.json")
