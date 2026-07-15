"""
tokenizer.py
------------
A simple, dependency-free word-level tokenizer for INNIE AI.

Why word-level (and not full BPE) for v0.0.1?
    A subword/BPE tokenizer is the right choice for a production LLM, but it
    adds real complexity. Starting with a word-level tokenizer keeps the
    whole pipeline (tokenizer -> embeddings -> model -> inference)
    understandable end-to-end. The interface below (encode/decode/
    build_vocab/save/load) is written so a future BPE tokenizer can be
    swapped in without touching any other file.
"""

import json
import os
import re
from collections import Counter

from config import (
    VOCAB_SIZE,
    VOCAB_PATH,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    SPECIAL_TOKENS,
)

# A simple regex that splits on words and punctuation, keeping punctuation
# as its own token (e.g. "Hello, world!" -> ["Hello", ",", "world", "!"])
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


class Tokenizer:
    """Converts raw text <-> lists of integer token ids."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size

        # token -> id and id -> token lookup tables
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        # Register special tokens first so their ids are stable (0, 1, 2, 3...)
        for token in SPECIAL_TOKENS:
            self._add_token(token)

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------
    def _add_token(self, token: str) -> int:
        """Add a token to the vocabulary if it isn't already present."""
        if token not in self.token_to_id:
            new_id = len(self.token_to_id)
            self.token_to_id[token] = new_id
            self.id_to_token[new_id] = token
        return self.token_to_id[token]

    def _pre_tokenize(self, text: str) -> list[str]:
        """Split raw text into rough word/punctuation tokens."""
        return _TOKEN_PATTERN.findall(text.lower())

    def build_vocab(self, texts: list[str]) -> None:
        """
        Build the vocabulary from a list of training documents.

        The most frequent `vocab_size` words become tokens; everything else
        collapses into UNK_TOKEN at encode time.
        """
        counter: Counter = Counter()
        for text in texts:
            counter.update(self._pre_tokenize(text))

        most_common = counter.most_common(self.vocab_size - len(SPECIAL_TOKENS))
        for token, _freq in most_common:
            self._add_token(token)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Turn a string into a list of token ids."""
        words = self._pre_tokenize(text)
        unk_id = self.token_to_id[UNK_TOKEN]
        ids = [self.token_to_id.get(w, unk_id) for w in words]

        if add_special_tokens:
            ids = [self.token_to_id[BOS_TOKEN]] + ids + [self.token_to_id[EOS_TOKEN]]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Turn a list of token ids back into a readable string."""
        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, UNK_TOKEN)
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)

        # Join words with spaces, but avoid a space before punctuation
        text = ""
        for token in tokens:
            if token in {",", ".", "!", "?", ";", ":"} and text:
                text = text.rstrip() + token + " "
            else:
                text += token + " "
        return text.strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str = VOCAB_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    def load(self, path: str = VOCAB_PATH) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)


if __name__ == "__main__":
    # Quick manual smoke test: `python tokenizer.py`
    sample_texts = [
        "Hello there, how are you today?",
        "INNIE AI is learning to understand language, one token at a time.",
    ]
    tok = Tokenizer(vocab_size=100)
    tok.build_vocab(sample_texts)

    encoded = tok.encode("Hello, INNIE AI!")
    decoded = tok.decode(encoded)

    print("Vocab size:", len(tok))
    print("Encoded:", encoded)
    print("Decoded:", decoded)
