"""
trainer.py
----------
The training pipeline for INNIE AI.

Takes raw text files from datasets/, builds a tokenizer vocabulary, turns
the text into (context, target) next-token-prediction pairs using a
sliding window, and trains InnieModel on them with plain SGD.

Run directly with:  python trainer.py
"""

import glob
import os

from config import (
    DATASETS_DIR,
    CONTEXT_WINDOW,
    LEARNING_RATE,
    EPOCHS,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    WEIGHTS_PATH,
    VOCAB_PATH,
    ensure_directories,
)
from tokenizer import Tokenizer
from model import InnieModel


def load_dataset_texts(datasets_dir: str = DATASETS_DIR) -> list[str]:
    """Read every .txt file under datasets/ (recursively) into a list of strings."""
    pattern = os.path.join(datasets_dir, "**", "*.txt")
    texts = []
    for filepath in glob.glob(pattern, recursive=True):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                texts.append(content)
    return texts


def build_training_pairs(token_ids: list[int], context_window: int) -> list[tuple[list[int], int]]:
    """
    Slide a window over a token sequence to build (context, next_token) pairs.

    Example with context_window=3 and tokens [A, B, C, D, E]:
        ([A, B, C], D)
        ([B, C, D], E)
    """
    pairs = []
    for i in range(len(token_ids) - context_window):
        context = token_ids[i: i + context_window]
        target = token_ids[i + context_window]
        pairs.append((context, target))
    return pairs


def train(
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    context_window: int = CONTEXT_WINDOW,
    verbose: bool = True,
) -> tuple[InnieModel, Tokenizer]:
    """Run the full training pipeline and save the resulting model + vocab."""
    ensure_directories()

    texts = load_dataset_texts()
    if not texts:
        # Fall back to a tiny built-in sample so the pipeline always runs,
        # even before any real dataset has been added.
        texts = [
            "INNIE AI is a modular artificial intelligence system built from scratch. "
            "It learns to predict the next word in a sentence by studying examples. "
            "Over time it will grow from a small nano network into a much larger model."
        ]
        if verbose:
            print("No .txt files found in datasets/, using built-in sample text instead.")

    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(texts)
    tokenizer.save(VOCAB_PATH)

    model = InnieModel(
        vocab_size=len(tokenizer),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    # Build training pairs across all documents
    all_pairs = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        all_pairs.extend(build_training_pairs(ids, context_window))

    if not all_pairs:
        raise ValueError(
            "Not enough text to build a single training example. "
            "Add longer documents to datasets/, or lower CONTEXT_WINDOW in config.py."
        )

    if verbose:
        print(f"Loaded {len(texts)} document(s), {len(all_pairs)} training pair(s), "
              f"vocab size {len(tokenizer)}")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for context, target in all_pairs:
            model.forward(context)
            loss = model.backward(target, learning_rate)
            total_loss += loss

        avg_loss = total_loss / len(all_pairs)
        if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs):
            print(f"Epoch {epoch:>4}/{epochs} | avg loss: {avg_loss:.4f}")

    model.save(WEIGHTS_PATH)
    if verbose:
        print(f"Training complete. Weights saved to {WEIGHTS_PATH}, vocab saved to {VOCAB_PATH}")

    return model, tokenizer


if __name__ == "__main__":
    train()
