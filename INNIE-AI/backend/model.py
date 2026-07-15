"""
model.py
--------
The core neural network for INNIE AI v0.0.1.

Architecture (deliberately simple, deliberately extensible):

    token ids -> EmbeddingLayer -> mean-pool over context window
              -> Dense(hidden) + ReLU
              -> Dense(vocab_size) -> softmax
              -> probability distribution over the *next* token

This is a "context averaging" feedforward network, not yet a transformer.
It is the nano-network stage of the roadmap: it proves out the full
tokenizer -> embed -> predict -> sample loop with real gradients, in code
simple enough to read top to bottom. The natural next step (already
scaffolded via CONTEXT_WINDOW and the modular layer design) is to replace
the mean-pool + dense block with self-attention to get a nano-transformer,
without touching tokenizer.py, embeddings.py, memory.py, or api.py at all.
"""

import numpy as np

from config import EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, SEED
from embeddings import EmbeddingLayer


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class InnieModel:
    """A small feedforward network that predicts the next token."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        rng = np.random.default_rng(SEED)

        self.vocab_size = vocab_size
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim)

        # He-style initialization scaled for ReLU activations
        self.W1 = rng.normal(0, np.sqrt(2 / embedding_dim), size=(embedding_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        self.W2 = rng.normal(0, np.sqrt(2 / hidden_dim), size=(hidden_dim, vocab_size))
        self.b2 = np.zeros(vocab_size)

        # Caches populated during forward(), used during backward()
        self._cache = {}

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, token_ids: list[int]) -> np.ndarray:
        """
        Run one forward pass.

        Args:
            token_ids: the context tokens (already truncated to CONTEXT_WINDOW)
        Returns:
            probs: shape (vocab_size,) -- probability distribution for the
                   token that should come next.
        """
        embedded = self.embedding.forward(token_ids)          # (seq_len, embed_dim)
        context_vector = embedded.mean(axis=0)                # (embed_dim,) mean-pool

        z1 = context_vector @ self.W1 + self.b1                # (hidden_dim,)
        a1 = np.maximum(0, z1)                                  # ReLU

        z2 = a1 @ self.W2 + self.b2                             # (vocab_size,)
        probs = softmax(z2)

        # Save everything backward() will need
        self._cache = {
            "token_ids": token_ids,
            "embedded": embedded,
            "context_vector": context_vector,
            "z1": z1,
            "a1": a1,
            "probs": probs,
        }
        return probs

    # ------------------------------------------------------------------
    # Backward pass (manual backpropagation, no autograd framework)
    # ------------------------------------------------------------------
    def backward(self, target_id: int, learning_rate: float) -> float:
        """
        Compute cross-entropy loss against `target_id` and update all
        weights via SGD.

        Returns:
            loss (float) for logging/monitoring.
        """
        cache = self._cache
        probs = cache["probs"]
        a1 = cache["a1"]
        z1 = cache["z1"]
        context_vector = cache["context_vector"]

        # Cross-entropy loss: -log(p_target)
        loss = -np.log(probs[target_id] + 1e-9)

        # dL/dz2 for softmax + cross-entropy simplifies to (probs - one_hot)
        one_hot = np.zeros(self.vocab_size)
        one_hot[target_id] = 1.0
        d_z2 = probs - one_hot                                  # (vocab_size,)

        # Gradients for output layer
        d_W2 = np.outer(a1, d_z2)
        d_b2 = d_z2

        # Backprop into hidden layer
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (z1 > 0)                                  # ReLU derivative

        # Gradients for input layer
        d_W1 = np.outer(context_vector, d_z1)
        d_b1 = d_z1

        # Gradient flowing back into the (mean-pooled) embedding
        d_context = d_z1 @ self.W1.T
        seq_len = len(cache["token_ids"])
        d_embedded = np.tile(d_context / seq_len, (seq_len, 1))

        # Apply updates
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.embedding.backward(d_embedded, learning_rate)

        return float(loss)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(
            path,
            embedding=self.embedding.weights,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        self.embedding.weights = data["embedding"]
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]


if __name__ == "__main__":
    # Quick manual smoke test: `python model.py`
    model = InnieModel(vocab_size=30, embedding_dim=8, hidden_dim=16)
    probs = model.forward([1, 2, 3])
    print("Output probability shape:", probs.shape)
    loss = model.backward(target_id=5, learning_rate=0.1)
    print("Loss after one training step:", loss)
