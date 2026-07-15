"""
embeddings.py
-------------
A minimal, from-scratch embedding layer built on numpy.

An embedding layer is just a lookup table: each token id maps to a vector
of learnable numbers. Those vectors are what the rest of the network
reasons about. This class also implements its own forward/backward pass so
it can be trained without any deep learning framework -- keeping INNIE AI
dependency-light while still being "real" gradient-based learning.
"""

import numpy as np

from config import EMBEDDING_DIM, VOCAB_SIZE, SEED


class EmbeddingLayer:
    """Learnable token_id -> vector lookup table."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, embedding_dim: int = EMBEDDING_DIM):
        rng = np.random.default_rng(SEED)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Small random initialization (standard practice: keeps early
        # activations from exploding or vanishing).
        self.weights = rng.normal(0, 0.02, size=(vocab_size, embedding_dim))

        # Cache for the backward pass
        self._last_ids: np.ndarray | None = None
        self.grad_weights = np.zeros_like(self.weights)

    def forward(self, token_ids: list[int]) -> np.ndarray:
        """
        Look up embeddings for a sequence of token ids.

        Args:
            token_ids: list of ints, length = sequence length
        Returns:
            array of shape (sequence_length, embedding_dim)
        """
        ids = np.array(token_ids, dtype=int)
        ids = np.clip(ids, 0, self.vocab_size - 1)  # guard against out-of-range ids
        self._last_ids = ids
        return self.weights[ids]

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> None:
        """
        Apply gradients computed by later layers back onto the embedding table.

        Args:
            grad_output: gradient w.r.t. this layer's output, shape (seq_len, embedding_dim)
            learning_rate: step size for the update
        """
        if self._last_ids is None:
            raise RuntimeError("backward() called before forward()")

        self.grad_weights.fill(0)
        # Accumulate gradients for each token id that appeared in the sequence
        np.add.at(self.grad_weights, self._last_ids, grad_output)

        # Simple SGD update
        self.weights -= learning_rate * self.grad_weights

    def save(self, path: str) -> None:
        np.save(path, self.weights)

    def load(self, path: str) -> None:
        self.weights = np.load(path)


if __name__ == "__main__":
    # Quick manual smoke test: `python embeddings.py`
    emb = EmbeddingLayer(vocab_size=50, embedding_dim=8)
    vectors = emb.forward([1, 4, 7, 2])
    print("Embedding output shape:", vectors.shape)
