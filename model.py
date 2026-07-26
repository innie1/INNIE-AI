"""
model.py
--------
The core neural network for INNIE AI.

Architecture (v0.1.0 — nano-transformer):

    token ids -> EmbeddingLayer + PositionalEmbedding
              -> Causal Self-Attention (single head)
              -> residual add
              -> Dense(hidden) + ReLU   (applied to the last position)
              -> Dense(vocab_size) -> softmax
              -> probability distribution over the *next* token

This replaces the v0.0.1 "mean-pool" block (which threw away word order
entirely — "the cat sat" and "sat cat the" produced identical input) with
real self-attention, so the model can actually use word order and which
words attend to which other words. tokenizer.py, embeddings.py, memory.py,
and api.py are untouched — this file still exposes the same
forward()/backward()/save()/load() interface trainer.py and inference.py
already call.

Everything below is hand-written NumPy (forward + manual backprop), no
autograd framework, matching the rest of the codebase's "transparent, no
hidden magic" philosophy.
"""

import os
import numpy as np

from config import EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, CONTEXT_WINDOW, SEED
from embeddings import EmbeddingLayer


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class InnieModel:
    """A small causal self-attention network that predicts the next token."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        context_window: int = CONTEXT_WINDOW,
    ):
        rng = np.random.default_rng(SEED)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim)

        # Learnable positional embeddings: one vector per position in the
        # context window. This is what lets the model tell "cat sat" apart
        # from "sat cat" -- the mean-pool version could not.
        self.pos_embedding = rng.normal(0, 0.02, size=(context_window, embedding_dim)).astype(np.float32)

        # Self-attention projections (single head, Q/K/V/output)
        d = embedding_dim
        attn_scale = np.sqrt(2 / d)
        self.Wq = rng.normal(0, attn_scale, size=(d, d)).astype(np.float32)
        self.Wk = rng.normal(0, attn_scale, size=(d, d)).astype(np.float32)
        self.Wv = rng.normal(0, attn_scale, size=(d, d)).astype(np.float32)
        self.Wo = rng.normal(0, attn_scale, size=(d, d)).astype(np.float32)

        # Feedforward head (applied to the last position only, same as the
        # v0.0.1 dense block so checkpoints/config stay compatible in shape)
        self.W1 = rng.normal(0, np.sqrt(2 / embedding_dim), size=(embedding_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = rng.normal(0, np.sqrt(2 / hidden_dim), size=(hidden_dim, vocab_size)).astype(np.float32)
        self.b2 = np.zeros(vocab_size, dtype=np.float32)

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
        seq_len = len(token_ids)
        d = self.embedding_dim

        token_vecs = self.embedding.forward(token_ids)          # (seq_len, d)
        pos_vecs = self.pos_embedding[:seq_len]                  # (seq_len, d)
        X = token_vecs + pos_vecs                                 # (seq_len, d)

        # --- Causal self-attention ---
        Q = X @ self.Wq                                           # (seq_len, d)
        K = X @ self.Wk                                           # (seq_len, d)
        V = X @ self.Wv                                           # (seq_len, d)

        scores = (Q @ K.T) / np.sqrt(d)                           # (seq_len, seq_len)
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(causal_mask, -1e9, scores)

        A = softmax(scores)                                       # (seq_len, seq_len)
        attn_out = A @ V                                          # (seq_len, d)
        O = attn_out @ self.Wo                                    # (seq_len, d)

        H = X + O                                                 # residual connection

        # --- Predict next token from the last position's representation ---
        # (causal attention means H[-1] already carries context from every
        # earlier token in the window, in the correct order)
        h_last = H[-1]                                            # (d,)

        z1 = h_last @ self.W1 + self.b1
        a1 = np.maximum(0, z1)                                    # ReLU
        z2 = a1 @ self.W2 + self.b2
        probs = softmax(z2)

        self._cache = {
            "token_ids": token_ids,
            "seq_len": seq_len,
            "X": X, "Q": Q, "K": K, "V": V,
            "A": A, "attn_out": attn_out, "H": H, "h_last": h_last,
            "z1": z1, "a1": a1, "probs": probs,
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
        c = self._cache
        probs, a1, z1 = c["probs"], c["a1"], c["z1"]
        h_last, H, X = c["h_last"], c["H"], c["X"]
        Q, K, V, A = c["Q"], c["K"], c["V"], c["A"]
        seq_len, d = c["seq_len"], self.embedding_dim

        loss = -np.log(probs[target_id] + 1e-9)

        # --- FFN head ---
        one_hot = np.zeros(self.vocab_size)
        one_hot[target_id] = 1.0
        d_z2 = probs - one_hot                                    # (vocab_size,)

        d_W2 = np.outer(a1, d_z2)
        d_b2 = d_z2

        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (z1 > 0)

        d_W1 = np.outer(h_last, d_z1)
        d_b1 = d_z1

        d_h_last = d_z1 @ self.W1.T                               # (d,)

        # Only the last position received direct FFN gradient; every
        # other position's gradient (for now) comes purely from attention.
        d_H = np.zeros((seq_len, d))
        d_H[-1] = d_h_last

        # --- Residual: H = X + O ---
        d_X_residual = d_H.copy()
        d_O = d_H

        # --- O = attn_out @ Wo ---
        d_Wo = c["attn_out"].T @ d_O
        d_attn_out = d_O @ self.Wo.T

        # --- attn_out = A @ V ---
        d_A = d_attn_out @ V.T                                    # (seq_len, seq_len)
        d_V = A.T @ d_attn_out                                    # (seq_len, d)

        # --- softmax backward (row-wise) ---
        d_scores = A * (d_A - np.sum(d_A * A, axis=-1, keepdims=True))

        # --- scores = (Q @ K.T) / sqrt(d) ---
        d_Q = (d_scores @ K) / np.sqrt(d)
        d_K = (d_scores.T @ Q) / np.sqrt(d)

        # --- Q, K, V = X @ Wq/Wk/Wv ---
        d_Wq = X.T @ d_Q
        d_Wk = X.T @ d_K
        d_Wv = X.T @ d_V

        d_X_attn = d_Q @ self.Wq.T + d_K @ self.Wk.T + d_V @ self.Wv.T

        d_X = d_X_residual + d_X_attn                             # (seq_len, d)

        # --- Gradient clipping ---
        # The attention path is deeper than the old mean-pool network and
        # has no layer normalization. Without clipping, SGD at the
        # learning rate tuned for the old shallow net reliably diverges
        # to NaN within a few thousand steps (verified during testing).
        # Clipping every gradient tensor to a max global norm keeps
        # training stable without changing the learning rate.
        grads = [d_W2, d_b2, d_W1, d_b1, d_Wo, d_Wq, d_Wk, d_Wv, d_X]
        max_norm = 5.0
        global_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        if global_norm > max_norm:
            scale = max_norm / (global_norm + 1e-6)
            d_W2 *= scale; d_b2 *= scale
            d_W1 *= scale; d_b1 *= scale
            d_Wo *= scale; d_Wq *= scale; d_Wk *= scale; d_Wv *= scale
            d_X = d_X * scale

        # Apply updates
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.Wo -= learning_rate * d_Wo
        self.Wq -= learning_rate * d_Wq
        self.Wk -= learning_rate * d_Wk
        self.Wv -= learning_rate * d_Wv

        self.pos_embedding[:seq_len] -= learning_rate * d_X
        self.embedding.backward(d_X, learning_rate)

        return float(loss)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            embedding=self.embedding.weights,
            pos_embedding=self.pos_embedding,
            Wq=self.Wq, Wk=self.Wk, Wv=self.Wv, Wo=self.Wo,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        self.embedding.weights = data["embedding"]
        self.pos_embedding = data["pos_embedding"]
        self.Wq, self.Wk, self.Wv, self.Wo = data["Wq"], data["Wk"], data["Wv"], data["Wo"]
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]


if __name__ == "__main__":
    # Quick manual smoke test: `python model.py`
    model = InnieModel(vocab_size=30, embedding_dim=8, hidden_dim=16, context_window=6)
    probs = model.forward([1, 2, 3])
    print("Output probability shape:", probs.shape)
    loss = model.backward(target_id=5, learning_rate=0.1)
    print("Loss after one training step:", loss)
