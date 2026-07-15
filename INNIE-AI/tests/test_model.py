"""
test_model.py
-------------
Unit tests for backend/model.py and backend/embeddings.py

Run with:  pytest tests/  (from the project root)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from embeddings import EmbeddingLayer  # noqa: E402
from model import InnieModel, softmax  # noqa: E402


def test_softmax_sums_to_one():
    logits = np.array([1.0, 2.0, 3.0])
    probs = softmax(logits)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0)


def test_embedding_forward_shape():
    emb = EmbeddingLayer(vocab_size=20, embedding_dim=8)
    output = emb.forward([1, 2, 3])
    assert output.shape == (3, 8)


def test_model_forward_output_is_probability_distribution():
    model = InnieModel(vocab_size=20, embedding_dim=8, hidden_dim=16)
    probs = model.forward([1, 2, 3])
    assert probs.shape == (20,)
    assert np.isclose(probs.sum(), 1.0)


def test_model_loss_decreases_after_training_steps():
    """A tiny sanity check: repeated training on the same example should
    reduce the loss for that example (proves gradients are flowing)."""
    model = InnieModel(vocab_size=15, embedding_dim=8, hidden_dim=16)
    context = [1, 2, 3]
    target = 7

    model.forward(context)
    first_loss = model.backward(target, learning_rate=0.1)

    for _ in range(20):
        model.forward(context)
        loss = model.backward(target, learning_rate=0.1)

    assert loss < first_loss


def test_model_save_and_load(tmp_path):
    model = InnieModel(vocab_size=15, embedding_dim=8, hidden_dim=16)
    model.forward([1, 2, 3])
    model.backward(5, learning_rate=0.1)

    save_path = str(tmp_path / "weights.npz")
    model.save(save_path)

    reloaded = InnieModel(vocab_size=15, embedding_dim=8, hidden_dim=16)
    reloaded.load(save_path)

    np.testing.assert_array_equal(reloaded.W1, model.W1)
    np.testing.assert_array_equal(reloaded.embedding.weights, model.embedding.weights)
