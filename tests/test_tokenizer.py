"""
test_tokenizer.py
------------------
Unit tests for backend/tokenizer.py

Run with:  pytest tests/  (from the project root, with backend/ on PYTHONPATH)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tokenizer import Tokenizer  # noqa: E402


SAMPLE_TEXTS = [
    "Hello there, how are you today?",
    "INNIE AI is learning to understand language, one token at a time.",
]


def test_vocab_contains_special_tokens():
    tok = Tokenizer(vocab_size=50)
    tok.build_vocab(SAMPLE_TEXTS)
    for special in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
        assert special in tok.token_to_id


def test_encode_decode_roundtrip():
    tok = Tokenizer(vocab_size=50)
    tok.build_vocab(SAMPLE_TEXTS)

    encoded = tok.encode("Hello there")
    decoded = tok.decode(encoded)

    assert "hello" in decoded.lower()
    assert "there" in decoded.lower()


def test_unknown_words_map_to_unk():
    tok = Tokenizer(vocab_size=50)
    tok.build_vocab(SAMPLE_TEXTS)

    encoded = tok.encode("zzzznotarealword", add_special_tokens=False)
    unk_id = tok.token_to_id["<UNK>"]
    assert unk_id in encoded


def test_save_and_load(tmp_path):
    tok = Tokenizer(vocab_size=50)
    tok.build_vocab(SAMPLE_TEXTS)

    save_path = tmp_path / "vocab.json"
    tok.save(str(save_path))

    reloaded = Tokenizer(vocab_size=50)
    reloaded.load(str(save_path))

    assert reloaded.token_to_id == tok.token_to_id
