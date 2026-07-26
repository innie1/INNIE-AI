"""
config.py
---------
Central configuration for INNIE AI.
"""

import os

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == "backend":
    BASE_DIR = os.path.dirname(CURRENT_DIR)
else:
    BASE_DIR = CURRENT_DIR

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_DIR = CHECKPOINTS_DIR  # Alias for performance tooling
TESTS_DIR = os.path.join(BASE_DIR, "tests")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Performance logging path
PERFORMANCE_LOG_PATH = os.path.join(LOGS_DIR, "performance.jsonl")
TRAINING_LOG_INTERVAL = 10  # log metrics every N batches

# File paths
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.json")
WEIGHTS_PATH = os.path.join(CHECKPOINTS_DIR, "innie_weights.npz")
MEMORY_PATH = os.path.join(BASE_DIR, "backend", "memory_store.json")

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
VOCAB_SIZE = 4000
EMBEDDING_DIM = 64
EMBED_DIM = EMBEDDING_DIM  # Alias for performance tooling
HIDDEN_DIM = 128
CONTEXT_WINDOW = 16
LEARNING_RATE = 0.05
EPOCHS = 50
BATCH_SIZE = 8
SEED = 42

# ---------------------------------------------------------------------------
# Memory system configuration
# ---------------------------------------------------------------------------
SHORT_TERM_MEMORY_LIMIT = 20
LONG_TERM_MEMORY_ENABLED = True

# ---------------------------------------------------------------------------
# API / server configuration
# ---------------------------------------------------------------------------
API_HOST = "127.0.0.1"
API_PORT = 5050
API_MAX_TOKENS = 50
API_TEMPERATURE = 0.3
DEBUG_MODE = True

# ---------------------------------------------------------------------------
# Special tokens used by the tokenizer
# ---------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def ensure_directories() -> None:
    """Create every directory this project depends on if it doesn't exist yet."""
    for path in (DATASETS_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR):
        os.makedirs(path, exist_ok=True)
