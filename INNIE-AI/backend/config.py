"""
config.py
---------
Central configuration for INNIE AI.

Every tunable value in the system lives here so that the rest of the
codebase never hardcodes a "magic number". As INNIE AI grows from a
tiny nano-network into something closer to a real language model,
this is the file you will touch most often.
"""

import os

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# BASE_DIR points at the root of the INNIE-AI project (one level above backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# Where the tokenizer stores its learned vocabulary
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.json")

# Where trained model weights are saved/loaded from
WEIGHTS_PATH = os.path.join(CHECKPOINTS_DIR, "innie_weights.npz")

# Where long-term conversational memory is persisted between runs
MEMORY_PATH = os.path.join(BASE_DIR, "backend", "memory_store.json")


# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
# NOTE: These are intentionally small so the model trains fast on a laptop
# with no GPU. Increase them as INNIE AI evolves into a larger system.

VOCAB_SIZE = 4000        # Max number of tokens the tokenizer will learn
EMBEDDING_DIM = 64       # Size of each token's embedding vector
HIDDEN_DIM = 128         # Size of the hidden layer in the network
CONTEXT_WINDOW = 16      # How many previous tokens the model looks at
LEARNING_RATE = 0.05
EPOCHS = 50
BATCH_SIZE = 8
SEED = 42                # Fixed seed for reproducible training runs


# ---------------------------------------------------------------------------
# Memory system configuration
# ---------------------------------------------------------------------------
SHORT_TERM_MEMORY_LIMIT = 20   # Number of recent turns kept in RAM
LONG_TERM_MEMORY_ENABLED = True


# ---------------------------------------------------------------------------
# API / server configuration
# ---------------------------------------------------------------------------
API_HOST = "127.0.0.1"
API_PORT = 5050
DEBUG_MODE = True


# ---------------------------------------------------------------------------
# Special tokens used by the tokenizer
# ---------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"   # Beginning of sequence
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def ensure_directories() -> None:
    """Create every directory this project depends on if it doesn't exist yet."""
    for path in (DATASETS_DIR, MODELS_DIR, CHECKPOINTS_DIR):
        os.makedirs(path, exist_ok=True)
