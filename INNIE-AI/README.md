# INNIE AI — v0.0.1

A modular, from-scratch AI system built by **INNIE Group**. INNIE AI does not
call OpenAI or any other closed-source AI API — every piece, from tokenization
to the neural network to the training loop, is implemented in this repository
using open-source libraries (NumPy, Flask).

This is version **0.0.1**: a small "nano network" that proves out the full
pipeline end to end. The architecture is intentionally modular so it can grow
into a much larger language model over time without a rewrite.

---

## Architecture

```
User types in browser (frontend/chat.html)
        │
        ▼
   app.js  ──POST /api/chat──▶  api.py (Flask)
                                   │
                                   ▼
                                brain.py  (orchestrator)
                                 │      │
                          memory.py   inference.py
                                          │
                                    tokenizer.py
                                          │
                                    embeddings.py
                                          │
                                     model.py
```

| File | Responsibility |
|---|---|
| `backend/config.py` | All paths and hyperparameters in one place |
| `backend/tokenizer.py` | Word-level tokenizer: text ↔ token ids |
| `backend/embeddings.py` | Learnable token → vector lookup table |
| `backend/model.py` | The neural network (embed → dense → softmax) with manual backprop |
| `backend/trainer.py` | Loads datasets, builds training pairs, trains the model |
| `backend/inference.py` | Loads a trained model and generates text |
| `backend/memory.py` | Short-term (in-RAM) + long-term (JSON file) conversational memory |
| `backend/brain.py` | Orchestrates tokenizer + model + memory into one `think()` call |
| `backend/api.py` | Flask API bridging the frontend and `brain.py` |
| `frontend/` | A plain HTML/CSS/JS chat interface (no build step required) |
| `datasets/` | Training text, organized by domain (`general`, `business`, `coding`, `images`) |
| `models/` | Saved tokenizer vocabulary |
| `checkpoints/` | Saved model weights |
| `tests/` | Unit tests for the tokenizer, embeddings, and model |

### Current model (v0.0.1)

The network is a **context-averaging feedforward network**, not yet a
transformer:

```
token ids → embedding lookup → mean-pool over context window
          → Dense(hidden) + ReLU → Dense(vocab_size) → softmax
```

It is trained on plain next-token prediction with cross-entropy loss and
manual gradient descent (no autograd framework). This keeps the whole
pipeline transparent and easy to extend.

### Growth path (nano → larger model)

Because every layer exposes a clean `forward()`/`backward()` interface, the
planned upgrade path is:

1. Replace the mean-pool + dense block in `model.py` with self-attention →
   nano-transformer (tokenizer, embeddings, memory, and API stay untouched).
2. Swap the word-level `tokenizer.py` for a subword/BPE tokenizer.
3. Replace manual NumPy backprop with a proper autograd/tensor library once
   model size outgrows hand-written gradients.
4. Expand `datasets/` (general, business, coding, images) into a real
   training corpus.

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Training

INNIE AI ships with a small sample dataset (`datasets/general/sample.txt`) so
you can train immediately:

```bash
cd backend
python trainer.py
```

This will:
- Build a tokenizer vocabulary from every `.txt` file under `datasets/`
- Train the model for `EPOCHS` passes (see `config.py`)
- Save the vocabulary to `models/vocab.json`
- Save model weights to `checkpoints/innie_weights.npz`

Add more `.txt` files to `datasets/general/`, `datasets/business/`, or
`datasets/coding/` and re-run `trainer.py` to teach INNIE AI more.

## Running the app

```bash
# Terminal 1 — start the API
cd backend
python api.py
```

Then open `frontend/chat.html` directly in your browser (double-click it, or
use "Open File"). It will connect to the API at `http://127.0.0.1:5050`.

## Running tests

```bash
pip install pytest
pytest tests/
```

---

## Roadmap

- [x] Tokenizer, embeddings, nano feedforward model, training loop
- [x] Memory system (short-term + long-term)
- [x] Local API + web chat interface
- [ ] Self-attention / nano-transformer architecture
- [ ] Subword (BPE) tokenizer
- [ ] Larger, curated datasets (business, coding, general)
- [ ] Image understanding pipeline (`datasets/images/`)

---

## License / Ownership

This project is proprietary to **INNIE Group** unless stated otherwise.
