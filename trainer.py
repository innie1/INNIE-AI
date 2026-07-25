"""
trainer.py — Training pipeline for INNIE AI with performance monitoring
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
    PERFORMANCE_LOG_PATH,
    TRAINING_LOG_INTERVAL,
    ensure_directories,
)
from tokenizer import Tokenizer
from model import InnieModel
from performance import PerformanceMonitor, ThroughputMeter


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
    """Slide a window over a token sequence to build (context, next_token) pairs."""
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
    """Run the full training pipeline with performance tracking and save resulting model + vocab."""
    ensure_directories()
    perf = PerformanceMonitor(log_path=PERFORMANCE_LOG_PATH)
    perf.tick("total_training")

    # Data loading
    perf.tick("data_loading")
    texts = load_dataset_texts()
    if not texts:
        texts = [
            "INNIE AI is a modular artificial intelligence system built from scratch. "
            "It learns to predict the next word in a sentence by studying examples. "
            "Over time it will grow from a small nano network into a much larger model."
        ]
        if verbose:
            print("No .txt files found in datasets/, using built-in sample text instead.")

    data_load_time = perf.tock("data_loading")
    total_chars = sum(len(t) for t in texts)
    perf.log("data_loading", {"elapsed_sec": data_load_time, "files": len(texts), "chars": total_chars})

    # Tokenizer build
    perf.tick("tokenizer_build")
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(texts)
    tokenizer.save(VOCAB_PATH)
    tok_time = perf.tock("tokenizer_build")
    vocab_size = len(tokenizer)
    perf.log("tokenizer_build", {"elapsed_sec": tok_time, "vocab_size": vocab_size})

    # Build pairs
    perf.tick("pair_building")
    all_pairs = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        all_pairs.extend(build_training_pairs(ids, context_window))
    pair_time = perf.tock("pair_building")
    perf.log("pair_building", {"elapsed_sec": pair_time, "pairs": len(all_pairs)})

    if not all_pairs:
        raise ValueError("Not enough text to build a single training example.")

    # Model init
    perf.tick("model_init")
    model = InnieModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    init_time = perf.tock("model_init")
    param_count = (
        model.W1.size + model.b1.size +
        model.W2.size + model.b2.size +
        model.embedding.weights.size
    )
    perf.log("model_init", {"elapsed_sec": init_time, "parameters": param_count})

    if verbose:
        print(f"Loaded {len(texts)} document(s), {len(all_pairs)} training pair(s), "
              f"vocab size {vocab_size}, params {param_count}")

    # Training loop
    epoch_meter = ThroughputMeter()
    batch_meter = ThroughputMeter()

    for epoch in range(1, epochs + 1):
        epoch_meter.start()
        perf.tick(f"epoch_{epoch}")
        total_loss = 0.0
        batch_meter.start()

        for idx, (context, target) in enumerate(all_pairs):
            perf.tick("forward_pass")
            model.forward(context)
            fwd_time = perf.tock("forward_pass")

            perf.tick("backward_pass")
            loss = model.backward(target, learning_rate)
            bwd_time = perf.tock("backward_pass")

            total_loss += loss
            batch_meter.add(1)

            if (idx + 1) % TRAINING_LOG_INTERVAL == 0 or idx == 0:
                perf.log("batch_step", {
                    "epoch": epoch,
                    "batch": idx + 1,
                    "loss": float(loss),
                    "forward_sec": fwd_time,
                    "backward_sec": bwd_time,
                    "tokens_per_sec": batch_meter.rate() * context_window,
                })

        avg_loss = total_loss / len(all_pairs)
        epoch_time = perf.tock(f"epoch_{epoch}")
        tokens_per_sec = (len(all_pairs) * context_window) / epoch_time if epoch_time > 0 else 0.0
        mem = perf.get_memory_mb()

        perf.log("epoch_complete", {
            "epoch": epoch,
            "avg_loss": float(avg_loss),
            "epoch_sec": epoch_time,
            "tokens_per_sec": tokens_per_sec,
            "rss_mb": mem["rss_mb"],
        })

        if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs):
            print(f"Epoch {epoch:>4}/{epochs} | avg loss: {avg_loss:.4f} | "
                  f"time: {epoch_time:.2f}s | tok/s: {tokens_per_sec:.1f} | "
                  f"mem: {mem['rss_mb']:.1f}MB")

    total_time = perf.tock("total_training")
    perf.log("training_complete", {
        "total_sec": total_time,
        "epochs": epochs,
        "final_avg_loss": float(avg_loss),
    })

    # Save checkpoint
    perf.tick("save_checkpoint")
    model.save(WEIGHTS_PATH)
    save_time = perf.tock("save_checkpoint")
    perf.log("save_checkpoint", {"elapsed_sec": save_time})

    if verbose:
        print(f"Training complete in {total_time:.2f}s. Weights saved to {WEIGHTS_PATH}")
        perf.print_summary("epoch_complete")

    return model, tokenizer


if __name__ == "__main__":
    train()
