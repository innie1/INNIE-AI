"""
inference.py
------------
Loads a trained InnieModel + Tokenizer and generates text with latency,
TTFT (Time to First Token), and throughput performance tracking.
"""

import os
import numpy as np
from typing import Dict, Any

from config import (
    CONTEXT_WINDOW,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    WEIGHTS_PATH,
    VOCAB_PATH,
    API_MAX_TOKENS,
    PERFORMANCE_LOG_PATH,
)
from tokenizer import Tokenizer
from model import InnieModel
from performance import PerformanceMonitor, ThroughputMeter


class InferenceEngine:
    """Wraps a trained model + tokenizer with performance instrumentation."""

    def __init__(self, weights_path: str = WEIGHTS_PATH, vocab_path: str = VOCAB_PATH):
        self.perf = PerformanceMonitor(log_path=PERFORMANCE_LOG_PATH)

        self.perf.tick("load_tokenizer")
        self.tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
        if os.path.exists(vocab_path):
            self.tokenizer.load(vocab_path)
        else:
            raise FileNotFoundError(
                f"No vocab found at {vocab_path}. Run trainer.py first to train a model."
            )
        tok_load_time = self.perf.tock("load_tokenizer")

        self.perf.tick("load_model")
        self.model = InnieModel(
            vocab_size=len(self.tokenizer),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
        )
        if os.path.exists(weights_path):
            self.model.load(weights_path)
        else:
            raise FileNotFoundError(
                f"No trained weights found at {weights_path}. Run trainer.py first."
            )
        model_load_time = self.perf.tock("load_model")

        self.perf.log("inference_init", {
            "tokenizer_load_sec": tok_load_time,
            "model_load_sec": model_load_time,
            "vocab_size": len(self.tokenizer),
        })

    def _sample(self, probs: np.ndarray, temperature: float = 0.3, top_k: int = 5) -> int:
        """Sample a token id using Top-K filtering and temperature scaling to prevent garbled outputs."""
        if temperature <= 0.05:
            return int(np.argmax(probs))

        top_k = min(top_k, len(probs))
        top_indices = np.argsort(probs)[-top_k:]
        top_probs = probs[top_indices]
        top_probs = top_probs / np.sum(top_probs)

        if temperature != 1.0:
            logits = np.log(top_probs + 1e-9) / max(temperature, 0.05)
            exp_logits = np.exp(logits - np.max(logits))
            scaled_probs = exp_logits / np.sum(exp_logits)
        else:
            scaled_probs = top_probs

        return int(np.random.choice(top_indices, p=scaled_probs))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 0.3,
    ) -> str:
        """Backward-compatible generation returning generated string."""
        res = self.generate_with_metrics(prompt, max_tokens=max_new_tokens, temperature=temperature)
        return res["generated"]

    def generate_with_metrics(
        self,
        prompt: str,
        max_tokens: int = API_MAX_TOKENS,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt with rich performance metrics.
        Returns a dict with: text, generated, metrics.
        """
        self.perf.tick("inference_total")

        self.perf.tick("prompt_tokenize")
        context_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        tok_time = self.perf.tock("prompt_tokenize")

        eos_id = self.tokenizer.token_to_id.get("<EOS>")
        generated_ids = []
        token_meter = ThroughputMeter()
        token_meter.start()

        self.perf.tick("ttft")
        ttft = 0.0

        for _ in range(max_tokens):
            window = context_ids[-CONTEXT_WINDOW:]
            
            self.perf.tick("token_forward")
            probs = self.model.forward(window)
            fwd_time = self.perf.tock("token_forward")

            next_id = self._sample(probs, temperature)

            if len(generated_ids) == 0:
                ttft = self.perf.tock("ttft")
                self.perf.log("time_to_first_token", {
                    "prompt_tokens": len(context_ids),
                    "ttft_sec": ttft,
                })

            if next_id == eos_id:
                break

            generated_ids.append(next_id)
            context_ids.append(next_id)
            token_meter.add(1)

        total_time = self.perf.tock("inference_total")
        tokens_per_sec = token_meter.rate()
        latency_per_token = total_time / len(generated_ids) if generated_ids else 0.0

        raw_generated = self.tokenizer.decode(generated_ids)
        # Clean special tokens from output text
        cleaned = raw_generated
        for tag in ["<EOS>", "<PAD>", "<INNIE>", "<USER>", "<SYSTEM>", "<ASSISTANT>"]:
            cleaned = cleaned.replace(tag, "")
        generated_text = " ".join(cleaned.split()).strip()

        full_text = prompt + " " + generated_text

        mem = self.perf.get_memory_mb()
        metrics = {
            "prompt": prompt,
            "prompt_tokens": len(self.tokenizer.encode(prompt, add_special_tokens=True)),
            "generated_tokens": len(generated_ids),
            "total_time_sec": total_time,
            "time_to_first_token_sec": ttft if generated_ids else 0.0,
            "latency_per_token_sec": latency_per_token,
            "tokens_per_sec": tokens_per_sec,
            "temperature": temperature,
            "rss_mb": mem["rss_mb"],
        }

        self.perf.log("inference_complete", metrics)

        return {
            "text": full_text,
            "generated": generated_text,
            "metrics": metrics,
        }


# Alias for backward compatibility
InnieInference = InferenceEngine


def interactive_demo():
    """CLI demo with live performance readout."""
    engine = InferenceEngine()
    print("\nINNIE AI Inference Demo (type 'quit' to exit)")
    print("=" * 50)

    while True:
        prompt = input("\nPrompt> ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        result = engine.generate_with_metrics(prompt, max_tokens=30)
        m = result["metrics"]

        print(f"\n[Generated]: {result['generated']}")
        print(f"[Perf] TTFT={m['time_to_first_token_sec']*1000:.1f}ms | "
              f"tok/s={m['tokens_per_sec']:.1f} | "
              f"latency/tok={m['latency_per_token_sec']*1000:.1f}ms | "
              f"mem={m['rss_mb']:.1f}MB")

    engine.perf.print_summary("inference_complete")


if __name__ == "__main__":
    interactive_demo()
