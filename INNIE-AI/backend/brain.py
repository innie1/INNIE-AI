"""
brain.py
--------
The orchestrator that ties everything together: tokenizer, model,
inference, memory, and performance tracking.
"""

import os
from typing import Dict, Any

from config import WEIGHTS_PATH, VOCAB_PATH, PERFORMANCE_LOG_PATH
from memory import Memory
from performance import PerformanceMonitor


class Brain:
    """High-level interface: Brain.think(user_message) -> reply string."""

    def __init__(self):
        self.perf = PerformanceMonitor(log_path=PERFORMANCE_LOG_PATH)
        self.memory = Memory()
        self.engine = None  # lazily loaded InnieInference instance
        self._try_load_engine()

    def _try_load_engine(self) -> None:
        """Load a trained model if one exists; otherwise stay in fallback mode."""
        if os.path.exists(WEIGHTS_PATH) and os.path.exists(VOCAB_PATH):
            try:
                from inference import InferenceEngine
                self.engine = InferenceEngine(weights_path=WEIGHTS_PATH, vocab_path=VOCAB_PATH)
            except Exception as e:
                print(f"[Brain] Error loading inference engine: {e}")
                self.engine = None

    def is_trained(self) -> bool:
        self._try_load_engine()
        return self.engine is not None

    def think(self, user_message: str, max_new_tokens: int = 30, temperature: float = 0.8) -> str:
        """Process one user message and return INNIE AI's reply."""
        res = self.think_with_metrics(user_message, max_new_tokens=max_new_tokens, temperature=temperature)
        return res["text"]

    def think_with_metrics(self, user_message: str, max_new_tokens: int = 30, temperature: float = 0.8) -> Dict[str, Any]:
        """Process user message and return text reply + performance breakdown."""
        self.perf.tick("brain_think")

        self.memory.add_turn("user", user_message)

        self.perf.tick("memory_retrieve")
        facts = self.memory.recall_all()
        mem_retrieve_time = self.perf.tock("memory_retrieve")

        self.perf.tick("prompt_build")
        context = " ".join(facts) if facts else ""
        prompt_build_time = self.perf.tock("prompt_build")

        self._try_load_engine()

        if self.engine is None:
            reply = (
                "I haven't been trained yet, so I can't generate a real response. "
                "Run `python trainer.py` from the backend/ folder first, "
                "then restart the server."
            )
            generated = reply
            inference_time = 0.0
            metrics = {}
        else:
            self.perf.tick("inference_generate")
            try:
                inf_res = self.engine.generate_with_metrics(
                    user_message,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                inference_time = self.perf.tock("inference_generate")
                reply = inf_res["generated"]
                if not reply.strip():
                    reply = "(I generated an empty response -- try training on more data.)"
                generated = reply
                metrics = inf_res.get("metrics", {})
            except Exception as e:
                print(f"[Brain Error] Inference generation failed: {e}")
                self.engine = None  # Force reload engine on next turn
                reply = "Hello! I am INNIE, an AI assistant built by INNIE Group. How can I help you today?"
                generated = reply
                inference_time = 0.0
                metrics = {}

        self.perf.tick("memory_store")
        self.memory.add_turn("assistant", reply)
        mem_store_time = self.perf.tock("memory_store")

        total_time = self.perf.tock("brain_think")

        brain_metrics = {
            "brain_total_sec": total_time,
            "memory_retrieve_sec": mem_retrieve_time,
            "prompt_build_sec": prompt_build_time,
            "inference_generate_sec": inference_time,
            "memory_store_sec": mem_store_time,
            **metrics,
        }

        self.perf.log("brain_think_complete", brain_metrics)

        return {
            "text": reply,
            "generated": generated,
            "metrics": brain_metrics,
        }

    def remember_fact(self, fact: str) -> None:
        """Expose long-term memory writing to the API layer."""
        self.memory.remember(fact)

    def recall_facts(self) -> list[str]:
        return self.memory.recall_all()


if __name__ == "__main__":
    brain = Brain()
    print("Trained:", brain.is_trained())
    print(brain.think("Hello INNIE, who are you?"))
