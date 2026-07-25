"""
benchmark.py — Run a standardized benchmark and print results
Usage: python benchmark.py --prompts "Hello world" "What is AI?" "Tell me a story"
"""
import argparse
import time
from inference import InferenceEngine
from performance import PerformanceMonitor
from config import PERFORMANCE_LOG_PATH


def run_benchmark(prompts: list, max_tokens: int = 30, temperature: float = 1.0):
    engine = InferenceEngine()
    perf = PerformanceMonitor(log_path=PERFORMANCE_LOG_PATH)

    print(f"\nRunning benchmark: {len(prompts)} prompts, {max_tokens} max tokens each")
    print("=" * 60)

    results = []
    for i, prompt in enumerate(prompts, 1):
        result = engine.generate_with_metrics(prompt, max_tokens=max_tokens, temperature=temperature)
        m = result["metrics"]
        results.append(m)
        print(f"[{i}/{len(prompts)}] Prompt: {prompt[:40]}...")
        print(f"         TTFT: {m['time_to_first_token_sec']*1000:.1f}ms | "
              f"Tok/s: {m['tokens_per_sec']:.1f} | "
              f"Latency/tok: {m['latency_per_token_sec']*1000:.1f}ms | "
              f"Tokens: {m['generated_tokens']}")

    # Aggregate
    ttfts = [r["time_to_first_token_sec"] * 1000 for r in results]
    tps = [r["tokens_per_sec"] for r in results]
    latencies = [r["latency_per_token_sec"] * 1000 for r in results]

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY".center(60))
    print("=" * 60)
    print(f"  TTFT (ms):        avg={sum(ttfts)/len(ttfts):.1f}  min={min(ttfts):.1f}  max={max(ttfts):.1f}")
    print(f"  Throughput (tok/s): avg={sum(tps)/len(tps):.1f}  min={min(tps):.1f}  max={max(tps):.1f}")
    print(f"  Latency/token (ms): avg={sum(latencies)/len(latencies):.1f}  min={min(latencies):.1f}  max={max(latencies):.1f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="+", default=[
        "Hello world",
        "What is artificial intelligence?",
        "Tell me a short story about a robot.",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
    ])
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    run_benchmark(args.prompts, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
