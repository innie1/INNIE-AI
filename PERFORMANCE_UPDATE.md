## Performance Monitoring (v0.0.1-perf)

INNIE AI now ships with a built-in performance monitoring system.

### New files

| File | Purpose |
|------|---------|
| `backend/performance.py` | Core monitoring: timers, memory tracking, throughput, JSONL logging |
| `backend/report.py` | CLI reporter: `python report.py --event epoch_complete --format table` |
| `backend/benchmark.py` | Standardized benchmark runner for inference |

### What gets tracked

**Training (`trainer.py`)**
- Data loading time, tokenizer build time, pair building time
- Per-epoch loss, elapsed time, tokens/sec
- Per-batch forward/backward pass timing
- Memory usage (RSS) at each epoch
- Total training duration

**Inference (`inference.py`)**
- Time to First Token (TTFT)
- Per-token forward pass latency
- Overall tokens/sec throughput
- Memory delta during generation

**API (`api.py`)**
- Per-request latency (before/after middleware)
- `brain.think()` breakdown
- Endpoint-level aggregation via `/api/performance`

**Brain (`brain.py`)**
- Memory retrieve/store timing
- Prompt building time
- Inference generation time

### Quick start

```bash
# 1. Install dependencies into virtualenv
.venv\Scripts\pip install -r requirements.txt

# 2. Train with live metrics
cd INNIE-AI/backend
python trainer.py

# 3. Run benchmark
python benchmark.py

# 4. View performance report
python report.py --event inference_complete --format table
python report.py --event epoch_complete --format csv --output epochs.csv

# 5. Check API performance (while server is running)
curl http://127.0.0.1:5050/api/performance
curl http://127.0.0.1:5050/api/performance/summary
```

### Log format

All events are appended to `logs/performance.jsonl` as newline-delimited JSON:

```json
{"timestamp": "2026-07-25T23:45:00Z", "event": "epoch_complete", "metrics": {"epoch": 1, "avg_loss": 2.34, "epoch_sec": 1.23, "tokens_per_sec": 456.7, "rss_mb": 128.5}}
```

### Design notes

- `PerformanceMonitor` is lightweight and can be instantiated anywhere
- `ThroughputMeter` handles tokens/sec, samples/sec, etc.
- The `@timed_call` decorator makes it easy to instrument any function
- All summaries support mean, std, min, max, median, p95, p99
