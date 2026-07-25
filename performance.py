"""
performance.py — Performance monitoring for INNIE AI
Tracks training speed, inference latency, memory usage, throughput, and GPU metrics.
"""
import time
import json
import os
import psutil
from datetime import datetime, timezone
from collections import deque
from typing import Dict, List, Optional, Callable, Any
import numpy as np

from config import PERFORMANCE_LOG_PATH

# ── Optional GPU monitoring ─────────────────────────────────────
_gpu_available = False
_pynvml = None
try:
    import pynvml
    pynvml.nvmlInit()
    _gpu_available = True
    _pynvml = pynvml
except Exception:
    pass


class GPUMonitor:
    """Lightweight GPU metrics wrapper. Falls back gracefully if no GPU."""

    def __init__(self):
        self.available = _gpu_available
        self.device_count = 0
        if self.available:
            try:
                self.device_count = _pynvml.nvmlDeviceGetCount()
            except Exception:
                self.available = False

    def get_metrics(self, device_index: int = 0) -> Dict[str, Any]:
        """Return GPU memory, utilization, and temperature."""
        if not self.available:
            return {
                "available": False,
                "memory_used_mb": 0.0,
                "memory_total_mb": 0.0,
                "utilization_percent": 0.0,
                "temperature_c": 0.0,
            }

        try:
            handle = _pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem_info = _pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = _pynvml.nvmlDeviceGetTemperature(handle, _pynvml.NVML_TEMPERATURE_GPU)

            return {
                "available": True,
                "memory_used_mb": mem_info.used / (1024 * 1024),
                "memory_total_mb": mem_info.total / (1024 * 1024),
                "memory_free_mb": mem_info.free / (1024 * 1024),
                "utilization_percent": util.gpu,
                "temperature_c": temp,
            }
        except Exception:
            return {
                "available": False,
                "memory_used_mb": 0.0,
                "memory_total_mb": 0.0,
                "utilization_percent": 0.0,
                "temperature_c": 0.0,
            }

    def get_all_devices(self) -> List[Dict[str, Any]]:
        """Return metrics for all GPUs."""
        if not self.available or self.device_count == 0:
            return [{"available": False}]
        return [self.get_metrics(i) for i in range(self.device_count)]


class PerformanceMonitor:
    """Central performance tracker with CPU, RAM, and GPU support."""

    def __init__(self, log_path: str = PERFORMANCE_LOG_PATH, max_history: int = 10000):
        self.log_path = log_path
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
        self.process = psutil.Process()
        self.gpu = GPUMonitor()
        self._start_times: Dict[str, float] = {}

        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    # ── Timing helpers ──────────────────────────────────────────────

    def tick(self, label: str) -> None:
        """Start a timed section."""
        self._start_times[label] = time.perf_counter()

    def tock(self, label: str) -> float:
        """End a timed section and return elapsed seconds."""
        if label not in self._start_times:
            raise ValueError(f'tick("{label}") was never called')
        elapsed = time.perf_counter() - self._start_times[label]
        del self._start_times[label]
        return elapsed

    def timed(self, label: str):
        """Context manager for timing blocks."""
        class _Timer:
            def __init__(inner_self, monitor, lbl):
                inner_self.monitor = monitor
                inner_self.label = lbl
                inner_self.elapsed = 0.0
            def __enter__(inner_self):
                inner_self.monitor.tick(inner_self.label)
                return inner_self
            def __exit__(inner_self, *args):
                inner_self.elapsed = inner_self.monitor.tock(inner_self.label)
        return _Timer(self, label)

    # ── System metrics ──────────────────────────────────────────────

    def get_memory_mb(self) -> Dict[str, float]:
        """Return RSS and VSS memory in MB."""
        info = self.process.memory_info()
        return {
            "rss_mb": info.rss / (1024 * 1024),
            "vms_mb": info.vms / (1024 * 1024),
        }

    def get_cpu_percent(self) -> float:
        """Return current CPU percent for this process."""
        return self.process.cpu_percent(interval=None)

    def get_system_snapshot(self) -> Dict[str, Any]:
        """Full snapshot: CPU, RAM, GPU."""
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": self.get_cpu_percent(),
            **self.get_memory_mb(),
        }
        if self.gpu.available:
            gpu_metrics = self.gpu.get_all_devices()
            snapshot["gpu"] = gpu_metrics
        else:
            snapshot["gpu"] = [{"available": False}]
        return snapshot

    # ── Logging ─────────────────────────────────────────────────────

    def log(self, event: str, metrics: Dict, flush: bool = True) -> None:
        """Log a performance event with timestamp."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "metrics": metrics,
        }
        self.history.append(entry)
        if flush:
            self._flush()

    def _flush(self) -> None:
        """Append history to JSONL file."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            for entry in self.history:
                f.write(json.dumps(entry) + "\n")
        self.history.clear()

    def load_history(self) -> List[Dict]:
        """Load all logged events."""
        if not os.path.exists(self.log_path):
            return []
        events = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
        return events

    # ── Summaries ───────────────────────────────────────────────────

    def summarize(self, event_filter: Optional[str] = None) -> Dict:
        """Compute summary statistics for logged events."""
        events = self.load_history()
        if event_filter:
            events = [e for e in events if e.get("event") == event_filter]
        if not events:
            return {}

        all_metrics: Dict[str, List[float]] = {}
        for e in events:
            metrics = e.get("metrics", {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    all_metrics.setdefault(k, []).append(float(v))

        summary = {}
        for k, vals in all_metrics.items():
            if not vals:
                continue
            arr = np.array(vals)
            summary[k] = {
                "count": int(len(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
        return summary

    def print_summary(self, event_filter: Optional[str] = None) -> None:
        """Pretty-print a performance summary."""
        summary = self.summarize(event_filter)
        if not summary:
            print("No performance data found.")
            return

        title = f"Performance Summary: {event_filter or 'ALL EVENTS'}"
        print("\n" + "=" * 60)
        print(title.center(60))
        print("=" * 60)
        for metric, stats in summary.items():
            print(f"\n  {metric}:")
            for stat, val in stats.items():
                if stat == "count":
                    print(f"    {stat:>8}: {val}")
                else:
                    print(f"    {stat:>8}: {val:.6f}")
        print("=" * 60 + "\n")


# ── Decorators for easy instrumentation ───────────────────────────

def timed_call(monitor: PerformanceMonitor, event_name: str):
    """Decorator to time a function call."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            monitor.tick(event_name)
            mem_before = monitor.get_memory_mb()["rss_mb"]
            gpu_before = monitor.gpu.get_metrics() if monitor.gpu.available else None

            result = func(*args, **kwargs)

            elapsed = monitor.tock(event_name)
            mem_after = monitor.get_memory_mb()["rss_mb"]
            metrics = {
                "elapsed_sec": elapsed,
                "memory_delta_mb": mem_after - mem_before,
                "memory_after_mb": mem_after,
            }
            if gpu_before and monitor.gpu.available:
                gpu_after = monitor.gpu.get_metrics()
                metrics["gpu_memory_delta_mb"] = gpu_after["memory_used_mb"] - gpu_before["memory_used_mb"]
                metrics["gpu_utilization"] = gpu_after["utilization_percent"]

            monitor.log(event_name, metrics)
            return result
        return wrapper
    return decorator


# ── Throughput calculator ─────────────────────────────────────────

class ThroughputMeter:
    """Measure tokens/sec, samples/sec, etc."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.total_items = 0

    def start(self):
        self.start_time = time.perf_counter()
        self.total_items = 0

    def add(self, n: int = 1):
        self.total_items += n

    def rate(self) -> float:
        if self.start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self.start_time
        return self.total_items / elapsed if elapsed > 0 else 0.0

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time
