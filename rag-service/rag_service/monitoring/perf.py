from __future__ import annotations

import time
from dataclasses import dataclass

import psutil
from prometheus_client import Gauge, Histogram, CollectorRegistry, exposition


registry = CollectorRegistry()

MEMORY_USAGE = Gauge('memory_usage_mb', 'Memory usage in MB', registry=registry)
GPU_ALLOC_MB = Gauge('gpu_allocated_mb', 'GPU allocated (MB)', registry=registry)
GPU_CACHED_MB = Gauge('gpu_cached_mb', 'GPU cached (MB)', registry=registry)
QUERY_LATENCY = Histogram('query_latency_seconds', 'Query latency', registry=registry)
LM_STUDIO_UP = Gauge('lm_studio_up', 'LM Studio availability (1=up)', registry=registry)
LM_STUDIO_LATENCY = Histogram('lm_studio_latency_seconds', 'LM Studio latency', registry=registry)


@dataclass
class PerformanceMonitor:
    start_time: float = time.time()
    peak_memory: float = 0.0

    def get_memory_usage(self) -> dict:
        process = psutil.Process()
        mem = process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, mem)
        MEMORY_USAGE.set(mem)
        return {
            "resident_mb": mem,
            "percent": process.memory_percent(),
            "peak_mb": self.peak_memory,
        }

    def update_gpu(self):
        try:
            import torch
            if torch.backends.mps.is_available():
                GPU_ALLOC_MB.set(getattr(torch.mps, 'allocated_memory', lambda: 0)() / 1024 / 1024)
                GPU_CACHED_MB.set(getattr(torch.mps, 'cached_memory', lambda: 0)() / 1024 / 1024)
        except Exception:
            pass

    def metrics_response(self) -> bytes:
        self.get_memory_usage()
        self.update_gpu()
        return exposition.generate_latest(registry)
