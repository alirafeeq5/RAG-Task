"""
Phase 2 – Performance Optimization & Monitoring
================================================
Covers:
  • PerformanceMonitor – real-time metrics collection
  • BatchProcessor     – efficient batch embedding & indexing
  • MetricsCollector   – sliding window latency/throughput stats
"""

import time
import math
import logging
import collections
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Metrics Collector
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OperationMetric:
    name:      str
    latency_ms: float
    success:   bool
    timestamp: float = field(default_factory=time.time)
    extra:     Dict  = field(default_factory=dict)


class MetricsCollector:
    """
    Sliding-window metrics store.
    Tracks latency percentiles, throughput, and error rates.
    """

    def __init__(self, window: int = 1000):
        self._window  = window
        self._metrics: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=window)
        )

    def record(self, op: OperationMetric) -> None:
        self._metrics[op.name].append(op)

    def summary(self, op_name: str = None) -> Dict:
        names = [op_name] if op_name else list(self._metrics.keys())
        result = {}
        for name in names:
            buf = list(self._metrics.get(name, []))
            if not buf:
                result[name] = {}
                continue
            latencies = [m.latency_ms for m in buf]
            errors    = sum(1 for m in buf if not m.success)
            latencies.sort()
            n = len(latencies)
            result[name] = {
                "count":       n,
                "error_rate":  round(errors / n, 3),
                "latency_ms":  {
                    "mean":  round(statistics.mean(latencies), 1),
                    "p50":   round(latencies[int(n * 0.50)], 1),
                    "p95":   round(latencies[int(n * 0.95)], 1),
                    "p99":   round(latencies[min(int(n * 0.99), n-1)], 1),
                    "max":   round(max(latencies), 1),
                },
            }
            # Throughput over the last 60 seconds
            now      = time.time()
            recent   = [m for m in buf if now - m.timestamp < 60]
            result[name]["throughput_per_min"] = len(recent)
        return result


# ──────────────────────────────────────────────────────────────────────────────
#  Performance Monitor (context manager / decorator)
# ──────────────────────────────────────────────────────────────────────────────

class PerformanceMonitor:
    """
    Wraps any operation with timing and records the result to MetricsCollector.

    Usage (as context manager)
    --------------------------
    monitor = PerformanceMonitor()
    with monitor.track("embed"):
        vecs = engine.embed_texts(texts)

    Usage (as decorator)
    ---------------------
    @monitor.decorator("llm_call")
    def call_llm(messages):
        ...
    """

    def __init__(self):
        self.collector = MetricsCollector()

    class _Timer:
        def __init__(self, monitor: "PerformanceMonitor", name: str, extra: dict):
            self._monitor  = monitor
            self._name     = name
            self._extra    = extra
            self._start    = 0.0
            self._success  = True

        def __enter__(self):
            self._start = time.time()
            return self

        def mark_failed(self):
            self._success = False

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                self._success = False
            elapsed = (time.time() - self._start) * 1000
            self._monitor.collector.record(OperationMetric(
                name       = self._name,
                latency_ms = elapsed,
                success    = self._success,
                extra      = self._extra,
            ))
            return False   # do not suppress exceptions

    def track(self, name: str, **extra) -> "_Timer":
        return self._Timer(self, name, extra)

    def decorator(self, name: str):
        def wrapper(fn: Callable) -> Callable:
            def inner(*args, **kwargs):
                with self.track(name):
                    return fn(*args, **kwargs)
            inner.__name__ = fn.__name__
            return inner
        return wrapper

    def report(self) -> Dict:
        return self.collector.summary()


# ──────────────────────────────────────────────────────────────────────────────
#  Batch Processor
# ──────────────────────────────────────────────────────────────────────────────

class BatchProcessor:
    """
    Efficient batch processing for embedding and indexing large corpora.

    Features:
      • Configurable batch size
      • Progress reporting (every N batches)
      • Error-resilient: skips failed batches and logs them
      • Throughput estimation
    """

    def __init__(self,
                 embedding_engine,
                 vector_store,
                 batch_size:      int  = 256,
                 report_every:    int  = 10,
                 monitor: Optional[PerformanceMonitor] = None):
        self._engine      = embedding_engine
        self._store       = vector_store
        self._batch_size  = batch_size
        self._report_every= report_every
        self._monitor     = monitor or PerformanceMonitor()

    def _batches(self, items: List, size: int) -> Iterator[List]:
        for i in range(0, len(items), size):
            yield items[i: i + size]

    def index_all(self, chunks: List, texts: Optional[List[str]] = None) -> dict:
        """
        Embed and index all chunks in batches.

        Returns
        -------
        {"indexed": int, "skipped": int, "elapsed_s": float}
        """
        from core.data_processor import DocumentChunk

        # Pre-fit TF-IDF if needed (one pass over all texts)
        all_texts = texts or [c.text for c in chunks]
        self._engine.fit_corpus(all_texts)

        total   = len(chunks)
        indexed = 0
        skipped = 0
        t0      = time.time()

        for batch_num, batch in enumerate(
                self._batches(chunks, self._batch_size), 1):
            try:
                with self._monitor.track("batch_embed", size=len(batch)):
                    batch_texts = [c.text     for c in batch]
                    batch_ids   = [c.chunk_id for c in batch]
                    vecs        = self._engine.embed_texts(batch_texts)

                with self._monitor.track("batch_index", size=len(batch)):
                    self._store.add(vecs, batch_ids)

                indexed += len(batch)

                if batch_num % self._report_every == 0 or indexed == total:
                    elapsed  = time.time() - t0
                    rate     = indexed / max(elapsed, 1e-9)
                    pct      = indexed / total * 100
                    logger.info(
                        f"  [{pct:5.1f}%] {indexed:,}/{total:,} chunks indexed "
                        f"({rate:.0f}/s) …"
                    )

            except Exception as e:
                logger.warning(f"Batch {batch_num} failed ({e}). Skipping.")
                skipped += len(batch)

        elapsed = time.time() - t0
        logger.info(
            f"BatchProcessor complete – "
            f"indexed={indexed:,}, skipped={skipped}, "
            f"elapsed={elapsed:.1f}s, rate={indexed/max(elapsed,1e-9):.0f}/s"
        )
        return {"indexed": indexed, "skipped": skipped, "elapsed_s": round(elapsed, 2)}

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Convenience wrapper for batch embedding with monitoring."""
        with self._monitor.track("embed_batch", size=len(texts)):
            return self._engine.embed_texts(texts)

    @property
    def monitor(self) -> PerformanceMonitor:
        return self._monitor
