"""Prometheus metrics for HTTP, inference, and training resource use."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psutil
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# HTTP 5xx — complements prometheus-fastapi-instrumentator status labels.
HTTP_5XX_TOTAL = Counter(
    "kepler_http_5xx_total",
    "HTTP responses with status code >= 500",
    labelnames=("method", "handler"),
)

INFERENCE_DURATION = Histogram(
    "kepler_inference_duration_seconds",
    "Wall-clock duration of InferenceService.predict",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

INFERENCE_TOTAL = Counter(
    "kepler_inference_requests_total",
    "Inference requests by outcome",
    labelnames=("status",),
)

TRAINING_DURATION = Histogram(
    "kepler_training_duration_seconds",
    "Wall-clock duration of training jobs",
    labelnames=("model_type", "status"),
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
)

TRAINING_TOTAL = Counter(
    "kepler_training_jobs_total",
    "Training jobs by outcome",
    labelnames=("model_type", "status"),
)

MODEL_OP_RSS_BYTES = Gauge(
    "kepler_model_op_rss_bytes",
    "Process RSS after a model operation",
    labelnames=("operation",),
)

MODEL_OP_PEAK_RSS_BYTES = Gauge(
    "kepler_model_op_peak_rss_bytes",
    "Peak process RSS sampled during a model operation",
    labelnames=("operation",),
)

MODEL_OP_CPU_PERCENT = Gauge(
    "kepler_model_op_cpu_percent",
    "Peak process CPU percent sampled during a model operation",
    labelnames=("operation",),
)

_metrics_server_started = False
_metrics_server_lock = threading.Lock()


@contextmanager
def track_model_operation(operation: str) -> Iterator[dict[str, Any]]:
    """Sample CPU/RSS during a model op; record peak gauges when the block exits."""
    process = psutil.Process()
    process.cpu_percent(interval=None)  # prime
    peak_rss = process.memory_info().rss
    peak_cpu = 0.0
    stop = threading.Event()
    state: dict[str, Any] = {"status": "success"}

    def _sample() -> None:
        nonlocal peak_rss, peak_cpu
        while not stop.wait(0.2):
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
                peak_cpu = max(peak_cpu, process.cpu_percent(interval=None))
            except (psutil.Error, OSError):
                break

    sampler = threading.Thread(target=_sample, name=f"metrics-{operation}", daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        yield state
    except Exception:
        state["status"] = "error"
        raise
    finally:
        stop.set()
        sampler.join(timeout=1.0)
        elapsed = time.perf_counter() - started
        try:
            peak_rss = max(peak_rss, process.memory_info().rss)
            peak_cpu = max(peak_cpu, process.cpu_percent(interval=None))
        except (psutil.Error, OSError):
            pass
        MODEL_OP_RSS_BYTES.labels(operation=operation).set(peak_rss)
        MODEL_OP_PEAK_RSS_BYTES.labels(operation=operation).set(peak_rss)
        MODEL_OP_CPU_PERCENT.labels(operation=operation).set(peak_cpu)
        state["duration_seconds"] = elapsed
        state["peak_rss_bytes"] = peak_rss
        state["peak_cpu_percent"] = peak_cpu


def observe_inference(status: str, duration_seconds: float) -> None:
    INFERENCE_TOTAL.labels(status=status).inc()
    INFERENCE_DURATION.observe(duration_seconds)


def observe_training(model_type: str, status: str, duration_seconds: float) -> None:
    TRAINING_TOTAL.labels(model_type=model_type, status=status).inc()
    TRAINING_DURATION.labels(model_type=model_type, status=status).observe(duration_seconds)


def record_http_5xx(method: str, handler: str) -> None:
    HTTP_5XX_TOTAL.labels(method=method, handler=handler).inc()


def start_worker_metrics_server(port: int) -> None:
    """Expose /metrics from the Celery worker process (idempotent)."""
    global _metrics_server_started
    with _metrics_server_lock:
        if _metrics_server_started or port <= 0:
            return
        start_http_server(port)
        _metrics_server_started = True
