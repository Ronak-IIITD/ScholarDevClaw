from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any
from functools import wraps


@dataclass
class MetricValue:
    """Base metric value container."""

    name: str
    help_text: str = ""
    metric_type: str = "gauge"


@dataclass
class Counter(MetricValue):
    """A counter that only increases."""

    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: str = "counter"

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("Counter can only increase")
        self.value += amount

    def reset(self) -> None:
        self.value = 0.0


@dataclass
class Gauge(MetricValue):
    """A gauge that can increase or decrease."""

    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"

    def set(self, value: float) -> None:
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount


@dataclass
class HistogramBucket:
    """A single histogram bucket."""

    upper_bound: float
    count: float = 0.0


@dataclass
class Histogram(MetricValue):
    """A histogram for observing distributions."""

    buckets: list[HistogramBucket] = field(default_factory=list)
    sum: float = 0.0
    count: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: str = "histogram"

    def __post_init__(self):
        if not self.buckets:
            self.buckets = [
                HistogramBucket(0.005),
                HistogramBucket(0.01),
                HistogramBucket(0.025),
                HistogramBucket(0.05),
                HistogramBucket(0.1),
                HistogramBucket(0.25),
                HistogramBucket(0.5),
                HistogramBucket(1.0),
                HistogramBucket(2.5),
                HistogramBucket(5.0),
                HistogramBucket(10.0),
                HistogramBucket(float("inf")),
            ]

    def observe(self, value: float) -> None:
        self.sum += value
        self.count += 1
        for bucket in self.buckets:
            if value <= bucket.upper_bound:
                bucket.count += 1

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0.0
        for bucket in self.buckets:
            bucket.count = 0.0


class MetricsRegistry:
    """Registry for all metrics."""

    def __init__(self, namespace: str = "scholardevclaw"):
        self.namespace = namespace
        self._metrics: dict[str, MetricValue] = {}
        self._lock = threading.RLock()

    def _make_name(self, name: str) -> str:
        return f"{self.namespace}_{name}"

    def counter(
        self,
        name: str,
        help_text: str = "",
        labels: dict[str, str] | None = None,
    ) -> Counter:
        full_name = self._make_name(name)
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Counter(
                    name=full_name,
                    help_text=help_text,
                    labels=labels or {},
                )
            return self._metrics[full_name]  # type: ignore

    def gauge(
        self,
        name: str,
        help_text: str = "",
        labels: dict[str, str] | None = None,
    ) -> Gauge:
        full_name = self._make_name(name)
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Gauge(
                    name=full_name,
                    help_text=help_text,
                    labels=labels or {},
                )
            return self._metrics[full_name]  # type: ignore

    def histogram(
        self,
        name: str,
        help_text: str = "",
        buckets: list[float] | None = None,
        labels: dict[str, str] | None = None,
    ) -> Histogram:
        full_name = self._make_name(name)
        with self._lock:
            if full_name not in self._metrics:
                bucket_objs = None
                if buckets:
                    bucket_objs = [HistogramBucket(b) for b in buckets] + [
                        HistogramBucket(float("inf"))
                    ]
                self._metrics[full_name] = Histogram(
                    name=full_name,
                    help_text=help_text,
                    buckets=bucket_objs or [],
                    labels=labels or {},
                )
            return self._metrics[full_name]  # type: ignore

    def get_metric(self, name: str) -> MetricValue | None:
        full_name = self._make_name(name)
        return self._metrics.get(full_name)

    def all_metrics(self) -> dict[str, MetricValue]:
        return dict(self._metrics)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        for name, metric in self._metrics.items():
            lines.append(f"# HELP {name} {metric.help_text}")
            lines.append(f"# TYPE {name} {metric.metric_type}")

            if isinstance(metric, Counter):
                label_str = self._format_labels(metric.labels)
                lines.append(f"{name}{label_str} {metric.value}")

            elif isinstance(metric, Gauge):
                label_str = self._format_labels(metric.labels)
                lines.append(f"{name}{label_str} {metric.value}")

            elif isinstance(metric, Histogram):
                for bucket in metric.buckets:
                    bound = (
                        "+Inf" if bucket.upper_bound == float("inf") else str(bucket.upper_bound)
                    )
                    label_str = self._format_labels({**metric.labels, "le": bound})
                    lines.append(f"{name}_bucket{label_str} {bucket.count}")

                sum_label = self._format_labels(metric.labels)
                lines.append(f"{name}_sum{sum_label} {metric.sum}")
                lines.append(f"{name}_count{sum_label} {metric.count}")

        return "\n".join(lines) + "\n"

    def _format_labels(self, labels: dict[str, str]) -> str:
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ", ".join(pairs) + "}"

    def clear(self) -> None:
        with self._lock:
            self._metrics.clear()


# Global registry
registry = MetricsRegistry()


# Pre-defined metrics
REQUESTS_TOTAL = registry.counter(
    "http_requests_total",
    "Total number of HTTP requests",
    labels={"method": "", "path": "", "status": ""},
)

REQUEST_DURATION = registry.histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
)

ACTIVE_REQUESTS = registry.gauge(
    "http_active_requests",
    "Number of active HTTP requests",
)

INTEGRATIONS_TOTAL = registry.counter(
    "integrations_total",
    "Total number of integrations run",
)

INTEGRATION_DURATION = registry.histogram(
    "integration_duration_seconds",
    "Integration duration in seconds",
)

PATCHES_GENERATED = registry.counter(
    "patches_generated_total",
    "Total number of patches generated",
)

VALIDATIONS_RUN = registry.counter(
    "validations_total",
    "Total number of validations run",
)

VALIDATIONS_PASSED = registry.counter(
    "validations_passed_total",
    "Total number of validations that passed",
)

ERRORS_TOTAL = registry.counter(
    "errors_total",
    "Total number of errors",
    labels={"type": ""},
)

WORKFLOW_NODES_EXECUTED = registry.counter(
    "workflow_nodes_executed_total",
    "Total number of workflow nodes executed",
)

WORKFLOW_NODE_DURATION = registry.histogram(
    "workflow_node_duration_seconds",
    "Workflow node execution duration in seconds",
)


def track_request(method: str, path: str, status: int, duration: float) -> None:
    """Track HTTP request metrics."""
    REQUESTS_TOTAL.labels = {"method": method, "path": path, "status": str(status)}
    REQUESTS_TOTAL.inc()
    REQUEST_DURATION.observe(duration)


def track_integration(duration: float, success: bool) -> None:
    """Track integration metrics."""
    INTEGRATIONS_TOTAL.inc()
    INTEGRATION_DURATION.observe(duration)
    if success:
        VALIDATIONS_RUN.inc()
        VALIDATIONS_PASSED.inc()
    else:
        ERRORS_TOTAL.labels = {"type": "integration"}
        ERRORS_TOTAL.inc()


def track_patch_generated() -> None:
    """Track patch generation."""
    PATCHES_GENERATED.inc()


def track_workflow_node(node_type: str, duration: float) -> None:
    """Track workflow node execution."""
    WORKFLOW_NODES_EXECUTED.inc()
    WORKFLOW_NODE_DURATION.observe(duration)


def timing_histogram(histogram: Histogram):
    """Decorator to time a function and record in histogram."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                histogram.observe(duration)

        return wrapper

    return decorator
