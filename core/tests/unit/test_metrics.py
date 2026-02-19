import time
import threading
from unittest.mock import Mock

import pytest

from scholardevclaw.utils.metrics import (
    Counter,
    Gauge,
    Histogram,
    HistogramBucket,
    MetricsRegistry,
    registry,
    REQUESTS_TOTAL,
    REQUEST_DURATION,
    ACTIVE_REQUESTS,
    INTEGRATIONS_TOTAL,
    INTEGRATION_DURATION,
    PATCHES_GENERATED,
    VALIDATIONS_RUN,
    VALIDATIONS_PASSED,
    ERRORS_TOTAL,
    WORKFLOW_NODES_EXECUTED,
    WORKFLOW_NODE_DURATION,
    track_request,
    track_integration,
    track_patch_generated,
    track_workflow_node,
    timing_histogram,
)


class TestCounter:
    def test_counter_starts_at_zero(self):
        counter = Counter(name="test_counter")
        assert counter.value == 0.0

    def test_counter_increment(self):
        counter = Counter(name="test_counter")
        counter.inc()
        assert counter.value == 1.0

    def test_counter_increment_by_amount(self):
        counter = Counter(name="test_counter")
        counter.inc(5.0)
        assert counter.value == 5.0

    def test_counter_cannot_decrement(self):
        counter = Counter(name="test_counter")
        with pytest.raises(ValueError, match="Counter can only increase"):
            counter.inc(-1.0)

    def test_counter_multiple_increments(self):
        counter = Counter(name="test_counter")
        counter.inc()
        counter.inc()
        counter.inc(3.0)
        assert counter.value == 5.0

    def test_counter_reset(self):
        counter = Counter(name="test_counter")
        counter.inc(10.0)
        counter.reset()
        assert counter.value == 0.0


class TestGauge:
    def test_gauge_starts_at_zero(self):
        gauge = Gauge(name="test_gauge")
        assert gauge.value == 0.0

    def test_gauge_set(self):
        gauge = Gauge(name="test_gauge")
        gauge.set(42.0)
        assert gauge.value == 42.0

    def test_gauge_increment(self):
        gauge = Gauge(name="test_gauge")
        gauge.set(10.0)
        gauge.inc(5.0)
        assert gauge.value == 15.0

    def test_gauge_decrement(self):
        gauge = Gauge(name="test_gauge")
        gauge.set(10.0)
        gauge.dec(3.0)
        assert gauge.value == 7.0

    def test_gauge_can_go_negative(self):
        gauge = Gauge(name="test_gauge")
        gauge.set(5.0)
        gauge.dec(10.0)
        assert gauge.value == -5.0


class TestHistogram:
    def test_histogram_default_buckets(self):
        histogram = Histogram(name="test_histogram")
        bucket_bounds = [b.upper_bound for b in histogram.buckets]
        assert 0.005 in bucket_bounds
        assert 0.01 in bucket_bounds
        assert 0.1 in bucket_bounds
        assert 1.0 in bucket_bounds
        assert float("inf") in bucket_bounds

    def test_histogram_observe(self):
        histogram = Histogram(name="test_histogram")
        histogram.observe(0.5)
        assert histogram.count == 1
        assert histogram.sum == 0.5

    def test_histogram_multiple_observations(self):
        histogram = Histogram(name="test_histogram")
        histogram.observe(0.1)
        histogram.observe(0.2)
        histogram.observe(0.3)
        assert histogram.count == 3
        assert abs(histogram.sum - 0.6) < 0.0001

    def test_histogram_bucket_counts(self):
        histogram = Histogram(name="test_histogram")
        histogram.observe(0.05)
        histogram.observe(0.05)

        buckets_005 = [b for b in histogram.buckets if b.upper_bound == 0.05]
        assert len(buckets_005) == 1
        assert buckets_005[0].count == 2

    def test_histogram_reset(self):
        histogram = Histogram(name="test_histogram")
        histogram.observe(0.5)
        histogram.observe(0.3)
        histogram.reset()
        assert histogram.sum == 0.0
        assert histogram.count == 0.0


class TestMetricsRegistry:
    def test_create_counter(self):
        reg = MetricsRegistry(namespace="test")
        counter = reg.counter("requests", help_text="Total requests")
        assert counter.name == "test_requests"
        assert counter.metric_type == "counter"

    def test_create_gauge(self):
        reg = MetricsRegistry(namespace="test")
        gauge = reg.gauge("active", help_text="Active connections")
        assert gauge.name == "test_active"
        assert gauge.metric_type == "gauge"

    def test_create_histogram(self):
        reg = MetricsRegistry(namespace="test")
        histogram = reg.histogram("duration", help_text="Request duration")
        assert histogram.name == "test_duration"
        assert histogram.metric_type == "histogram"

    def test_reuse_same_metric(self):
        reg = MetricsRegistry(namespace="test")
        counter1 = reg.counter("requests")
        counter2 = reg.counter("requests")
        counter1.inc()
        assert counter2.value == 1.0

    def test_get_metric(self):
        reg = MetricsRegistry(namespace="test")
        reg.counter("requests")
        metric = reg.get_metric("requests")
        assert metric is not None
        assert metric.name == "test_requests"

    def test_get_nonexistent_metric(self):
        reg = MetricsRegistry(namespace="test")
        metric = reg.get_metric("nonexistent")
        assert metric is None

    def test_all_metrics(self):
        reg = MetricsRegistry(namespace="test")
        reg.counter("requests")
        reg.gauge("active")
        all_metrics = reg.all_metrics()
        assert "test_requests" in all_metrics
        assert "test_active" in all_metrics

    def test_clear(self):
        reg = MetricsRegistry(namespace="test")
        reg.counter("requests")
        reg.clear()
        assert len(reg.all_metrics()) == 0

    def test_thread_safety(self):
        reg = MetricsRegistry(namespace="test")
        counter = reg.counter("concurrent")

        def increment():
            for _ in range(100):
                counter.inc()

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.value == 1000


class TestPrometheusExport:
    def test_export_counter(self):
        reg = MetricsRegistry(namespace="test")
        counter = reg.counter("requests", help_text="Total requests")
        counter.inc(5)
        output = reg.export_prometheus()
        assert "# HELP test_requests Total requests" in output
        assert "# TYPE test_requests counter" in output
        assert "test_requests 5" in output

    def test_export_gauge(self):
        reg = MetricsRegistry(namespace="test")
        gauge = reg.gauge("active", help_text="Active requests")
        gauge.set(10)
        output = reg.export_prometheus()
        assert "# HELP test_active Active requests" in output
        assert "# TYPE test_active gauge" in output
        assert "test_active 10" in output

    def test_export_histogram(self):
        reg = MetricsRegistry(namespace="test")
        histogram = reg.histogram("duration", help_text="Duration", buckets=[0.1, 1.0])
        histogram.observe(0.5)
        output = reg.export_prometheus()
        assert "# HELP test_duration Duration" in output
        assert "# TYPE test_duration histogram" in output
        assert "test_duration_bucket" in output
        assert "test_duration_sum 0.5" in output
        assert "test_duration_count 1" in output

    def test_export_with_labels(self):
        reg = MetricsRegistry(namespace="test")
        counter = reg.counter("requests", labels={"method": "GET"})
        counter.inc()
        output = reg.export_prometheus()
        assert 'method="GET"' in output

    def test_export_inf_bucket(self):
        reg = MetricsRegistry(namespace="test")
        histogram = reg.histogram("duration")
        histogram.observe(100.0)
        output = reg.export_prometheus()
        assert 'le="+Inf"' in output


class TestPredefinedMetrics:
    def test_requests_total_exists(self):
        assert REQUESTS_TOTAL.name == "scholardevclaw_http_requests_total"

    def test_request_duration_exists(self):
        assert REQUEST_DURATION.name == "scholardevclaw_http_request_duration_seconds"

    def test_active_requests_exists(self):
        assert ACTIVE_REQUESTS.name == "scholardevclaw_http_active_requests"

    def test_integrations_total_exists(self):
        assert INTEGRATIONS_TOTAL.name == "scholardevclaw_integrations_total"


class TestTrackFunctions:
    def test_track_request(self):
        before = REQUESTS_TOTAL.value
        track_request("GET", "/api/test", 200, 0.5)
        assert REQUESTS_TOTAL.value == before + 1
        assert REQUEST_DURATION.count > 0

    def test_track_integration_success(self):
        before_total = INTEGRATIONS_TOTAL.value
        before_passed = VALIDATIONS_PASSED.value

        track_integration(1.0, success=True)

        assert INTEGRATIONS_TOTAL.value == before_total + 1
        assert VALIDATIONS_PASSED.value == before_passed + 1

    def test_track_integration_failure(self):
        before_total = INTEGRATIONS_TOTAL.value
        before_errors = ERRORS_TOTAL.value

        track_integration(1.0, success=False)

        assert INTEGRATIONS_TOTAL.value == before_total + 1
        assert ERRORS_TOTAL.value == before_errors + 1

    def test_track_patch_generated(self):
        before = PATCHES_GENERATED.value
        track_patch_generated()
        assert PATCHES_GENERATED.value == before + 1

    def test_track_workflow_node(self):
        before = WORKFLOW_NODES_EXECUTED.value
        before_count = WORKFLOW_NODE_DURATION.count

        track_workflow_node("test_node", 0.5)

        assert WORKFLOW_NODES_EXECUTED.value == before + 1
        assert WORKFLOW_NODE_DURATION.count == before_count + 1


class TestTimingDecorator:
    def test_timing_histogram_decorator(self):
        reg = MetricsRegistry(namespace="test")
        histogram = reg.histogram("func_duration")

        @timing_histogram(histogram)
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"
        assert histogram.count == 1
        assert histogram.sum >= 0.01

    def test_timing_decorator_with_exception(self):
        reg = MetricsRegistry(namespace="test")
        histogram = reg.histogram("func_duration")

        @timing_histogram(histogram)
        def failing_function():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            failing_function()

        assert histogram.count == 1
