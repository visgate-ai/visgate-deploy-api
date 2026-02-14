"""OpenTelemetry and Cloud Trace integration with custom metrics."""

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

# Optional: only load if packages available
try:
    from opentelemetry import trace
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    trace = None  # type: ignore
    FastAPIInstrumentor = None  # type: ignore

# In-memory metrics (production would use OpenCensus/OTel metrics exporter to GCP)
_metrics: dict[str, list[float] | int] = {
    "deployments_created_total": 0,
    "deployments_ready_duration_seconds": [],
    "webhook_delivery_failures_total": 0,
    "runpod_api_errors_total": 0,
}


def get_tracer(name: str = "deployment-orchestrator") -> Any:
    """Return OpenTelemetry tracer or no-op."""
    if _OTEL_AVAILABLE and trace is not None:
        return trace.get_tracer(name)
    return None


def get_current_span() -> Any:
    """Return current span for trace_id/span_id in logs."""
    if _OTEL_AVAILABLE and trace is not None:
        return trace.get_current_span()
    return None


def get_trace_context() -> dict[str, str]:
    """Return trace_id and span_id for current span (for log correlation)."""
    span = get_current_span()
    if span is None or not span.is_recording():
        return {}
    ctx = span.get_span_context()
    return {"trace_id": format(ctx.trace_id, "032x"), "span_id": format(ctx.span_id, "016x")}


def init_telemetry(service_name: str = "deployment-orchestrator", project_id: Optional[str] = None) -> None:
    """Initialize Cloud Trace exporter and tracer provider."""
    if not _OTEL_AVAILABLE:
        return
    project = project_id or os.getenv("GCP_PROJECT_ID", "")
    if not project:
        return
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = CloudTraceSpanExporter(project_id=project)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def instrument_fastapi(app: Any) -> None:
    """Instrument FastAPI app for automatic tracing."""
    if _OTEL_AVAILABLE and FastAPIInstrumentor is not None:
        FastAPIInstrumentor.instrument_app(app)


def record_deployment_created() -> None:
    """Increment deployments_created_total counter."""
    _metrics["deployments_created_total"] = _metrics.get("deployments_created_total", 0) + 1


def record_deployment_ready_duration(seconds: float) -> None:
    """Record deployment ready duration for histogram."""
    _metrics.setdefault("deployments_ready_duration_seconds", []).append(seconds)


def record_webhook_failure() -> None:
    """Increment webhook_delivery_failures_total."""
    _metrics["webhook_delivery_failures_total"] = _metrics.get("webhook_delivery_failures_total", 0) + 1


def record_runpod_api_error() -> None:
    """Increment runpod_api_errors_total."""
    _metrics["runpod_api_errors_total"] = _metrics.get("runpod_api_errors_total", 0) + 1


def get_metrics() -> dict[str, Any]:
    """Return current metrics snapshot (for /metrics or tests)."""
    out: dict[str, Any] = {}
    for k, v in _metrics.items():
        if isinstance(v, list):
            out[k] = {"count": len(v), "sum": sum(v), "values": v}
        else:
            out[k] = v
    return out


@contextmanager
def span(name: str, attributes: Optional[dict[str, Any]] = None) -> Generator[Any, None, None]:
    """Context manager for a child span."""
    if not _OTEL_AVAILABLE or trace is None:
        yield None
        return
    tracer = trace.get_tracer("deployment-orchestrator")
    with tracer.start_as_current_span(name) as span_obj:
        if attributes:
            for key, val in attributes.items():
                span_obj.set_attribute(key, str(val))
        yield span_obj
