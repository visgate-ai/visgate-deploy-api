"""Core configuration, logging, errors, and telemetry."""

from src.core.config import Settings, get_settings
from src.core.errors import (
    DeploymentNotFoundError,
    HuggingFaceModelNotFoundError,
    OrchestratorError,
    RateLimitError,
    RunpodAPIError,
    RunpodInsufficientGPUError,
    UnauthorizedError,
    WebhookDeliveryError,
)
from src.core.logging import configure_logging, structured_log
from src.core.telemetry import (
    get_trace_context,
    get_tracer,
    init_telemetry,
    instrument_fastapi,
    record_deployment_created,
    record_deployment_ready_duration,
    record_runpod_api_error,
    record_webhook_failure,
    span,
)

__all__ = [
    "Settings",
    "get_settings",
    "OrchestratorError",
    "HuggingFaceModelNotFoundError",
    "RunpodInsufficientGPUError",
    "RunpodAPIError",
    "WebhookDeliveryError",
    "DeploymentNotFoundError",
    "UnauthorizedError",
    "RateLimitError",
    "configure_logging",
    "structured_log",
    "init_telemetry",
    "instrument_fastapi",
    "get_tracer",
    "get_trace_context",
    "record_deployment_created",
    "record_deployment_ready_duration",
    "record_webhook_failure",
    "record_runpod_api_error",
    "span",
]
