"""Structured JSON logging for Cloud Logging with trace correlation."""

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any, Optional

# Patterns to redact from log output
SECRET_PATTERNS = (
    re.compile(r"(api_key|token|secret|password)\s*[:=]\s*['\"]?[\w-]{20,}['\"]?", re.I),
    re.compile(r"rpa_[\w]+", re.I),
    re.compile(r"hf_[\w]+", re.I),
)


def _redact(message: str) -> str:
    def repl(m: re.Match[str]) -> str:
        if m.lastindex and m.lastindex >= 1:
            return f"{m.group(1)}=***REDACTED***"
        return "***REDACTED***"

    for pat in SECRET_PATTERNS:
        message = pat.sub(repl, message)
    return message


def _redact_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        redacted = {}
        for k, v in obj.items():
            key_lower = str(k).lower()
            if any(s in key_lower for s in ("api_key", "token", "secret", "password", "authorization")):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = _redact_dict(v)
        return redacted
    if isinstance(obj, list):
        return [_redact_dict(i) for i in obj]
    if isinstance(obj, str):
        return _redact(obj)
    return obj


def structured_log(
    level: str,
    message: str,
    *,
    deployment_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    operation: Optional[str] = None,
    duration_ms: Optional[int | float] = None,
    metadata: Optional[dict[str, Any]] = None,
    error: Optional[dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Emit a structured log entry for Cloud Logging."""
    log = logging.getLogger(logger.name if logger else __name__)
    payload: dict[str, Any] = {
        "severity": level.upper(),
        "message": _redact(message),
        "timestamp": datetime.now(UTC).isoformat() + "Z",
    }
    if deployment_id:
        payload["deployment_id"] = deployment_id
    if trace_id:
        payload["trace"] = f"projects/PLACEHOLDER/traces/{trace_id}"
        payload["trace_id"] = trace_id
    if span_id:
        payload["span_id"] = span_id
    if operation:
        payload["operation"] = operation
    if duration_ms is not None:
        payload["duration_ms"] = round(duration_ms, 2)
    if metadata:
        payload["metadata"] = _redact_dict(metadata)
    if error:
        payload["error"] = _redact_dict(error)

    msg = json.dumps(payload) if _use_json() else _format_readable(payload)
    getattr(log, level.lower(), log.info)(msg)


def _use_json() -> bool:
    """Use JSON format when not in development (e.g. Cloud Run)."""
    import os
    return os.getenv("LOG_FORMAT", "json").lower() == "json"


def _format_readable(payload: dict[str, Any]) -> str:
    parts = [f"[{payload.get('severity', 'INFO')}]", payload.get("message", "")]
    if payload.get("deployment_id"):
        parts.append(f"deployment_id={payload['deployment_id']}")
    if payload.get("operation"):
        parts.append(f"operation={payload['operation']}")
    if payload.get("duration_ms") is not None:
        parts.append(f"duration_ms={payload['duration_ms']}")
    return " ".join(parts)


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logger with JSON or readable format."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        if _use_json():
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        payload: dict[str, Any] = {
            "severity": record.levelname,
            "message": _redact(record.getMessage()),
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat().replace("+00:00", "Z"),
        }
        if getattr(record, "deployment_id", None):
            payload["deployment_id"] = record.deployment_id
        if getattr(record, "trace_id", None):
            payload["trace"] = f"projects/PLACEHOLDER/traces/{record.trace_id}"
            payload["trace_id"] = record.trace_id
        if getattr(record, "span_id", None):
            payload["span_id"] = record.span_id
        if getattr(record, "operation", None):
            payload["operation"] = record.operation
        if getattr(record, "duration_ms", None) is not None:
            payload["duration_ms"] = record.duration_ms
        if getattr(record, "metadata", None):
            payload["metadata"] = _redact_dict(record.metadata)
        if record.exc_info:
            payload["error"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "",
                "message": str(record.exc_info[1]) if record.exc_info[1] else "",
                "stack_trace": self.formatException(record.exc_info) if record.exc_info[2] else "",
            }
        return json.dumps(payload)
