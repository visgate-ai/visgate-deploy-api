"""Unit tests for custom exceptions."""

import pytest

from src.core.errors import (
    HuggingFaceModelNotFoundError,
    RunpodInsufficientGPUError,
    WebhookDeliveryError,
    DeploymentNotFoundError,
    OrchestratorError,
)


def test_hf_error_has_status_404() -> None:
    e = HuggingFaceModelNotFoundError("foo/bar")
    assert e.status_code == 404
    assert "foo/bar" in e.message


def test_runpod_insufficient_gpu() -> None:
    e = RunpodInsufficientGPUError(24)
    assert e.status_code == 503
    assert e.details.get("required_vram_gb") == 24


def test_webhook_delivery_error() -> None:
    e = WebhookDeliveryError("https://x.com")
    assert "x.com" in e.message


def test_deployment_not_found() -> None:
    e = DeploymentNotFoundError("dep_123")
    assert e.status_code == 404
    assert e.details.get("deployment_id") == "dep_123"


def test_orchestrator_error_base() -> None:
    e = OrchestratorError("msg", status_code=500, error_code="TestError")
    assert str(e) == "msg"
    assert e.error_code == "TestError"
