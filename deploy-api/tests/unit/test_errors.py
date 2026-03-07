"""Unit tests for custom exceptions."""


from src.core.errors import (
    DeploymentNotFoundError,
    HuggingFaceModelNotFoundError,
    OrchestratorError,
    RunpodInsufficientGPUError,
    WebhookDeliveryError,
)


def test_hf_error_has_status_404() -> None:
    e = HuggingFaceModelNotFoundError("foo/bar")
    assert e.status_code == 404
    assert e.error_code == "HF_MODEL_NOT_FOUND"
    assert "foo/bar" in e.message


def test_runpod_insufficient_gpu() -> None:
    e = RunpodInsufficientGPUError(24)
    assert e.status_code == 503
    assert e.error_code == "RUNPOD_INSUFFICIENT_GPU"
    assert e.details.get("required_vram_gb") == 24


def test_webhook_delivery_error() -> None:
    e = WebhookDeliveryError("https://x.com")
    assert e.error_code == "WEBHOOK_DELIVERY_FAILED"
    assert "x.com" in e.message


def test_deployment_not_found() -> None:
    e = DeploymentNotFoundError("dep_123")
    assert e.status_code == 404
    assert e.error_code == "DEPLOYMENT_NOT_FOUND"
    assert e.details.get("deployment_id") == "dep_123"


def test_unauthorized_error_code() -> None:
    from src.core.errors import UnauthorizedError

    e = UnauthorizedError()
    assert e.status_code == 401
    assert e.error_code == "UNAUTHORIZED"


def test_invalid_deployment_request_error_code() -> None:
    from src.core.errors import InvalidDeploymentRequestError

    e = InvalidDeploymentRequestError("bad request")
    assert e.status_code == 400
    assert e.error_code == "INVALID_DEPLOYMENT_REQUEST"


def test_orchestrator_error_base() -> None:
    e = OrchestratorError("msg", status_code=500, error_code="TestError")
    assert str(e) == "msg"
    assert e.error_code == "TestError"
