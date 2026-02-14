"""Custom exceptions for the deployment orchestrator."""

from typing import Any, Optional


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class HuggingFaceModelNotFoundError(OrchestratorError):
    """Raised when the requested model does not exist on Hugging Face Hub."""

    def __init__(self, model_id: str, message: Optional[str] = None) -> None:
        super().__init__(
            message or f"Hugging Face model not found: {model_id}",
            status_code=404,
            details={"hf_model_id": model_id},
        )


class RunpodInsufficientGPUError(OrchestratorError):
    """Raised when no suitable GPU is available on Runpod for the model."""

    def __init__(self, vram_gb: int, message: Optional[str] = None) -> None:
        super().__init__(
            message or f"No Runpod GPU with sufficient VRAM (required >= {vram_gb} GB)",
            status_code=503,
            details={"required_vram_gb": vram_gb},
        )


class RunpodAPIError(OrchestratorError):
    """Raised when Runpod API returns an error."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, status_code=status_code, details=details or {})


class WebhookDeliveryError(OrchestratorError):
    """Raised when user webhook delivery fails after retries."""

    def __init__(self, url: str, message: Optional[str] = None) -> None:
        super().__init__(
            message or f"Webhook delivery failed after retries: {url}",
            status_code=502,
            details={"webhook_url": url},
        )


class DeploymentNotFoundError(OrchestratorError):
    """Raised when deployment ID is not found."""

    def __init__(self, deployment_id: str) -> None:
        super().__init__(
            f"Deployment not found: {deployment_id}",
            status_code=404,
            details={"deployment_id": deployment_id},
        )


class UnauthorizedError(OrchestratorError):
    """Raised when API key is missing or invalid."""

    def __init__(self, message: str = "Invalid or missing API key") -> None:
        super().__init__(message, status_code=401)


class RateLimitError(OrchestratorError):
    """Raised when rate limit (100 req/min per API key) is exceeded."""

    def __init__(self, retry_after_seconds: int = 60) -> None:
        super().__init__(
            "Rate limit exceeded. Try again later.",
            status_code=429,
            details={"retry_after_seconds": retry_after_seconds},
        )


class InvalidDeploymentRequestError(OrchestratorError):
    """Raised when deployment request has invalid combination of fields (e.g. must provide hf_model_id OR model_name)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)
