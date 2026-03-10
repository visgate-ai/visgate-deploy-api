from abc import ABC, abstractmethod
from typing import Any, TypedDict


class ProviderEndpoint(TypedDict):
    id: str
    url: str
    raw_response: Any


class ProviderEndpointSummary(TypedDict):
    id: str
    name: str
    status: str
    url: str | None
    raw_response: Any


class ProviderJobAccepted(TypedDict):
    id: str
    status: str
    raw_response: Any


class ProviderJobStatus(TypedDict, total=False):
    id: str
    status: str
    output: Any
    error: Any
    delay_time: int | None
    execution_time: int | None
    cost_usd: float | None
    raw_response: Any

class BaseInferenceProvider(ABC):
    """
    Abstract base class for inference providers (Runpod, Vast.ai, Lambda, etc.)
    Ensures the orchestrator doesn't care WHO is running the pod.
    """

    @abstractmethod
    async def create_endpoint(
        self,
        name: str,
        gpu_id: str,
        image: str,
        env: dict[str, str],
        api_key: str,
        **kwargs: Any
    ) -> ProviderEndpoint:
        """Provision a new serverless/pod endpoint."""
        pass

    @abstractmethod
    async def delete_endpoint(self, endpoint_id: str, api_key: str) -> None:
        """Tear down an endpoint."""
        pass

    @abstractmethod
    async def list_endpoints(self, api_key: str) -> list[ProviderEndpointSummary]:
        """List endpoints for the current account."""
        pass

    @abstractmethod
    async def list_gpu_types(self, api_key: str) -> list[dict[str, Any]]:
        """List available GPU types and pricing."""
        pass

    @abstractmethod
    def get_run_url(self, endpoint_id: str) -> str:
        """Return the public URL to send inference requests to."""
        pass

    @abstractmethod
    async def submit_job(
        self,
        endpoint_url: str,
        api_key: str,
        job_input: dict[str, Any],
        *,
        webhook_url: str | None = None,
        policy: dict[str, Any] | None = None,
        s3_config: dict[str, Any] | None = None,
    ) -> ProviderJobAccepted:
        """Submit an async inference job."""
        pass

    @abstractmethod
    async def get_job_status(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        """Fetch provider job status."""
        pass

    @abstractmethod
    async def cancel_job(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        """Cancel a provider job."""
        pass

    @abstractmethod
    async def retry_job(self, endpoint_url: str, job_id: str, api_key: str) -> ProviderJobStatus:
        """Retry a failed provider job."""
        pass

    @abstractmethod
    async def get_endpoint_health(self, endpoint_url: str, api_key: str) -> dict[str, Any]:
        """Return provider endpoint health information."""
        pass

    @abstractmethod
    async def check_endpoint_health(self, endpoint_id: str, api_key: str) -> dict[str, Any]:
        """Return provider endpoint health information by endpoint id."""
        pass
