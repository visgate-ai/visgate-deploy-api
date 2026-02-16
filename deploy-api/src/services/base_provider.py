from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict

class ProviderEndpoint(TypedDict):
    id: str
    url: str
    raw_response: Any


class ProviderEndpointSummary(TypedDict):
    id: str
    name: str
    status: str
    url: Optional[str]
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
    def get_run_url(self, endpoint_id: str) -> str:
        """Return the public URL to send inference requests to."""
        pass
