"""Base interface for inference pipelines."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BasePipeline(ABC):
    """Base class for model pipelines (Flux, SDXL, etc.)."""

    def __init__(self, model_id: str, token: Optional[str] = None, device: str = "cuda") -> None:
        self.model_id = model_id
        self.token = token
        self.device = device
        self._pipeline: Any = None

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (GPU)."""
        pass

    @abstractmethod
    def run(
        self,
        prompt: str,
        *,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run inference. Returns dict with at least:
        - "image_base64": str (or "images": list of base64)
        - "model_id": str
        - "seed": int (if used)
        """
        pass

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None
