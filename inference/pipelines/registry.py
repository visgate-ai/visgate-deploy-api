"""Map HF model ID to pipeline class and load the right pipeline."""

import re
from typing import Optional, Type

from pipelines.base import BasePipeline
from pipelines.flux import FluxPipeline
from pipelines.sdxl import SDXLPipeline


# Order matters: first match wins. Pattern (regex or prefix) -> pipeline class
MODEL_REGISTRY: list[tuple[str, Type[BasePipeline]]] = [
    # Flux (black-forest-labs)
    (r"black-forest-labs/FLUX\.1", FluxPipeline),
    # SDXL / Stability
    (r"stabilityai/sdxl", SDXLPipeline),
    (r"stabilityai/stable-diffusion", SDXLPipeline),
    # Generic diffusers text2image
    (r"runwayml/stable-diffusion", SDXLPipeline),
    (r"CompVis/stable-diffusion", SDXLPipeline),
]


def get_pipeline_for_model(model_id: str) -> Type[BasePipeline]:
    """Return the pipeline class for the given HF model ID."""
    if not model_id or not model_id.strip():
        raise ValueError("model_id is required")
    model_id_lower = model_id.strip().lower()
    for pattern, pipeline_cls in MODEL_REGISTRY:
        if re.search(pattern, model_id, re.IGNORECASE):
            return pipeline_cls
    # Default: try Flux first (common), then SDXL
    if "flux" in model_id_lower or "black-forest" in model_id_lower:
        return FluxPipeline
    return SDXLPipeline


def load_pipeline(
    model_id: str,
    token: Optional[str] = None,
    device: str = "cuda",
) -> BasePipeline:
    """Load and initialize the appropriate pipeline for the model."""
    pipeline_cls = get_pipeline_for_model(model_id)
    pipeline = pipeline_cls(model_id=model_id, token=token, device=device)
    pipeline.load()
    return pipeline
