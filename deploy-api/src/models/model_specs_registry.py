"""Initial seed data: Hugging Face model specs (VRAM, GPU, precision)."""

from typing import Any

# Model ID -> specs for GPU selection and response metadata
MODEL_SPECS_REGISTRY: dict[str, dict[str, Any]] = {
    "black-forest-labs/FLUX.1-schnell": {
        "vram_gb": 12,
        "min_gpu_memory": 14,
        "supported_precisions": ["float16"],
        "average_inference_time_a40": 2.1,
        "tasks": ["text2img"],
    },
    "black-forest-labs/FLUX.1-dev": {
        "vram_gb": 24,
        "min_gpu_memory": 26,
        "supported_precisions": ["float16"],
        "average_inference_time_a100": 4.5,
        "tasks": ["text2img"],
    },
    "stabilityai/sdxl-turbo": {
        "vram_gb": 8,
        "min_gpu_memory": 10,
        "supported_precisions": ["float16"],
        "average_inference_time_a40": 1.2,
        "tasks": ["text2img", "image2img"],
    },
}


def get_model_specs(model_id: str) -> dict[str, Any] | None:
    """Return registered specs for a model, or None if not in registry."""
    return MODEL_SPECS_REGISTRY.get(model_id)


def get_vram_gb(model_id: str) -> int:
    """Return required VRAM in GB for the model; default 12 if unknown."""
    spec = get_model_specs(model_id)
    if spec and "vram_gb" in spec:
        return int(spec["vram_gb"])
    return 12
