"""Services: Providers, Hugging Face, Webhook, GPU selection."""

from src.services.huggingface import ModelInfo, validate_model
from src.services.webhook import notify
from src.services.gpu_selection import select_gpu
from src.services.provider_factory import get_provider
from src.services.gpu_registry import (
    select_gpu_id_for_vram,
    gpu_id_to_display_name,
)
from src.services.model_resolver import get_hf_name, UnknownModelError

__all__ = [
    "ModelInfo",
    "validate_model",
    "select_gpu",
    "notify",
    "get_provider",
    "select_gpu_id_for_vram",
    "gpu_id_to_display_name",
    "get_hf_name",
    "UnknownModelError",
]
