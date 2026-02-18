from typing import Optional
from src.core.errors import RunpodInsufficientGPUError # We should rename this to InsufficientGPUError but keeping for compatibility
from src.services.gpu_registry import gpu_id_to_display_name, select_gpu_candidates_for_vram, select_gpu_id_for_vram

def select_gpu(
    vram_gb: int,
    gpu_tier: Optional[str] = None,
    registry: Optional[list] = None,
    tier_mapping: Optional[dict] = None,
) -> tuple[str, str]:
    """
    Common GPU selection wrapper.
    Returns (gpu_id, display_name).
    """
    gpu_id = select_gpu_id_for_vram(vram_gb, gpu_tier, registry=registry, tier_mapping=tier_mapping)
    if not gpu_id:
        raise RunpodInsufficientGPUError(vram_gb)
    display = gpu_id_to_display_name(gpu_id, registry=registry)
    return gpu_id, display


def select_gpu_candidates(
    vram_gb: int,
    gpu_tier: Optional[str] = None,
    registry: Optional[list] = None,
    tier_mapping: Optional[dict] = None,
) -> list[tuple[str, str]]:
    """Return ordered (gpu_id, display_name) candidates from cheapest to expensive."""
    ids = select_gpu_candidates_for_vram(vram_gb, gpu_tier, registry=registry, tier_mapping=tier_mapping)
    if not ids:
        raise RunpodInsufficientGPUError(vram_gb)
    return [(gpu_id, gpu_id_to_display_name(gpu_id, registry=registry)) for gpu_id in ids]
