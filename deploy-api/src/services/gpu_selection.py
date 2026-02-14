from typing import Optional
from src.core.errors import RunpodInsufficientGPUError # We should rename this to InsufficientGPUError but keeping for compatibility
from src.services.gpu_registry import select_gpu_id_for_vram, gpu_id_to_display_name

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
