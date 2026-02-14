"""Dynamic GPU selection for Runpod serverless with Firestore support and cost-efficiency."""

from typing import Optional, TypedDict

class GPUSpec(TypedDict):
    id: str
    display: str
    vram: int
    cost_index: int  # Qualitative index: 1 (cheapest) to 10 (most expensive)

# FAST FALLBACKS (Used if Firestore is unavailable or for local dev)
DEFAULT_GPU_REGISTRY: list[GPUSpec] = [
    {"id": "AMPERE_16", "display": "NVIDIA A16", "vram": 16, "cost_index": 1},
    {"id": "AMPERE_24", "display": "NVIDIA A10 / A30", "vram": 24, "cost_index": 2},
    {"id": "ADA_24", "display": "NVIDIA L40 / RTX 4090", "vram": 24, "cost_index": 3},
    {"id": "AMPERE_48", "display": "NVIDIA A40", "vram": 48, "cost_index": 5},
    {"id": "ADA_48_PRO", "display": "NVIDIA L40S", "vram": 48, "cost_index": 6},
    {"id": "AMPERE_80", "display": "NVIDIA A100", "vram": 80, "cost_index": 8},
    {"id": "ADA_80_PRO", "display": "NVIDIA H100", "vram": 80, "cost_index": 10},
]

DEFAULT_TIER_MAPPING: dict[str, list[str]] = {
    "ECONOMY": ["AMPERE_16", "AMPERE_24"],
    "STANDARD": ["ADA_24", "AMPERE_24"],
    "PRO": ["AMPERE_48", "ADA_48_PRO"],
    "ULTIMATE": ["AMPERE_80", "ADA_80_PRO"],
    "A16": ["AMPERE_16"],
    "A10": ["AMPERE_24"],
    "A40": ["AMPERE_48"],
    "A100": ["AMPERE_80"],
    "H100": ["ADA_80_PRO"],
    "4090": ["ADA_24"],
}

def select_gpu_id_for_vram(
    vram_gb: int, 
    gpu_tier: Optional[str] = None,
    registry: Optional[list[GPUSpec]] = None,
    tier_mapping: Optional[dict[str, list[str]]] = None
) -> Optional[str]:
    """
    Select optimal GPU. Falls back to static defaults if registry/tier_mapping aren't provided (e.g. from Firestore).
    """
    reg = registry if registry else DEFAULT_GPU_REGISTRY
    mapping = tier_mapping if tier_mapping else DEFAULT_TIER_MAPPING

    # 1. Resolve Tier Candidates
    tier_candidates: list[str] = []
    if gpu_tier:
        normalized = gpu_tier.strip().upper()
        tier_candidates = mapping.get(normalized, [])

    # 2. Filter Registry by VRAM
    # Sort by cost_index (cheapest first) then by vram (narrowest fit first)
    sorted_registry = sorted(reg, key=lambda x: (x.get("cost_index", 5), x.get("vram", 0)))

    # Priority 1: Match within user-specified tier
    if tier_candidates:
        for gpu in sorted_registry:
            if gpu["id"] in tier_candidates and gpu["vram"] >= vram_gb:
                return gpu["id"]

    # Priority 2: Absolute cheapest match (regardless of tier)
    for gpu in sorted_registry:
        if gpu["vram"] >= vram_gb:
            return gpu["id"]

    return None

def gpu_id_to_display_name(gpu_id: str, registry: Optional[list[GPUSpec]] = None) -> str:
    """Resolve display name from registry or default."""
    reg = registry if registry else DEFAULT_GPU_REGISTRY
    for gpu in reg:
        if gpu["id"] == gpu_id:
            return gpu["display"]
    return f"NVIDIA {gpu_id}"

def get_gpu_vram(gpu_id: str, registry: Optional[list[GPUSpec]] = None) -> int:
    """Get VRAM for a specific GPU ID."""
    reg = registry if registry else DEFAULT_GPU_REGISTRY
    for gpu in reg:
        if gpu["id"] == gpu_id:
            return gpu.get("vram", 0)
    return 0
