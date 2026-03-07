"""Dynamic GPU selection for Runpod serverless with Firestore support and cost-efficiency."""

from typing import TypedDict


class GPUSpec(TypedDict):
    id: str
    display: str
    vram: int
    cost_index: int  # Qualitative index: 1 (cheapest) to 10 (most expensive)

# FAST FALLBACKS (Used if Firestore is unavailable or for local dev)
# vram column = total GPU memory.  GPU selection algorithm picks the cheapest GPU
# whose vram >= model's min_gpu_memory_gb so there is headroom for activations.
DEFAULT_GPU_REGISTRY: list[GPUSpec] = [
    {"id": "AMPERE_16",    "display": "AMPERE 16GB", "vram": 16, "cost_index": 1},
    {"id": "AMPERE_24",    "display": "AMPERE 24GB", "vram": 24, "cost_index": 2},
    {"id": "ADA_24",       "display": "ADA 24GB",    "vram": 24, "cost_index": 3},
    {"id": "AMPERE_48",    "display": "AMPERE 48GB", "vram": 48, "cost_index": 5},
    {"id": "ADA_48_PRO",   "display": "ADA 48GB PRO", "vram": 48, "cost_index": 6},
    {"id": "AMPERE_80",    "display": "A100 80GB",   "vram": 80, "cost_index": 8},
    {"id": "HOPPER_80",    "display": "H100 80GB",   "vram": 80, "cost_index": 10},
]

DEFAULT_TIER_MAPPING: dict[str, list[str]] = {
    "ECONOMY": ["AMPERE_16", "AMPERE_24"],
    "STANDARD": ["AMPERE_24", "ADA_24"],
    "PRO": ["AMPERE_48", "ADA_48_PRO"],
    "ULTIMATE": ["AMPERE_80", "HOPPER_80"],
    "A4000": ["AMPERE_16"],
    "3090": ["AMPERE_24"],
    "4090": ["ADA_24"],
    "A6000": ["AMPERE_48"],
    "A100": ["AMPERE_80"],
    "H100": ["HOPPER_80"],
}

def select_gpu_id_for_vram(
    vram_gb: int,
    gpu_tier: str | None = None,
    registry: list[GPUSpec] | None = None,
    tier_mapping: dict[str, list[str]] | None = None
) -> str | None:
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


def select_gpu_candidates_for_vram(
    vram_gb: int,
    gpu_tier: str | None = None,
    registry: list[GPUSpec] | None = None,
    tier_mapping: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return candidate GPU IDs sorted by lowest cost that satisfy VRAM and optional tier."""
    reg = registry if registry else DEFAULT_GPU_REGISTRY
    mapping = tier_mapping if tier_mapping else DEFAULT_TIER_MAPPING

    tier_candidates: list[str] = []
    if gpu_tier:
        normalized = gpu_tier.strip().upper()
        tier_candidates = mapping.get(normalized, [])

    sorted_registry = sorted(reg, key=lambda x: (x.get("cost_index", 5), x.get("vram", 0)))
    result: list[str] = []
    for gpu in sorted_registry:
        if gpu.get("vram", 0) < vram_gb:
            continue
        gpu_id = gpu["id"]
        if tier_candidates and gpu_id not in tier_candidates:
            continue
        result.append(gpu_id)

    # If strict tier filtering produced no match, gracefully fallback to global candidates.
    if not result and tier_candidates:
        for gpu in sorted_registry:
            if gpu.get("vram", 0) >= vram_gb:
                result.append(gpu["id"])
    return result


def get_runpod_gpu_ids(gpu_tier: str | None) -> list[str]:
    """Return GPU IDs for a tier (used by tests and compatibility checks)."""
    if not gpu_tier:
        return list({gpu["id"] for gpu in DEFAULT_GPU_REGISTRY})
    normalized = gpu_tier.strip().upper()
    return DEFAULT_TIER_MAPPING.get(normalized, [])

def gpu_id_to_display_name(gpu_id: str, registry: list[GPUSpec] | None = None) -> str:
    """Resolve display name from registry or default."""
    reg = registry if registry else DEFAULT_GPU_REGISTRY
    for gpu in reg:
        if gpu["id"] == gpu_id:
            return gpu["display"]
    return f"NVIDIA {gpu_id}"

def get_gpu_vram(gpu_id: str, registry: list[GPUSpec] | None = None) -> int:
    """Get VRAM for a specific GPU ID."""
    reg = registry if registry else DEFAULT_GPU_REGISTRY
    for gpu in reg:
        if gpu["id"] == gpu_id:
            return gpu.get("vram", 0)
    return 0
