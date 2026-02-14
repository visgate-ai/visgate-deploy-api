"""Map user GPU tier (e.g. A40) to Runpod gpuIds and VRAM."""

# Runpod gpuIds from docs: AMPERE_16, AMPERE_24, ADA_24, AMPERE_48, ADA_48_PRO, AMPERE_80, ADA_80_PRO
# Map user-facing tier -> Runpod gpuIds (comma-separated for alternatives)
GPU_TIER_TO_RUNPOD: dict[str, list[str]] = {
    "A40": ["AMPERE_48"],       # 48GB
    "A100": ["AMPERE_80"],      # 80GB
    "A100-40": ["AMPERE_48"],
    "A10": ["AMPERE_24"],
    "L40": ["ADA_24"],
    "L4": ["ADA_24"],
    "RTX4090": ["ADA_24"],
    "default": ["AMPERE_24", "ADA_24", "AMPERE_16"],
}

# Runpod gpuId -> approximate VRAM GB for selection
RUNPOD_GPU_VRAM_GB: dict[str, int] = {
    "AMPERE_16": 16,
    "AMPERE_24": 24,
    "ADA_24": 24,
    "AMPERE_48": 48,
    "ADA_48_PRO": 48,
    "AMPERE_80": 80,
    "ADA_80_PRO": 80,
}


def get_runpod_gpu_ids(gpu_tier: str | None) -> list[str]:
    """Return list of Runpod gpuIds for the given tier; default if tier unknown."""
    if gpu_tier:
        normalized = gpu_tier.strip().upper()
        if normalized in GPU_TIER_TO_RUNPOD:
            return GPU_TIER_TO_RUNPOD[normalized]
    return GPU_TIER_TO_RUNPOD["default"]


def select_gpu_id_for_vram(vram_gb: int, gpu_tier: str | None) -> str | None:
    """Pick a single Runpod gpuId that has at least vram_gb; prefer tier if specified."""
    candidates = get_runpod_gpu_ids(gpu_tier)
    for gpu_id in candidates:
        if RUNPOD_GPU_VRAM_GB.get(gpu_id, 0) >= vram_gb:
            return gpu_id
    # Fallback: any GPU with enough VRAM
    for gpu_id, gb in sorted(RUNPOD_GPU_VRAM_GB.items(), key=lambda x: x[1]):
        if gb >= vram_gb:
            return gpu_id
    return None


def gpu_id_to_display_name(gpu_id: str) -> str:
    """Return human-readable GPU name for response."""
    names = {
        "AMPERE_16": "NVIDIA A16",
        "AMPERE_24": "NVIDIA A10",
        "ADA_24": "NVIDIA L40 / RTX 4090",
        "AMPERE_48": "NVIDIA A40",
        "ADA_48_PRO": "NVIDIA L40S",
        "AMPERE_80": "NVIDIA A100",
        "ADA_80_PRO": "NVIDIA H100",
    }
    return names.get(gpu_id, f"NVIDIA {gpu_id}")
