"""Hugging Face model specs registry: VRAM requirements and capabilities.

All gpu_memory_gb values represent the *minimum GPU memory required to run the model*,
not the raw model weight size.  This is what the GPU selection algorithm uses for
choosing the cheapest GPU that will actually fit the model without OOM.
"""

from __future__ import annotations

from typing import Any

# Model ID -> specs
# gpu_memory_gb: minimum GPU VRAM in GB required (includes weights + activations + overhead)
MODEL_SPECS_REGISTRY: dict[str, dict[str, Any]] = {
    # ── Flux family ──────────────────────────────────────────────────────────
    "black-forest-labs/FLUX.1-schnell": {
        "gpu_memory_gb": 16,   # 12 GB weights + ~4 GB activation headroom
        "tasks": ["text2img"],
    },
    "black-forest-labs/FLUX.1-dev": {
        "gpu_memory_gb": 28,   # 24 GB weights + overhead; AMPERE_24 OOMs
        "tasks": ["text2img"],
    },
    # ── SDXL family ──────────────────────────────────────────────────────────
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "gpu_memory_gb": 12,
        "tasks": ["text2img", "image2img"],
    },
    "stabilityai/sdxl-turbo": {
        "gpu_memory_gb": 10,
        "tasks": ["text2img", "image2img"],
    },
    # ── SD-Turbo / SD 2.x / SD 1.x ──────────────────────────────────────────
    "stabilityai/sd-turbo": {
        "gpu_memory_gb": 8,
        "tasks": ["text2img", "image2img"],
    },
    "stabilityai/stable-diffusion-2-1": {
        "gpu_memory_gb": 8,
        "tasks": ["text2img", "image2img"],
    },
    "runwayml/stable-diffusion-v1-5": {
        "gpu_memory_gb": 6,
        "tasks": ["text2img", "image2img"],
    },
    # ── SD 3.x ───────────────────────────────────────────────────────────────
    "stabilityai/stable-diffusion-3-medium-diffusers": {
        "gpu_memory_gb": 18,
        "tasks": ["text2img"],
    },
    "stabilityai/stable-diffusion-3.5-large": {
        "gpu_memory_gb": 40,
        "tasks": ["text2img"],
    },
    "stabilityai/stable-diffusion-3.5-large-turbo": {
        "gpu_memory_gb": 40,
        "tasks": ["text2img"],
    },
    "stabilityai/stable-diffusion-3.5-medium": {
        "gpu_memory_gb": 18,
        "tasks": ["text2img"],
    },
    # ── PixArt ───────────────────────────────────────────────────────────────
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": {
        "gpu_memory_gb": 18,
        "tasks": ["text2img"],
    },
    # ── Kandinsky ────────────────────────────────────────────────────────────
    "kandinsky-community/kandinsky-2-2-decoder": {
        "gpu_memory_gb": 10,
        "tasks": ["text2img", "image2img"],
    },
    # ── IF (DeepFloyd) ───────────────────────────────────────────────────────
    "DeepFloyd/IF-I-XL-v1.0": {
        "gpu_memory_gb": 40,
        "tasks": ["text2img"],
    },
    # ── Wan Video ────────────────────────────────────────────────────────────
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": {
        "gpu_memory_gb": 80,
        "tasks": ["text2video"],
    },
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {
        "gpu_memory_gb": 16,
        "tasks": ["text2video"],
    },
    # ── CogVideoX ────────────────────────────────────────────────────────────
    "THUDM/CogVideoX-5b": {
        "gpu_memory_gb": 48,
        "tasks": ["text2video"],
    },
}


# dtype string → bytes per parameter
# Covers all dtypes returned by huggingface_hub SafeTensorsInfo.parameters
_BYTES_PER_PARAM: dict[str, float] = {
    "BF16":    2,   "F16":  2, "F32":  4, "F64":  8,
    "I64":     8,   "I32":  4, "I16":  2, "I8":   1,
    "U8":      1,   "U16":  2, "U32":  4, "U64":  8,
    "F8_E4M3": 1,  "F8_E5M2": 1,
}

# GPU tier snap table (GB) — we always round up to a real tier
_GPU_TIERS: tuple[int, ...] = (6, 8, 10, 12, 16, 24, 28, 40, 48, 80)

# Multiplier applied to raw weight bytes to estimate total VRAM needed.
# Covers activations, KV-cache, framework buffers, CUDA context.
# 1.35 is safe for diffusion models with attention slicing enabled.
_ACTIVATION_MULTIPLIER = 1.35


def _snap_to_gpu_tier(vram_gb: float) -> int:
    """Round up to the nearest real GPU tier size."""
    import math
    for tier in _GPU_TIERS:
        if tier >= math.ceil(vram_gb):
            return tier
    return _GPU_TIERS[-1]


def _estimate_vram_from_weight_bytes(weight_bytes: int) -> int:
    """
    Estimate minimum GPU memory in GB from raw model weight bytes.

    Formula::

        weight_gb = weight_bytes / 1024³
        min_vram  = snap_to_tier(weight_gb × 1.35)

    Example: FLUX.1-schnell BF16 11.9 B params → 22.1 GB weight
             → 22.1 × 1.35 = 29.8 → snapped to 40 GB tier.
    """
    weight_gb = weight_bytes / (1024 ** 3)
    return _snap_to_gpu_tier(weight_gb * _ACTIVATION_MULTIPLIER)


# Rough parameter-count → minimum GPU memory table (fallback when dtype unknown).
_PARAM_TO_VRAM: list[tuple[int, int]] = [
    #  params (M)  min_vram_gb
    (500,         6),
    (1_000,       8),
    (3_000,      12),
    (7_000,      16),
    (13_000,     24),
    (30_000,     40),
    (70_000,     80),
]


def _estimate_vram_from_params(params_millions: int) -> int:
    """Return minimum GPU memory estimate given parameter count in millions (no dtype info)."""
    for threshold, vram in _PARAM_TO_VRAM:
        if params_millions <= threshold:
            return vram
    return 80  # >70 B parameters → H100 class


def get_model_specs(model_id: str) -> dict[str, Any] | None:
    """Return registered specs for a model, or None if not in registry."""
    return MODEL_SPECS_REGISTRY.get(model_id)


def get_min_gpu_memory_gb(
    model_id: str,
    hf_params_millions: int | None = None,
    hf_weight_bytes: int | None = None,
) -> int:
    """Return minimum GPU memory required for the model in GB.

    Priority:
    1. Registry entry  — most accurate, hand-tuned per model
    2. Byte-based estimate — ``hf_weight_bytes`` from ``safetensors.parameters``
       gives true weight size per dtype → ``× 1.35`` headroom → tier snap
    3. Param-count estimate — coarser fallback when dtype is unavailable
    4. Conservative default: 16 GB
    """
    spec = get_model_specs(model_id)
    if spec:
        return int(spec.get("gpu_memory_gb", 16))
    if hf_weight_bytes is not None and hf_weight_bytes > 0:
        return _estimate_vram_from_weight_bytes(hf_weight_bytes)
    if hf_params_millions is not None:
        return _estimate_vram_from_params(hf_params_millions)
    return 16  # safer default than 12: avoids wasting a cold start on OOM


def get_vram_gb(model_id: str) -> int:
    """Alias for backwards compatibility.  Delegates to get_min_gpu_memory_gb."""
    return get_min_gpu_memory_gb(model_id)
