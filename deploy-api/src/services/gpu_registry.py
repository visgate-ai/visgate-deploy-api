"""Dynamic GPU selection for Runpod serverless with cost-efficiency.

GPU IDs MUST match RunPod's actual ``gpuTypes.id`` values exactly.  These are the
real machine identifiers returned by the RunPod GraphQL API (query ``gpuTypes``).
Using abstract/symbolic IDs will cause RunPod to fail to schedule workers.

The registry can be populated dynamically from the RunPod API via
``fetch_live_gpu_registry()`` (TTL-cached) so that new GPU types, pricing
changes, and availability updates are picked up automatically without code
changes.  ``DEFAULT_GPU_REGISTRY`` serves as a static fallback when the API
is unreachable.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, TypedDict

from src.core.logging import structured_log


class GPUSpec(TypedDict):
    id: str
    display: str
    vram: int
    cost_index: int  # Qualitative index: 1 (cheapest) to 10 (most expensive)

# ── Real RunPod GPU IDs ──────────────────────────────────────────────────────
# Prices sourced from RunPod serverless pricing (March 2026).
# Ordered by VRAM tier then ascending cost_index for the selection algorithm.
DEFAULT_GPU_REGISTRY: list[GPUSpec] = [
    # 16 GB
    {"id": "NVIDIA RTX A4000",                      "display": "RTX A4000 16GB",       "vram": 16, "cost_index": 1},
    {"id": "NVIDIA GeForce RTX 4080",               "display": "RTX 4080 16GB",        "vram": 16, "cost_index": 2},
    {"id": "NVIDIA GeForce RTX 4080 SUPER",         "display": "RTX 4080S 16GB",       "vram": 16, "cost_index": 2},
    # 20 GB
    {"id": "NVIDIA RTX A4500",                      "display": "RTX A4500 20GB",       "vram": 20, "cost_index": 2},
    {"id": "NVIDIA RTX 4000 Ada Generation",        "display": "RTX 4000 Ada 20GB",    "vram": 20, "cost_index": 2},
    # 24 GB
    {"id": "NVIDIA RTX A5000",                      "display": "RTX A5000 24GB",       "vram": 24, "cost_index": 2},
    {"id": "NVIDIA GeForce RTX 3090",               "display": "RTX 3090 24GB",        "vram": 24, "cost_index": 2},
    {"id": "NVIDIA GeForce RTX 4090",               "display": "RTX 4090 24GB",        "vram": 24, "cost_index": 3},
    {"id": "NVIDIA L4",                             "display": "L4 24GB",              "vram": 24, "cost_index": 4},
    # 32 GB
    {"id": "NVIDIA RTX 5000 Ada Generation",        "display": "RTX 5000 Ada 32GB",    "vram": 32, "cost_index": 5},
    {"id": "NVIDIA GeForce RTX 5090",               "display": "RTX 5090 32GB",        "vram": 32, "cost_index": 6},
    # 48 GB
    {"id": "NVIDIA RTX A6000",                      "display": "RTX A6000 48GB",       "vram": 48, "cost_index": 4},
    {"id": "NVIDIA A40",                            "display": "A40 48GB",             "vram": 48, "cost_index": 4},
    {"id": "NVIDIA L40",                            "display": "L40 48GB",             "vram": 48, "cost_index": 6},
    {"id": "NVIDIA L40S",                           "display": "L40S 48GB",            "vram": 48, "cost_index": 7},
    {"id": "NVIDIA RTX 6000 Ada Generation",        "display": "RTX 6000 Ada 48GB",    "vram": 48, "cost_index": 7},
    # 80 GB
    {"id": "NVIDIA A100 80GB PCIe",                 "display": "A100 80GB PCIe",       "vram": 80, "cost_index": 8},
    {"id": "NVIDIA A100-SXM4-80GB",                 "display": "A100 80GB SXM",        "vram": 80, "cost_index": 8},
    {"id": "NVIDIA H100 PCIe",                      "display": "H100 80GB PCIe",       "vram": 80, "cost_index": 9},
    {"id": "NVIDIA H100 80GB HBM3",                 "display": "H100 80GB SXM",        "vram": 80, "cost_index": 10},
]

DEFAULT_TIER_MAPPING: dict[str, list[str]] = {
    "ECONOMY": [
        "NVIDIA RTX A4000",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA GeForce RTX 4080 SUPER",
        "NVIDIA RTX A4500",
        "NVIDIA RTX 4000 Ada Generation",
    ],
    "STANDARD": [
        "NVIDIA RTX A5000",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA L4",
    ],
    "PRO": [
        "NVIDIA RTX A6000",
        "NVIDIA A40",
        "NVIDIA L40",
        "NVIDIA L40S",
        "NVIDIA RTX 6000 Ada Generation",
    ],
    "ULTIMATE": [
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA H100 PCIe",
        "NVIDIA H100 80GB HBM3",
    ],
    # Direct GPU aliases
    "A4000": ["NVIDIA RTX A4000"],
    "3090":  ["NVIDIA GeForce RTX 3090"],
    "4090":  ["NVIDIA GeForce RTX 4090"],
    "A5000": ["NVIDIA RTX A5000"],
    "A6000": ["NVIDIA RTX A6000"],
    "A40":   ["NVIDIA A40"],
    "L40S":  ["NVIDIA L40S"],
    "A100":  ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
    "H100":  ["NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"],
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


# ── Live RunPod API Registry (TTL-cached) ───────────────────────────────────

_CACHE_TTL_SECONDS = 300  # 5 minutes

_live_cache: dict[str, Any] = {
    "registry": None,
    "fetched_at": 0.0,
}
_cache_lock = asyncio.Lock()


def _effective_price(gpu: dict[str, Any]) -> float:
    """Pick the best available price for cost ranking (secure preferred)."""
    sp = gpu.get("securePrice") or 0
    cp = gpu.get("communityPrice") or 0
    if sp and sp > 0:
        return float(sp)
    if cp and cp > 0:
        return float(cp)
    return 999.0


def _derive_cost_index(price: float, p_min: float, p_max: float) -> int:
    """Map a price to a 1–10 qualitative cost index via linear interpolation."""
    if p_max <= p_min:
        return 5
    ratio = (price - p_min) / (p_max - p_min)
    return max(1, min(10, int(ratio * 9) + 1))


def _runpod_gpu_to_spec(gpu: dict[str, Any], cost_index: int) -> GPUSpec:
    # Penalize community-only GPUs: they're cheap but rarely provision on serverless
    if not gpu.get("secureCloud"):
        cost_index = min(10, cost_index + 3)
    return {
        "id": gpu["id"],
        "display": gpu.get("displayName") or gpu["id"],
        "vram": int(gpu.get("memoryInGb") or 0),
        "cost_index": cost_index,
    }


async def fetch_live_gpu_registry(api_key: str) -> list[GPUSpec] | None:
    """Fetch GPU types from RunPod API with TTL cache.

    Returns a list of ``GPUSpec`` sorted by (cost_index, vram), or ``None``
    when the API is unreachable so the caller can fall back to the static
    ``DEFAULT_GPU_REGISTRY``.
    """
    now = time.monotonic()

    # Fast path: cached and fresh
    cached = _live_cache["registry"]
    if cached and (now - _live_cache["fetched_at"]) < _CACHE_TTL_SECONDS:
        return cached

    async with _cache_lock:
        # Double-check after acquiring lock (another coroutine may have refreshed)
        if _live_cache["registry"] and (time.monotonic() - _live_cache["fetched_at"]) < _CACHE_TTL_SECONDS:
            return _live_cache["registry"]

        try:
            from src.services.runpod import RunpodProvider

            raw_gpus = await RunpodProvider().list_gpu_types(api_key)
            if not raw_gpus:
                return None

            # Keep only GPUs that have VRAM info and are available somewhere
            valid = [
                g for g in raw_gpus
                if (g.get("memoryInGb") or 0) > 0
                and (g.get("secureCloud") or g.get("communityCloud"))
            ]
            if not valid:
                return None

            prices = [_effective_price(g) for g in valid]
            p_min, p_max = min(prices), max(prices)

            registry: list[GPUSpec] = []
            for g, price in zip(valid, prices):
                ci = _derive_cost_index(price, p_min, p_max)
                registry.append(_runpod_gpu_to_spec(g, ci))

            registry.sort(key=lambda x: (x["cost_index"], x["vram"]))

            _live_cache["registry"] = registry
            _live_cache["fetched_at"] = time.monotonic()

            structured_log(
                "INFO",
                "Refreshed live GPU registry from RunPod API",
                operation="gpu_registry.live_fetch",
                metadata={"gpu_count": len(registry)},
            )
            return registry

        except Exception as exc:
            structured_log(
                "WARNING",
                f"Failed to fetch live GPU registry, will use static fallback: {exc}",
                operation="gpu_registry.live_fetch",
            )
            return None


def derive_tier_mapping(registry: list[GPUSpec]) -> dict[str, list[str]]:
    """Auto-derive tier + alias mapping from a registry based on VRAM ranges."""
    tiers: dict[str, list[str]] = {
        "ECONOMY": [],
        "STANDARD": [],
        "PRO": [],
        "ULTIMATE": [],
    }

    for gpu in registry:
        vram = gpu["vram"]
        if vram <= 16:
            tiers["ECONOMY"].append(gpu["id"])
        elif vram <= 24:
            tiers["STANDARD"].append(gpu["id"])
        elif vram <= 48:
            tiers["PRO"].append(gpu["id"])
        else:
            tiers["ULTIMATE"].append(gpu["id"])

    # Direct GPU aliases (match by substring in GPU id)
    _ALIAS_PATTERNS: dict[str, str] = {
        "A4000": "RTX A4000",
        "3090": "RTX 3090",
        "4090": "RTX 4090",
        "A5000": "RTX A5000",
        "A6000": "RTX A6000",
        "A40": " A40",
        "L40S": "L40S",
        "A100": "A100",
        "H100": "H100",
    }
    for alias, pattern in _ALIAS_PATTERNS.items():
        matching = [gpu["id"] for gpu in registry if pattern in gpu["id"]]
        if matching:
            tiers[alias] = matching

    return tiers


def invalidate_live_cache() -> None:
    """Force next ``fetch_live_gpu_registry`` call to refresh. Mainly for tests."""
    _live_cache["registry"] = None
    _live_cache["fetched_at"] = 0.0
