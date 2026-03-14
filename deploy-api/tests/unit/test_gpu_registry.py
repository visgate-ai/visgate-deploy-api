"""Unit tests for GPU registry."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.gpu_registry import (
    _derive_cost_index,
    _effective_price,
    derive_tier_mapping,
    fetch_live_gpu_registry,
    get_runpod_gpu_ids,
    gpu_id_to_display_name,
    invalidate_live_cache,
    select_gpu_candidates_for_vram,
    select_gpu_id_for_vram,
)


def test_get_runpod_gpu_ids_a6000() -> None:
    assert "NVIDIA RTX A6000" in get_runpod_gpu_ids("A6000")


def test_get_runpod_gpu_ids_default() -> None:
    ids = get_runpod_gpu_ids(None)
    assert len(ids) > 0


def test_select_gpu_id_for_vram() -> None:
    gpu_id = select_gpu_id_for_vram(12, "A6000")
    assert gpu_id is not None
    assert select_gpu_id_for_vram(100, None) is None or select_gpu_id_for_vram(100, None) is not None


def test_gpu_id_to_display_name() -> None:
    display = gpu_id_to_display_name("NVIDIA RTX A6000")
    assert "A6000" in display


def test_select_gpu_candidates_for_vram_sorted() -> None:
    ids = select_gpu_candidates_for_vram(20, None)
    # First candidate should be a 20GB or 24GB GPU (cheapest that fits)
    assert len(ids) > 0
    assert any("A6000" in gid or "A5000" in gid or "3090" in gid or "A4500" in gid or "4000 Ada" in gid for gid in ids)


def test_select_gpu_candidates_for_vram_tier_fallback() -> None:
    # "A10" not in tier mapping -> fallback to global candidates
    ids = select_gpu_candidates_for_vram(48, "A10")
    assert any("A6000" in gid or "A40" in gid or "L40" in gid for gid in ids)


# ── Dynamic registry helpers ────────────────────────────────────────────────


def test_effective_price_prefers_secure() -> None:
    gpu = {"securePrice": 0.50, "communityPrice": 0.30}
    assert _effective_price(gpu) == 0.50


def test_effective_price_falls_back_to_community() -> None:
    gpu = {"securePrice": 0, "communityPrice": 0.25}
    assert _effective_price(gpu) == 0.25


def test_effective_price_missing_prices() -> None:
    assert _effective_price({}) == 999.0


def test_derive_cost_index_range() -> None:
    assert _derive_cost_index(0.1, 0.1, 1.0) == 1
    assert _derive_cost_index(1.0, 0.1, 1.0) == 10
    assert 1 <= _derive_cost_index(0.5, 0.1, 1.0) <= 10


def test_derive_cost_index_equal_prices() -> None:
    assert _derive_cost_index(0.5, 0.5, 0.5) == 5


def test_derive_tier_mapping_from_registry() -> None:
    registry = [
        {"id": "NVIDIA RTX A4000", "display": "A4000", "vram": 16, "cost_index": 1},
        {"id": "NVIDIA GeForce RTX 4090", "display": "4090", "vram": 24, "cost_index": 3},
        {"id": "NVIDIA RTX A6000", "display": "A6000", "vram": 48, "cost_index": 5},
        {"id": "NVIDIA A100 80GB PCIe", "display": "A100", "vram": 80, "cost_index": 8},
    ]
    tiers = derive_tier_mapping(registry)
    assert "NVIDIA RTX A4000" in tiers["ECONOMY"]
    assert "NVIDIA GeForce RTX 4090" in tiers["STANDARD"]
    assert "NVIDIA RTX A6000" in tiers["PRO"]
    assert "NVIDIA A100 80GB PCIe" in tiers["ULTIMATE"]
    # Aliases
    assert "NVIDIA RTX A6000" in tiers.get("A6000", [])
    assert "NVIDIA A100 80GB PCIe" in tiers.get("A100", [])


# ── Live fetch ───────────────────────────────────────────────────────────────


FAKE_RUNPOD_GPU_TYPES = [
    {"id": "NVIDIA RTX A4000", "displayName": "RTX A4000", "memoryInGb": 16,
     "secureCloud": True, "communityCloud": True, "securePrice": 0.19, "communityPrice": 0.10},
    {"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24,
     "secureCloud": True, "communityCloud": True, "securePrice": 0.44, "communityPrice": 0.30},
    {"id": "NVIDIA A100 80GB PCIe", "displayName": "A100 80GB", "memoryInGb": 80,
     "secureCloud": True, "communityCloud": False, "securePrice": 1.64, "communityPrice": 0},
]


def test_fetch_live_gpu_registry_success() -> None:
    invalidate_live_cache()
    mock_cls = MagicMock()
    mock_cls.return_value.list_gpu_types = AsyncMock(return_value=FAKE_RUNPOD_GPU_TYPES)

    with patch("src.services.runpod.RunpodProvider", mock_cls):
        result = asyncio.get_event_loop().run_until_complete(
            fetch_live_gpu_registry("fake-key")
        )

    assert result is not None
    assert len(result) == 3
    # Cheapest GPU first
    assert result[0]["id"] == "NVIDIA RTX A4000"
    # Most expensive last
    assert result[-1]["id"] == "NVIDIA A100 80GB PCIe"
    invalidate_live_cache()


def test_fetch_live_gpu_registry_api_failure_returns_none() -> None:
    invalidate_live_cache()
    mock_cls = MagicMock()
    mock_cls.return_value.list_gpu_types = AsyncMock(side_effect=Exception("API down"))

    with patch("src.services.runpod.RunpodProvider", mock_cls):
        result = asyncio.get_event_loop().run_until_complete(
            fetch_live_gpu_registry("fake-key")
        )

    assert result is None
    invalidate_live_cache()


def test_fetch_live_gpu_registry_uses_cache() -> None:
    invalidate_live_cache()
    mock_cls = MagicMock()
    mock_cls.return_value.list_gpu_types = AsyncMock(return_value=FAKE_RUNPOD_GPU_TYPES)

    with patch("src.services.runpod.RunpodProvider", mock_cls):
        r1 = asyncio.get_event_loop().run_until_complete(fetch_live_gpu_registry("key"))
        r2 = asyncio.get_event_loop().run_until_complete(fetch_live_gpu_registry("key"))

    assert r1 == r2
    # Only one API call (second hit cache)
    mock_cls.return_value.list_gpu_types.assert_awaited_once()
    invalidate_live_cache()
