"""Unit tests for models and registry."""

import pytest
from pydantic import ValidationError

from src.models.model_specs_registry import (
    _BYTES_PER_PARAM,
    _estimate_vram_from_weight_bytes,
    get_min_gpu_memory_gb,
    get_model_specs,
    get_vram_gb,
    MODEL_SPECS_REGISTRY,
)
from src.models.schemas import DeploymentCreate, DeploymentResponse202, LogEntrySchema
from src.models.entities import DeploymentDoc, LogEntry


def test_model_specs_registry_has_seed_models() -> None:
    assert "black-forest-labs/FLUX.1-schnell" in MODEL_SPECS_REGISTRY
    assert "stabilityai/sdxl-turbo" in MODEL_SPECS_REGISTRY


def test_get_vram_gb_known_model() -> None:
    # Backward-compat alias must still work
    assert get_vram_gb("black-forest-labs/FLUX.1-schnell") == get_min_gpu_memory_gb("black-forest-labs/FLUX.1-schnell")
    assert get_vram_gb("stabilityai/sdxl-turbo") == get_min_gpu_memory_gb("stabilityai/sdxl-turbo")


def test_get_vram_gb_unknown_defaults() -> None:
    # Unknown models default to 16 GB (safe headroom)
    assert get_vram_gb("unknown/model") == 16


def test_get_model_specs() -> None:
    spec = get_model_specs("black-forest-labs/FLUX.1-schnell")
    assert spec is not None
    # New field name: gpu_memory_gb (= true minimum GPU memory needed)
    assert "gpu_memory_gb" in spec
    assert spec["gpu_memory_gb"] >= 16  # FLUX.1-schnell needs at least 16 GB
    assert "tasks" in spec


def test_bytes_per_param_coverage() -> None:
    """All common HF safetensors dtype strings must be in the lookup table."""
    for dtype in ("BF16", "F16", "F32", "F64", "I8", "U8", "F8_E4M3", "F8_E5M2"):
        assert dtype in _BYTES_PER_PARAM, f"{dtype} missing from _BYTES_PER_PARAM"


def test_estimate_vram_from_weight_bytes_sdxl() -> None:
    # SDXL-turbo: ~2.57 B BF16 params → 2.57e9 × 2 ≈ 5.1 GB weights → ×1.35 ≈ 6.9 → snap 8 GB
    sdxl_bytes = int(2_570_004_818 * 2)
    result = _estimate_vram_from_weight_bytes(sdxl_bytes)
    assert result == 8


def test_estimate_vram_from_weight_bytes_flux_schnell() -> None:
    # FLUX.1-schnell: ~11.9 B BF16 params → 11.9e9 × 2 ≈ 22.1 GB weights → ×1.35 ≈ 29.8 → snap 40 GB
    flux_bytes = int(11_900_069_376 * 2)
    result = _estimate_vram_from_weight_bytes(flux_bytes)
    assert result == 40


def test_estimate_vram_from_weight_bytes_sd15() -> None:
    # SD v1.5: ~860 M F16 params → 860e6 × 2 ≈ 1.6 GB weights → ×1.35 ≈ 2.2 → snap 6 GB
    sd15_bytes = int(859_520_964 * 2)
    result = _estimate_vram_from_weight_bytes(sd15_bytes)
    assert result == 6


def test_get_min_gpu_memory_gb_byte_path() -> None:
    # Registry miss → use hf_weight_bytes path
    # 800 M F32 params → 800e6 × 4 ≈ 3 GB weight → ×1.35 ≈ 4.3 → snap 6 GB
    weight_bytes = int(800_000_000 * 4)  # 800 M F32 params
    result = get_min_gpu_memory_gb("unknown/new-model", hf_weight_bytes=weight_bytes)
    assert result == 6


def test_get_min_gpu_memory_gb_registry_wins() -> None:
    # Registry entry ALWAYS wins over byte estimate
    # FLUX.1-dev registry = 28 GB, but raw byte estimate for 12B BF16 would be ~40 GB
    flux_dev_bytes = int(12_000_000_000 * 2)
    result = get_min_gpu_memory_gb("black-forest-labs/FLUX.1-dev", hf_weight_bytes=flux_dev_bytes)
    assert result == 28  # registry value wins


def test_deployment_create_valid() -> None:
    body = DeploymentCreate(
        hf_model_id="stabilityai/sdxl-turbo",
        user_runpod_key="rpa_xxx",
        user_webhook_url="https://example.com/hook",
    )
    assert body.gpu_tier is None
    assert str(body.user_webhook_url) == "https://example.com/hook"


def test_deployment_create_invalid_url() -> None:
    with pytest.raises(ValidationError):
        DeploymentCreate(
            hf_model_id="x",
            user_runpod_key="y",
            user_webhook_url="not-a-url",
        )


def test_log_entry_roundtrip() -> None:
    e = LogEntry(timestamp="2024-02-14T10:00:00Z", level="INFO", message="test")
    d = e.to_dict()
    assert LogEntry.from_dict(d).message == e.message


def test_deployment_doc_roundtrip() -> None:
    doc = DeploymentDoc(
        deployment_id="dep_1",
        status="ready",
        hf_model_id="m",
        user_runpod_key_ref="ref",
        user_webhook_url="https://x.com",
        logs=[LogEntry("2024-01-01T00:00:00Z", "INFO", "msg")],
    )
    d = doc.to_firestore_dict()
    doc2 = DeploymentDoc.from_firestore_dict(d)
    assert doc2.deployment_id == doc.deployment_id
    assert len(doc2.logs) == 1
    assert doc2.logs[0].message == "msg"
