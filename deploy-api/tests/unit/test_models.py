"""Unit tests for models and registry."""

import pytest
from pydantic import ValidationError

from src.models.model_specs_registry import get_model_specs, get_vram_gb, get_min_gpu_memory_gb, MODEL_SPECS_REGISTRY
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
