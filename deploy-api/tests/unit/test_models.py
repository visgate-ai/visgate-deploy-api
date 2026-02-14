"""Unit tests for models and registry."""

import pytest
from pydantic import ValidationError

from src.models.model_specs_registry import get_model_specs, get_vram_gb, MODEL_SPECS_REGISTRY
from src.models.schemas import DeploymentCreate, DeploymentResponse202, LogEntrySchema
from src.models.entities import DeploymentDoc, LogEntry


def test_model_specs_registry_has_seed_models() -> None:
    assert "black-forest-labs/FLUX.1-schnell" in MODEL_SPECS_REGISTRY
    assert "stabilityai/sdxl-turbo" in MODEL_SPECS_REGISTRY


def test_get_vram_gb_known_model() -> None:
    assert get_vram_gb("black-forest-labs/FLUX.1-schnell") == 12
    assert get_vram_gb("stabilityai/sdxl-turbo") == 8


def test_get_vram_gb_unknown_defaults() -> None:
    assert get_vram_gb("unknown/model") == 12


def test_get_model_specs() -> None:
    spec = get_model_specs("black-forest-labs/FLUX.1-schnell")
    assert spec is not None
    assert spec["vram_gb"] == 12
    assert "float16" in spec["supported_precisions"]


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
