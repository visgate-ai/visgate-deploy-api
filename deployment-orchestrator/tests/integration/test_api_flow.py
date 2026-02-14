"""Integration tests: full API flow with mocked Runpod/HF (respx)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Contract: validate Pydantic schemas against sample responses
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.mark.skipif(not FIXTURES_DIR.joinpath("sample_responses.json").exists(), reason="No fixtures")
def test_contract_deployment_response_202_schema() -> None:
    """Response 202 matches DeploymentResponse202 schema."""
    from src.models.schemas import DeploymentResponse202
    path = FIXTURES_DIR / "sample_responses.json"
    data = json.loads(path.read_text())
    obj = data["deployment_response_202"]
    # Fix date for Pydantic
    obj["created_at"] = obj["created_at"].replace("Z", "+00:00")
    parsed = DeploymentResponse202.model_validate(obj)
    assert parsed.status == "validating"
    assert parsed.deployment_id == "dep_2024_abc123"


@pytest.mark.skipif(not FIXTURES_DIR.joinpath("sample_responses.json").exists(), reason="No fixtures")
def test_contract_deployment_response_ready_schema() -> None:
    """Ready response matches DeploymentResponse schema."""
    from src.models.schemas import DeploymentResponse, LogEntrySchema
    path = FIXTURES_DIR / "sample_responses.json"
    data = json.loads(path.read_text())
    obj = data["deployment_response_ready"]
    for log in obj["logs"]:
        log["timestamp"] = log["timestamp"].replace("Z", "+00:00")
    obj["created_at"] = obj["created_at"].replace("Z", "+00:00")
    obj["ready_at"] = obj["ready_at"].replace("Z", "+00:00") if obj.get("ready_at") else None
    parsed = DeploymentResponse.model_validate(obj)
    assert parsed.status == "ready"
    assert parsed.runpod_endpoint_id == "xyz789"
    assert len(parsed.logs) == 3
