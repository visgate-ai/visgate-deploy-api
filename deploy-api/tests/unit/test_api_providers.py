"""Unit tests for POST /v1/providers/validate."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


@patch("src.api.routes.providers._validate_runpod_key", new_callable=AsyncMock)
def test_validate_provider_runpod_success(
    mock_validate: AsyncMock,
    client: TestClient,
) -> None:
    """RunPod keys are validated through the provider-specific helper."""
    mock_validate.return_value = {"valid": True, "message": "RunPod API key is valid"}

    resp = client.post(
        "/v1/providers/validate",
        json={"provider": "runpod", "api_key": "rpa_test_key"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"valid": True, "message": "RunPod API key is valid"}
    mock_validate.assert_awaited_once_with("rpa_test_key")


@patch("src.api.routes.providers._validate_huggingface_key", new_callable=AsyncMock)
def test_validate_provider_normalizes_case_and_whitespace(
    mock_validate: AsyncMock,
    client: TestClient,
) -> None:
    """Provider names are normalized before dispatch and keys are trimmed."""
    mock_validate.return_value = {"valid": True, "message": "Hugging Face token is valid"}

    resp = client.post(
        "/v1/providers/validate",
        json={"provider": " HuggingFace ", "api_key": "  hf_secret  "},
    )

    assert resp.status_code == 200
    assert resp.json()["valid"] is True
    mock_validate.assert_awaited_once_with("hf_secret")


def test_validate_provider_rejects_unsupported_provider(client: TestClient) -> None:
    """Unsupported providers return a structured invalid response."""
    resp = client.post(
        "/v1/providers/validate",
        json={"provider": "openai", "api_key": "sk-test"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert "Unsupported provider" in data["message"]


@patch("src.api.routes.providers._validate_replicate_key", new_callable=AsyncMock)
def test_validate_provider_http_errors_become_invalid_response(
    mock_validate: AsyncMock,
    client: TestClient,
) -> None:
    """Provider-side failures should not surface as 500s to callers."""
    mock_validate.side_effect = RuntimeError("upstream exploded")

    resp = client.post(
        "/v1/providers/validate",
        json={"provider": "replicate", "api_key": "r8_test"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert "validation error" in data["message"]