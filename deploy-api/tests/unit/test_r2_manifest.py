"""Unit tests for the in-memory R2 manifest cache helpers."""

import json
from unittest.mock import MagicMock, patch


def _body(payload: dict) -> MagicMock:
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode("utf-8")
    return body


@patch("boto3.client")
def test_fetch_cached_model_ids_uses_ttl_cache(mock_boto_client, monkeypatch):
    """Second manifest read should hit the local TTL cache instead of R2."""
    from src.core.config import get_settings
    from src.services import r2_manifest

    monkeypatch.setenv("CACHE_MANIFEST_TTL_SECONDS", "600")
    get_settings.cache_clear()
    r2_manifest.invalidate_cached_model_ids("https://r2.example")

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": _body({"models": ["model/a"]})}
    mock_boto_client.return_value = mock_s3

    first = r2_manifest.fetch_cached_model_ids("https://r2.example", "key", "secret", force_refresh=True)
    second = r2_manifest.fetch_cached_model_ids("https://r2.example", "key", "secret")

    assert first == {"model/a"}
    assert second == {"model/a"}
    assert mock_s3.get_object.call_count == 1
    r2_manifest.invalidate_cached_model_ids("https://r2.example")
    get_settings.cache_clear()


@patch("boto3.client")
def test_add_model_to_manifest_updates_local_cache(mock_boto_client, monkeypatch):
    """Manifest writes should also update the local cache to avoid immediate re-fetches."""
    from src.core.config import get_settings
    from src.services import r2_manifest

    monkeypatch.setenv("CACHE_MANIFEST_TTL_SECONDS", "600")
    get_settings.cache_clear()
    r2_manifest.invalidate_cached_model_ids("https://r2.example")

    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = [
        {"Body": _body({"models": []})},
        {"Body": _body({"models": ["model/b"]})},
    ]
    mock_boto_client.return_value = mock_s3

    assert r2_manifest.add_model_to_manifest("model/b", "https://r2.example", "key", "secret") is True

    cached = r2_manifest.fetch_cached_model_ids("https://r2.example", "key", "secret")
    assert cached == {"model/b"}
    assert mock_s3.put_object.call_count == 1
    assert mock_s3.get_object.call_count == 2
    r2_manifest.invalidate_cached_model_ids("https://r2.example")
    get_settings.cache_clear()