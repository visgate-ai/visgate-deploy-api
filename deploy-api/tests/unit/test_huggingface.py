from __future__ import annotations

import pytest

from src.services import huggingface


class _FakeModelInfo:
    safetensors = None


@pytest.mark.asyncio
async def test_validate_model_normalizes_blank_token_to_none(monkeypatch):
    captured_tokens: list[str | None] = []

    class FakeHfApi:
        def __init__(self, token=None):
            captured_tokens.append(token)

        def model_info(self, model_id: str, timeout: int = 10):
            return _FakeModelInfo()

    monkeypatch.setattr(huggingface, "_HF_AVAILABLE", True)
    monkeypatch.setattr(huggingface, "HfApi", FakeHfApi)

    info = await huggingface.validate_model("unknown/model", token="   ")

    assert captured_tokens == [None]
    assert info.model_id == "unknown/model"
    assert info.min_gpu_memory_gb == 16