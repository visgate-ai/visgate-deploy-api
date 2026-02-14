"""Unit tests for model_name -> HF resolution."""

import pytest

from src.services.model_resolver import UnknownModelError, get_hf_name


def test_get_hf_name_fal_veo3() -> None:
    assert get_hf_name("veo3", "fal") == "black-forest-labs/FLUX.1-schnell"


def test_get_hf_name_provider_agnostic() -> None:
    assert get_hf_name("veo3", None) == "black-forest-labs/FLUX.1-schnell"
    assert get_hf_name("flux-schnell", None) == "black-forest-labs/FLUX.1-schnell"
    assert get_hf_name("sdxl-turbo", None) == "stabilityai/sdxl-turbo"


def test_get_hf_name_unknown_raises() -> None:
    with pytest.raises(UnknownModelError) as exc_info:
        get_hf_name("unknown-model", "fal")
    assert exc_info.value.status_code == 400
    assert "unknown-model" in exc_info.value.message


def test_get_hf_name_empty_raises() -> None:
    with pytest.raises(UnknownModelError):
        get_hf_name("", None)
