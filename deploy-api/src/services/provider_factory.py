from typing import Any, Optional
from src.services.base_provider import BaseInferenceProvider

_providers: dict[str, BaseInferenceProvider] = {}

def register_provider(name: str, provider: BaseInferenceProvider):
    _providers[name.lower()] = provider

def get_provider(name: str = "runpod") -> BaseInferenceProvider:
    provider = _providers.get(name.lower())
    if not provider:
        raise ValueError(f"Provider {name} not found. Available: {list(_providers.keys())}")
    return provider
