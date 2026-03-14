from src.services.base_provider import BaseInferenceProvider

_providers: dict[str, BaseInferenceProvider] = {}


def register_provider(name: str, provider: BaseInferenceProvider):
    _providers[name.lower()] = provider


def _bootstrap_default_providers() -> None:
    if "runpod" not in _providers:
        import src.services.runpod  # noqa: F401
    if "vast" not in _providers:
        import src.services.vast  # noqa: F401


def get_provider(name: str = "runpod") -> BaseInferenceProvider:
    _bootstrap_default_providers()
    provider = _providers.get(name.lower())
    if not provider:
        raise ValueError(f"Provider {name} not found. Available: {list(_providers.keys())}")
    return provider
