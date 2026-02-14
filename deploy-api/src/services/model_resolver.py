"""Resolve provider + model_name (from external get_models) to Hugging Face model ID."""

from src.core.errors import OrchestratorError


class UnknownModelError(OrchestratorError):
    """Raised when model_name (and optional provider) cannot be resolved to an HF model."""

    def __init__(self, model_name: str, provider: str | None = None) -> None:
        key = f"{provider}.{model_name}" if provider else model_name
        super().__init__(
            message=f"Unknown model: {key}",
            status_code=400,
            details={"model_name": model_name, "provider": provider},
        )


# (provider, model_name) -> hf_model_id
# Provider comes from external get_models(provider=fal, "veo3"); we map to HF ID for deploy_model.
MODEL_NAME_TO_HF: dict[tuple[str | None, str], str] = {
    # fal.veo3 -> HF model (example: use FLUX for image gen; replace with actual mapping)
    ("fal", "veo3"): "black-forest-labs/FLUX.1-schnell",
    ("fal", "veo2"): "black-forest-labs/FLUX.1-schnell",
    # provider-agnostic by model_name
    (None, "veo3"): "black-forest-labs/FLUX.1-schnell",
    (None, "flux-schnell"): "black-forest-labs/FLUX.1-schnell",
    (None, "flux-dev"): "black-forest-labs/FLUX.1-dev",
    (None, "sdxl-turbo"): "stabilityai/sdxl-turbo",
}


def get_hf_name(model_name: str, provider: str | None = None) -> str:
    """
    Resolve model_name (and optional provider) to Hugging Face model ID.

    External flow: model_name = get_models(provider=fal, "veo3") -> then hf_name = get_hf_name(model_name, provider).
    """
    if not model_name or not model_name.strip():
        raise UnknownModelError(model_name or "", provider)
    key = model_name.strip()
    prov = provider.strip().lower() if (provider and provider.strip()) else None
    # Try (provider, model_name) then (None, model_name)
    for p in (prov, None):
        hf_id = MODEL_NAME_TO_HF.get((p, key))
        if hf_id:
            return hf_id
    raise UnknownModelError(key, provider)
