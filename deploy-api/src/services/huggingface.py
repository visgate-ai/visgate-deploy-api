"""Hugging Face Hub validation and model metadata."""

import asyncio
from typing import Any, Optional

from src.core.errors import HuggingFaceModelNotFoundError
from src.core.logging import structured_log
from src.core.telemetry import span
from src.models.model_specs_registry import get_model_specs, get_vram_gb

# Optional dependency
try:
    from huggingface_hub import HfApi

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    HfApi = None  # type: ignore


class ModelInfo:
    """Result of HF model validation."""

    def __init__(
        self,
        model_id: str,
        vram_gb: int,
        exists: bool = True,
        raw: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_id = model_id
        self.vram_gb = vram_gb
        self.exists = exists
        self.raw = raw or {}


async def validate_model(
    model_id: str,
    token: Optional[str] = None,
    timeout_seconds: int = 10,
) -> ModelInfo:
    """
    Check if model exists on Hugging Face Hub and return metadata.
    Uses registry for vram_gb; validates existence via HfApi.
    """
    with span("huggingface.validate_model", {"hf_model_id": model_id}):
        vram_gb = get_vram_gb(model_id)
        # Registry modelleri için HF API'yi atla (429 rate limit önleme)
        if get_model_specs(model_id) is not None:
            return ModelInfo(model_id=model_id, vram_gb=vram_gb, exists=True)
        if not _HF_AVAILABLE or HfApi is None:
            structured_log(
                "WARNING",
                "huggingface_hub not installed; skipping existence check",
                operation="model.validate",
                metadata={"hf_model_id": model_id},
            )
            return ModelInfo(model_id=model_id, vram_gb=vram_gb, exists=True)

        def _check() -> None:
            api = HfApi(token=token)
            last_err = None
            for attempt in range(3):
                try:
                    api.model_info(model_id, timeout=timeout_seconds)
                    return
                except Exception as e:
                    last_err = e
                    err_str = str(e).lower()
                    if "404" in err_str or "not found" in err_str or "does not exist" in err_str:
                        raise HuggingFaceModelNotFoundError(model_id)
                    if "429" in str(e) and attempt < 2:
                        import time as _t
                        _t.sleep(2 ** attempt)
                        continue
                    raise
            if last_err:
                raise last_err

        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, _check),
                timeout=timeout_seconds + 2,
            )
        except asyncio.TimeoutError:
            raise HuggingFaceModelNotFoundError(
                model_id,
                message=f"Model validation timed out after {timeout_seconds}s",
            )
        except HuggingFaceModelNotFoundError:
            raise
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise HuggingFaceModelNotFoundError(model_id)
            raise HuggingFaceModelNotFoundError(
                model_id,
                message=f"Failed to validate model: {e}",
            )

        return ModelInfo(model_id=model_id, vram_gb=vram_gb, exists=True)
