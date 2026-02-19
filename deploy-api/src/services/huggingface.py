"""Hugging Face Hub validation and model metadata."""

import asyncio
from typing import Any, Optional

from src.core.errors import HuggingFaceModelNotFoundError
from src.core.logging import structured_log
from src.core.telemetry import span
from src.models.model_specs_registry import (
    _BYTES_PER_PARAM,
    get_min_gpu_memory_gb,
    get_model_specs,
)

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
        min_gpu_memory_gb: int,
        exists: bool = True,
        raw: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_id = model_id
        self.min_gpu_memory_gb = min_gpu_memory_gb
        # backwards-compat alias
        self.vram_gb = min_gpu_memory_gb
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
        # Fast path: registry hit (also skips HF rate-limit)
        if get_model_specs(model_id) is not None:
            mem_gb = get_min_gpu_memory_gb(model_id)
            return ModelInfo(model_id=model_id, min_gpu_memory_gb=mem_gb, exists=True)

        if not _HF_AVAILABLE or HfApi is None:
            structured_log(
                "WARNING",
                "huggingface_hub not installed; skipping existence check",
                operation="model.validate",
                metadata={"hf_model_id": model_id},
            )
            mem_gb = get_min_gpu_memory_gb(model_id)
            return ModelInfo(model_id=model_id, min_gpu_memory_gb=mem_gb, exists=True)

        # For unknown models: fetch HF metadata to extract weight bytes (dtype-aware)
        _hf_weight_bytes: list[int | None] = [None]
        _hf_params_millions: list[int | None] = [None]

        def _check() -> None:
            api = HfApi(token=token)
            last_err = None
            for attempt in range(3):
                try:
                    info = api.model_info(model_id, timeout=timeout_seconds)
                    # Primary: dtype-aware byte calculation from safetensors.parameters
                    # safetensors.parameters = {"BF16": 11_900_069_376, "F32": 122_880, ...}
                    try:
                        sf = getattr(info, "safetensors", None)
                        if sf and hasattr(sf, "parameters") and sf.parameters:
                            total_bytes = sum(
                                int(count) * _BYTES_PER_PARAM.get(dtype, 4)
                                for dtype, count in sf.parameters.items()
                            )
                            if total_bytes > 0:
                                _hf_weight_bytes[0] = total_bytes
                        # Fallback: raw param count (no dtype)
                        if _hf_weight_bytes[0] is None and sf and hasattr(sf, "total") and sf.total:
                            _hf_params_millions[0] = int(sf.total) // 1_000_000
                    except Exception:
                        pass
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

        mem_gb = get_min_gpu_memory_gb(
            model_id,
            hf_weight_bytes=_hf_weight_bytes[0],
            hf_params_millions=_hf_params_millions[0],
        )
        structured_log(
            "INFO",
            "Unknown model VRAM estimated from HF metadata",
            operation="model.validate",
            metadata={
                "hf_model_id": model_id,
                "weight_bytes": _hf_weight_bytes[0],
                "params_millions": _hf_params_millions[0],
                "min_gpu_memory_gb": mem_gb,
            },
        )
        return ModelInfo(model_id=model_id, min_gpu_memory_gb=mem_gb, exists=True)
