"""Model capability checks for task-aware provisioning."""

from __future__ import annotations

import os

from src.core.tasks import normalize_task
from src.models.model_specs_registry import get_model_specs


def supports_task(model_id: str, task: str) -> bool:
    specs = get_model_specs(model_id)
    normalized_task = normalize_task(task)
    if (
        normalized_task == "text_to_video"
        and (os.getenv("INFERENCE_PROVIDER", "").strip().lower() == "local")
        and (os.getenv("ALLOW_VIDEO_SMOKE_IMAGE_FALLBACK", "false").strip().lower() == "true")
    ):
        return True
    if not specs:
        return True
    tasks = {normalize_task(item) for item in (specs.get("tasks") or ["text2img"])}
    return normalized_task in tasks
