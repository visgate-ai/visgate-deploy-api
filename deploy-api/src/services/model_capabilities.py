"""Model capability checks for task-aware provisioning."""

from __future__ import annotations

from src.core.tasks import normalize_task
from src.models.model_specs_registry import get_model_specs


def supports_task(model_id: str, task: str) -> bool:
    specs = get_model_specs(model_id)
    normalized_task = normalize_task(task)
    if not specs:
        return True
    tasks = {normalize_task(item) for item in (specs.get("tasks") or ["text2img"])}
    return normalized_task in tasks
