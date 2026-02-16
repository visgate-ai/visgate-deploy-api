"""Model capability checks for task-aware provisioning."""

from __future__ import annotations

from typing import Iterable

from src.models.model_specs_registry import get_model_specs


def supports_task(model_id: str, task: str) -> bool:
    specs = get_model_specs(model_id)
    if not specs:
        return True
    tasks = specs.get("tasks") or ["text2img"]
    return task in tasks
