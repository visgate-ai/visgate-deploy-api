"""Warm pool policy selection and scheduling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Iterable
from zoneinfo import ZoneInfo

from src.core.config import get_settings


@dataclass(frozen=True)
class PoolPolicy:
    name: str
    is_warm: bool


def _parse_csv(value: str) -> set[str]:
    return {v.strip() for v in (value or "").split(",") if v.strip()}


def _parse_windows(value: str) -> list[tuple[time, time]]:
    windows = []
    for chunk in (value or "").split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" not in part:
            continue
        start_str, end_str = part.split("-", 1)
        try:
            start_h = int(start_str.strip())
            end_h = int(end_str.strip())
            windows.append((time(hour=start_h), time(hour=end_h)))
        except ValueError:
            continue
    return windows


def _within_window(now: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= now <= end
    # Overnight window
    return now >= start or now <= end


def choose_pool_policy(model_id: str) -> PoolPolicy:
    """Select warm pool policy for a model based on settings and schedule."""
    settings = get_settings()
    always_on = _parse_csv(settings.warm_pool_always_on_models)
    scheduled = _parse_csv(settings.warm_pool_scheduled_models)
    model_key = (model_id or "").strip()

    if model_key in always_on:
        return PoolPolicy(name="always_on", is_warm=True)

    if model_key in scheduled:
        tz = ZoneInfo(settings.warm_pool_schedule_timezone)
        now = datetime.now(tz).time()
        windows = _parse_windows(settings.warm_pool_schedule_hours)
        for start, end in windows:
            if _within_window(now, start, end):
                return PoolPolicy(name="scheduled_warm", is_warm=True)
        return PoolPolicy(name="scheduled_warm", is_warm=False)

    return PoolPolicy(name="on_demand", is_warm=False)
