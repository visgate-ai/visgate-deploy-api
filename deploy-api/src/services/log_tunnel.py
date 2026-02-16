"""In-memory log tunnel for stateless live streaming."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque

from src.core.config import get_settings


@dataclass(frozen=True)
class LiveLogEntry:
    timestamp: float
    level: str
    message: str


_store: dict[str, Deque[LiveLogEntry]] = {}
_lock = Lock()


def append_live_log(deployment_id: str, level: str, message: str) -> None:
    settings = get_settings()
    max_entries = settings.log_stream_max_entries
    now = time.time()
    entry = LiveLogEntry(timestamp=now, level=level, message=message)
    with _lock:
        buf = _store.get(deployment_id)
        if not buf:
            buf = deque(maxlen=max_entries)
            _store[deployment_id] = buf
        buf.append(entry)


def get_live_logs_since(deployment_id: str, since_ts: float) -> list[LiveLogEntry]:
    settings = get_settings()
    ttl = settings.log_stream_ttl_seconds
    now = time.time()
    with _lock:
        buf = _store.get(deployment_id)
        if not buf:
            return []
        # Prune old entries
        while buf and (now - buf[0].timestamp) > ttl:
            buf.popleft()
        return [entry for entry in list(buf) if entry.timestamp > since_ts]
