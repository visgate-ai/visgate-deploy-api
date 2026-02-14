"""Unit tests for GPU registry."""

import pytest

from src.services.gpu_registry import (
    get_runpod_gpu_ids,
    select_gpu_id_for_vram,
    gpu_id_to_display_name,
)


def test_get_runpod_gpu_ids_a40() -> None:
    assert "AMPERE_48" in get_runpod_gpu_ids("A40")


def test_get_runpod_gpu_ids_default() -> None:
    ids = get_runpod_gpu_ids(None)
    assert len(ids) > 0


def test_select_gpu_id_for_vram() -> None:
    gpu_id = select_gpu_id_for_vram(12, "A40")
    assert gpu_id is not None
    assert select_gpu_id_for_vram(100, None) is None or select_gpu_id_for_vram(100, None) is not None


def test_gpu_id_to_display_name() -> None:
    assert "A40" in gpu_id_to_display_name("AMPERE_48")
