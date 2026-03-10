"""Select the correct RunPod worker profile for a model/task pair."""

from __future__ import annotations

from typing import Any

from src.core.config import Settings
from src.core.tasks import normalize_task
from src.models.model_specs_registry import get_model_specs

IMAGE_PROFILE = "image"
AUDIO_PROFILE = "audio"
VIDEO_PROFILE = "video"

IMAGE_TASKS = {"text_to_image", "image_to_image"}
AUDIO_TASKS = {"text_to_speech", "speech_to_text", "audio_to_audio"}
VIDEO_TASKS = {"text_to_video", "video_to_video", "video_edit"}


def infer_worker_profile(model_id: str, task: str | None = None) -> str:
    normalized_task = normalize_task(task)
    if normalized_task in IMAGE_TASKS:
        return IMAGE_PROFILE
    if normalized_task in AUDIO_TASKS:
        return AUDIO_PROFILE
    if normalized_task in VIDEO_TASKS:
        return VIDEO_PROFILE

    specs = get_model_specs(model_id) or {}
    registered_tasks = {normalize_task(item) for item in (specs.get("tasks") or []) if normalize_task(item)}
    if registered_tasks & VIDEO_TASKS:
        return VIDEO_PROFILE
    if registered_tasks & AUDIO_TASKS:
        return AUDIO_PROFILE
    if registered_tasks & IMAGE_TASKS:
        return IMAGE_PROFILE

    model_id_lower = model_id.lower()
    if any(token in model_id_lower for token in ("whisper", "wav2vec", "bark", "xtts", "tts", "audiogen", "musicgen", "audio")):
        return AUDIO_PROFILE
    if any(token in model_id_lower for token in ("video", "cogvideo", "wan", "zeroscope", "t2v")):
        return VIDEO_PROFILE
    return IMAGE_PROFILE


def resolve_worker_target(settings: Settings, model_id: str, task: str | None = None) -> dict[str, Any]:
    profile = infer_worker_profile(model_id, task)
    if profile == AUDIO_PROFILE:
        image = settings.docker_image_audio or settings.docker_image
    elif profile == VIDEO_PROFILE:
        image = settings.docker_image_video or settings.docker_image
    else:
        image = settings.docker_image_image or settings.docker_image

    return {
        "profile": profile,
        "image": image,
    }