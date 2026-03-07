"""Task normalization helpers shared across deployment and inference APIs."""

from __future__ import annotations

CANONICAL_TASKS: tuple[str, ...] = (
    "text_to_image",
    "image_to_image",
    "text_to_video",
    "video_to_video",
    "video_edit",
    "text_to_speech",
    "speech_to_text",
    "audio_to_audio",
)

TASK_ALIASES: dict[str, str] = {
    "text2img": "text_to_image",
    "txt2img": "text_to_image",
    "text-to-image": "text_to_image",
    "image2img": "image_to_image",
    "img2img": "image_to_image",
    "image-to-image": "image_to_image",
    "text2video": "text_to_video",
    "txt2video": "text_to_video",
    "text-to-video": "text_to_video",
    "video2video": "video_to_video",
    "vid2vid": "video_to_video",
    "video-to-video": "video_to_video",
    "prompt_video_to_video": "video_edit",
    "prompt+video-to-video": "video_edit",
    "video-edit": "video_edit",
    "text2speech": "text_to_speech",
    "text2audio": "text_to_speech",
    "prompt-to-voice": "text_to_speech",
    "prompt_to_voice": "text_to_speech",
    "text-to-speech": "text_to_speech",
    "speech2text": "speech_to_text",
    "audio2text": "speech_to_text",
    "speech-to-text": "speech_to_text",
    "audio2audio": "audio_to_audio",
    "audio-to-audio": "audio_to_audio",
}


def normalize_task(task: str | None) -> str | None:
    """Return canonical task name or the stripped original value if unknown."""
    if task is None:
        return None
    cleaned = task.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower().replace(" ", "_")
    return TASK_ALIASES.get(lowered, lowered)


def is_known_task(task: str | None) -> bool:
    normalized = normalize_task(task)
    return normalized in CANONICAL_TASKS if normalized else False
