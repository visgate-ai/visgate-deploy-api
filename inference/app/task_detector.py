from __future__ import annotations

from typing import Optional


def detect_task(model_id: str, environ_task: str = "") -> str:
    environ_task = environ_task.strip().lower()
    canonical_map = {
        "text2img": "text2img",
        "text_to_image": "text2img",
        "image2img": "text2img",
        "image_to_image": "text2img",
        "text2video": "text2video",
        "text_to_video": "text2video",
        "video2video": "text2video",
        "video_to_video": "text2video",
        "video_edit": "text2video",
        "speech_to_text": "speech_to_text",
        "text_to_speech": "text_to_speech",
    }
    mapped_task = canonical_map.get(environ_task)
    if mapped_task:
        return mapped_task

    model_id_lower = model_id.lower()
    
    # Video Models
    if any(token in model_id_lower for token in ["wan2", "cogvideo", "svd", "animatediff", "t2v"]):
        return "text2video"
        
    # Audio Models
    if any(token in model_id_lower for token in ["whisper", "wav2vec", "asr"]):
        return "speech_to_text"
    if any(token in model_id_lower for token in ["tts", "bark", "speecht5", "xtts", "audioldm"]):
        return "text_to_speech"
        
    # Default to Image (largest category)
    return "text2img"
