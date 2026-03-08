"""Audio-focused RunPod worker."""

from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time
from typing import Any, Optional

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    CLEANUP_FAILURE_THRESHOLD,
    CLEANUP_IDLE_TIMEOUT_SECONDS,
    DEVICE,
    HF_MODEL_ID,
    HF_TOKEN,
    RUNPOD_API_KEY,
    VISGATE_WEBHOOK,
)
from app.loader import resolve_model_source
from app.runtime_common import download_to_tempfile, log_tunnel, post_json, request_cleanup, upload_bytes

_pipeline: Optional[Any] = None
_load_error: Optional[str] = None
_last_request_at: float = 0.0
_failure_count: int = 0
_task_kind: str = "speech_to_text"


def _notify_orchestrator(status: str, message: str | None = None) -> None:
    if not VISGATE_WEBHOOK or not VISGATE_WEBHOOK.strip():
        return
    payload = {"status": status}
    if message:
        payload["message"] = message
    for attempt in range(3):
        try:
            post_json(VISGATE_WEBHOOK, payload)
            return
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)


def _device_index() -> int:
    return 0 if DEVICE.startswith("cuda") else -1


def _infer_task_kind(model_id: str) -> str:
    task = (os.environ.get("TASK") or "").strip().lower()
    if task in {"speech_to_text", "speech2text"}:
        return "speech_to_text"
    if task in {"text_to_speech", "text2speech", "text2audio"}:
        return "text_to_speech"
    model_id_lower = model_id.lower()
    if any(token in model_id_lower for token in ("whisper", "wav2vec", "asr")):
        return "speech_to_text"
    if any(token in model_id_lower for token in ("tts", "bark", "speecht5", "xtts")):
        return "text_to_speech"
    return "speech_to_text"


def _load_model_background() -> None:
    global _pipeline, _load_error, _task_kind
    from transformers import pipeline

    try:
        _task_kind = _infer_task_kind(HF_MODEL_ID)
        model_source, _ = resolve_model_source(HF_MODEL_ID)
        pipeline_task = "automatic-speech-recognition" if _task_kind == "speech_to_text" else "text-to-audio"
        log_tunnel("INFO", f"Loading audio model {HF_MODEL_ID} as {pipeline_task}")
        _notify_orchestrator("loading_model", "Audio model loading started")
        _pipeline = pipeline(pipeline_task, model=model_source, token=HF_TOKEN, device=_device_index())
        log_tunnel("INFO", "Audio model loaded successfully")
        _notify_orchestrator("ready", "Audio model loaded successfully")
    except Exception as exc:
        _load_error = str(exc)
        log_tunnel("ERROR", f"Audio model load failed: {exc}")
        _notify_orchestrator("failed", _load_error)
        request_cleanup("startup_failure")


def _handle_speech_to_text(job_input: dict[str, Any]) -> dict[str, Any]:
    audio_url = job_input.get("audio_url") or job_input.get("audioUrl")
    audio_base64 = job_input.get("audio_base64") or job_input.get("audioBase64")
    temp_path = None
    try:
        if audio_url:
            temp_path = download_to_tempfile(str(audio_url), suffix=".wav")
            result = _pipeline(temp_path)
        elif audio_base64:
            raw = base64.b64decode(audio_base64)
            with io.BytesIO(raw) as buffer:
                audio_data, sample_rate = sf.read(buffer)
            result = _pipeline({"raw": audio_data, "sampling_rate": sample_rate})
        else:
            return {"error": "Missing 'audio_url' or 'audio_base64' in input", "status": "failed"}
        return {
            "text": result.get("text") if isinstance(result, dict) else str(result),
            "model_id": HF_MODEL_ID,
            "task": "speech_to_text",
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _handle_text_to_speech(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    text = job_input.get("text") or job_input.get("prompt")
    if not text:
        return {"error": "Missing 'text' or 'prompt' in input", "status": "failed"}
    result = _pipeline(str(text))
    audio = result.get("audio")
    sampling_rate = int(result.get("sampling_rate", 22050))
    if audio is None:
        return {"error": "Audio pipeline returned no audio", "status": "failed"}
    buffer = io.BytesIO()
    sf.write(buffer, audio, sampling_rate, format="WAV")
    raw_bytes = buffer.getvalue()
    artifact = upload_bytes(raw_bytes, job, job_input, content_type="audio/wav", extension="wav")
    payload: dict[str, Any] = {
        "model_id": HF_MODEL_ID,
        "task": "text_to_speech",
        "sampling_rate": sampling_rate,
    }
    if artifact:
        payload["artifact"] = artifact
    else:
        payload["audio_base64"] = {
            "stored": False,
            "length": len(base64.b64encode(raw_bytes)),
            "reason": "no_s3_config",
        }
    return payload


def handler(job: dict[str, Any]) -> dict[str, Any]:
    global _last_request_at, _failure_count
    job_input = job.get("input") or {}
    if job_input.get("debug") is True:
        return {"status": "ok", "pipeline_loaded": _pipeline is not None, "task_kind": _task_kind, "load_error": _load_error}
    if _pipeline is None:
        wait_start = time.time()
        while _pipeline is None and _load_error is None:
            if time.time() - wait_start > 270:
                return {"error": "Model failed to load within timeout.", "status": "loading"}
            time.sleep(2)
        if _load_error:
            return {"error": f"Model load failed: {_load_error}", "status": "failed"}
    try:
        _last_request_at = time.time()
        started_at = time.time()
        if _task_kind == "text_to_speech":
            result = _handle_text_to_speech(job, job_input)
        else:
            result = _handle_speech_to_text(job_input)
        result["execution_duration_ms"] = max(int((time.time() - started_at) * 1000), 0)
        _failure_count = 0
        return result
    except Exception as exc:
        _failure_count += 1
        log_tunnel("ERROR", f"Audio inference failed: {exc}")
        if _failure_count >= CLEANUP_FAILURE_THRESHOLD:
            request_cleanup("inference_failure_threshold")
        return {"error": str(exc), "model_id": HF_MODEL_ID}


def main() -> None:
    import runpod

    def _idle_watchdog() -> None:
        global _last_request_at
        while True:
            time.sleep(15)
            if _pipeline is None or _last_request_at == 0.0:
                continue
            if time.time() - _last_request_at > CLEANUP_IDLE_TIMEOUT_SECONDS:
                request_cleanup("idle_timeout")
                return

    threading.Thread(target=_load_model_background, daemon=True).start()
    threading.Thread(target=_idle_watchdog, daemon=True).start()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()