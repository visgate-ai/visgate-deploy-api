"""Universal RunPod worker supporting text-to-image, video, and audio tasks via unified entry point."""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import threading
import time
from typing import Any, Optional

import numpy as np
import imageio.v2 as imageio
import soundfile as sf
import torch

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    CLEANUP_FAILURE_THRESHOLD,
    CLEANUP_IDLE_TIMEOUT_SECONDS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEVICE,
    HF_MODEL_ID,
    HF_TOKEN,
    MODEL_LOAD_WAIT_TIMEOUT_SECONDS,
    VISGATE_WEBHOOK,
)
from app.loader import resolve_model_source
from app.runtime_common import download_to_tempfile, log_tunnel, post_json, request_cleanup, upload_bytes
from app.task_detector import detect_task
from PIL import Image

_pipeline: Optional[Any] = None
_load_error: Optional[str] = None
_last_request_at: float = 0.0
_failure_count: int = 0
_task_kind: str = "text2img"


def _notify_orchestrator(
    status: str,
    message: str | None = None,
    timings: dict[str, float | bool | None] | None = None,
) -> None:
    if not VISGATE_WEBHOOK or not VISGATE_WEBHOOK.strip():
        return
    payload: dict[str, Any] = {"status": status}
    if message:
        payload["message"] = message
    if timings:
        payload.update(
            {
                "t_r2_sync_s": timings.get("t_r2_sync_s"),
                "t_model_load_s": timings.get("t_model_load_s"),
                "loaded_from_cache": timings.get("loaded_from_cache"),
            }
        )
    for attempt in range(5):
        try:
            post_json(VISGATE_WEBHOOK, payload)
            return
        except Exception:
            if attempt < 4:
                time.sleep(2 ** attempt)


def _torch_dtype() -> torch.dtype:
    if DEVICE.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if DEVICE.startswith("cuda") else torch.float32


def _runtime_video_model_id(model_id: str) -> str:
    aliases = {
        "Wan-AI/Wan2.1-T2V-1.3B": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    }
    return aliases.get(model_id, model_id)


def _load_model_background() -> None:
    global _pipeline, _load_error, _task_kind

    try:
        _task_kind = detect_task(HF_MODEL_ID, os.environ.get("TASK", ""))
        log_tunnel("INFO", f"Detected task: {_task_kind} for model: {HF_MODEL_ID}")

        effective_model_id = _runtime_video_model_id(HF_MODEL_ID) if _task_kind == "text2video" else HF_MODEL_ID
        model_source, use_local, t_r2_sync_s, loaded_from_cache = resolve_model_source(effective_model_id)

        _notify_orchestrator("loading_model", f"Model loading started ({_task_kind})")
        t0 = time.time()

        if _task_kind == "text2img":
            from pipelines.registry import load_pipeline
            _pipeline = load_pipeline(model_id=effective_model_id, token=HF_TOKEN, device=DEVICE)
        
        elif _task_kind == "text2video":
            from diffusers import DiffusionPipeline
            _pipeline = DiffusionPipeline.from_pretrained(model_source, torch_dtype=_torch_dtype(), token=HF_TOKEN)
            if DEVICE.startswith("cuda"):
                _pipeline.to(DEVICE)
                
        elif _task_kind in ("speech_to_text", "text_to_speech"):
            from transformers import pipeline
            pipeline_task = "automatic-speech-recognition" if _task_kind == "speech_to_text" else "text-to-audio"
            _pipeline = pipeline(pipeline_task, model=model_source, token=HF_TOKEN, device=0 if DEVICE.startswith("cuda") else -1)
            
        load_elapsed = time.time() - t0
        log_tunnel("INFO", f"Pipeline loaded successfully in {load_elapsed:.1f}s")
        _notify_orchestrator(
            "ready",
            "Model loaded successfully",
            timings={
                "t_r2_sync_s": round(t_r2_sync_s, 3) if t_r2_sync_s is not None else None,
                "t_model_load_s": round(load_elapsed, 3),
                "loaded_from_cache": loaded_from_cache,
            },
        )
    except Exception as exc:
        import traceback
        _load_error = str(exc)
        log_tunnel("ERROR", f"Model load failed: {exc}")
        _notify_orchestrator("failed", _load_error)
        request_cleanup("startup_failure")
        traceback.print_exc()


def _handle_image(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    prompt = job_input.get("prompt")
    if not prompt and not job_input.get("input_image_url"):
        return {"error": "Missing 'prompt' or 'input_image_url'"}

    kwargs = {}
    tmp_img = None
    if job_input.get("input_image_url"):
        try:
            tmp_img = download_to_tempfile(job_input["input_image_url"], ".png")
            kwargs["image"] = Image.open(tmp_img).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to download input image: {e}"}

    try:
        result = _pipeline.run(
            prompt=str(prompt) if prompt else None,
            num_inference_steps=int(job_input.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)),
            guidance_scale=float(job_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)),
            height=job_input.get("height"),
            width=job_input.get("width"),
            seed=job_input.get("seed"),
            **kwargs,
        )
        # S3 enforced
        if result.get("image_base64"):
            data = base64.b64decode(result["image_base64"])
            artifact = upload_bytes(data, job, job_input, content_type="image/png", extension="png")
            if artifact:
                result["artifact"] = artifact
            result.pop("image_base64", None)  # Force remove base64
        return result
    finally:
        if tmp_img and os.path.exists(tmp_img):
            os.remove(tmp_img)


def _handle_video(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing or empty 'prompt' in input", "status": "failed"}

    output = _pipeline(
        prompt=str(prompt),
        num_inference_steps=int(job_input.get("num_inference_steps", 16)),
        guidance_scale=float(job_input.get("guidance_scale", 7.5)),
        num_frames=int(job_input.get("num_frames", 16)),
    )

    # Simplified frame extraction
    frames_raw = None
    if hasattr(output, "frames") and output.frames is not None:
        frames_raw = output.frames
    elif isinstance(output, dict) and output.get("frames") is not None:
        frames_raw = output["frames"]
    elif hasattr(output, "images") and output.images is not None:
        frames_raw = output.images

    if frames_raw is None:
        raise RuntimeError("Video pipeline returned no frames")

    if isinstance(frames_raw, np.ndarray):
        if frames_raw.ndim == 5:
            frames_raw = frames_raw[0]
        arrays = [frames_raw[i] for i in range(len(frames_raw))]
    else:
        # Fallback list handling
        if isinstance(frames_raw, list) and isinstance(frames_raw[0], list):
            frames_raw = frames_raw[0]
        arrays = [np.array(f.convert("RGB")) if hasattr(f, "convert") else np.asarray(f) for f in frames_raw]

    result_frames = []
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            arr = (arr * 255 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
        result_frames.append(arr)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        imageio.mimwrite(tmp.name, result_frames, fps=int(job_input.get("fps", 8)))
        tmp.flush()
        with open(tmp.name, "rb") as fh:
            video_bytes = fh.read()

    artifact = upload_bytes(video_bytes, job, job_input, content_type="video/mp4", extension="mp4")
    os.remove(tmp.name)

    payload: dict[str, Any] = {
        "model_id": HF_MODEL_ID,
        "frame_count": len(result_frames),
    }
    if artifact:
        payload["artifact"] = artifact
    return payload


def _handle_audio(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    if _task_kind == "text_to_speech":
        text = job_input.get("text") or job_input.get("prompt")
        if not text:
            return {"error": "Missing 'text' or 'prompt' in input", "status": "failed"}

        output = _pipeline(str(text))
        audio = output.get("audio")
        sampling_rate = int(output.get("sampling_rate", 22050))
        if audio is None:
            return {"error": "Audio pipeline returned no audio", "status": "failed"}

        buffer = io.BytesIO()
        sf.write(buffer, audio, sampling_rate, format="WAV")
        raw_bytes = buffer.getvalue()
        artifact = upload_bytes(raw_bytes, job, job_input, content_type="audio/wav", extension="wav")
        return {
            "model_id": HF_MODEL_ID,
            "task": "text_to_speech",
            "sampling_rate": sampling_rate,
            "artifact": artifact,
        }
    else:
        # Speech to text
        audio_url = job_input.get("audio_url") or job_input.get("audioUrl")
        temp_path = None
        try:
            if audio_url:
                temp_path = download_to_tempfile(str(audio_url), suffix=".wav")
                output = _pipeline(temp_path)
            else:
                return {"error": "Missing 'audio_url' in input (base64 deprecated)", "status": "failed"}
            
            return {
                "text": output.get("text") if isinstance(output, dict) else str(output),
                "model_id": HF_MODEL_ID,
                "task": "speech_to_text",
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    global _failure_count
    job_input = job.get("input") or {}
    
    if job_input.get("debug") is True:
        return {"status": "ok", "pipeline_loaded": _pipeline is not None, "task": _task_kind, "error": _load_error}
        
    if _pipeline is None:
        wait_start = time.time()
        while _pipeline is None and _load_error is None:
            if time.time() - wait_start > MODEL_LOAD_WAIT_TIMEOUT_SECONDS:
                return {"error": "Model failed to load within timeout.", "status": "loading"}
            time.sleep(2)
        if _load_error:
            return {"error": f"Model load failed: {_load_error}", "status": "failed"}

    try:
        global _last_request_at
        _last_request_at = time.time()
        started_at = time.time()

        if _task_kind == "text2img":
            result = _handle_image(job, job_input)
        elif _task_kind == "text2video":
            result = _handle_video(job, job_input)
        else:
            result = _handle_audio(job, job_input)

        if "error" not in result:
            result["execution_duration_ms"] = max(int((time.time() - started_at) * 1000), 0)
            _failure_count = 0
            
        return result
    except Exception as exc:
        import traceback
        _failure_count += 1
        log_tunnel("ERROR", f"Inference failed: {exc}")
        if _failure_count >= CLEANUP_FAILURE_THRESHOLD:
            request_cleanup("inference_failure_threshold")
        return {"error": str(exc), "traceback": traceback.format_exc(), "model_id": HF_MODEL_ID}


def main() -> None:
    import runpod

    def _idle_watchdog() -> None:
        while True:
            time.sleep(15)
            if _pipeline is None or _last_request_at == 0.0:
                continue
            if time.time() - _last_request_at > CLEANUP_IDLE_TIMEOUT_SECONDS:
                log_tunnel("WARNING", "Idle timeout reached, requesting cleanup")
                request_cleanup("idle_timeout")
                return

    threading.Thread(target=_load_model_background, daemon=True).start()
    threading.Thread(target=_idle_watchdog, daemon=True).start()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
