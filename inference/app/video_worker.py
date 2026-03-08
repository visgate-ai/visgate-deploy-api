"""Video-focused RunPod worker."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from typing import Any, Optional

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    CLEANUP_FAILURE_THRESHOLD,
    CLEANUP_IDLE_TIMEOUT_SECONDS,
    DEVICE,
    HF_MODEL_ID,
    HF_TOKEN,
    VISGATE_WEBHOOK,
)
from app.loader import resolve_model_source
from app.runtime_common import log_tunnel, post_json, request_cleanup, upload_bytes

_pipeline: Optional[Any] = None
_load_error: Optional[str] = None
_last_request_at: float = 0.0
_failure_count: int = 0


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


def _torch_dtype() -> torch.dtype:
    if DEVICE.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if DEVICE.startswith("cuda") else torch.float32


def _load_model_background() -> None:
    global _pipeline, _load_error
    from diffusers import DiffusionPipeline

    try:
        model_source, _ = resolve_model_source(HF_MODEL_ID)
        log_tunnel("INFO", f"Loading video model: {HF_MODEL_ID}")
        _notify_orchestrator("loading_model", "Video model loading started")
        _pipeline = DiffusionPipeline.from_pretrained(model_source, torch_dtype=_torch_dtype(), token=HF_TOKEN)
        if DEVICE.startswith("cuda"):
            _pipeline.to(DEVICE)
        _notify_orchestrator("ready", "Video model loaded successfully")
        log_tunnel("INFO", "Video model loaded successfully")
    except Exception as exc:
        _load_error = str(exc)
        log_tunnel("ERROR", f"Video model load failed: {exc}")
        _notify_orchestrator("failed", _load_error)
        request_cleanup("startup_failure")


def _extract_frames(output: Any) -> list[np.ndarray]:
    if hasattr(output, "frames") and output.frames:
        frames = output.frames[0] if isinstance(output.frames, list) and output.frames and isinstance(output.frames[0], list) else output.frames
    elif isinstance(output, dict) and output.get("frames"):
        frames = output["frames"]
        if frames and isinstance(frames[0], list):
            frames = frames[0]
    elif hasattr(output, "images") and output.images:
        frames = output.images
    else:
        raise RuntimeError("Video pipeline returned no frames")
    arrays: list[np.ndarray] = []
    for frame in frames:
        if hasattr(frame, "convert"):
            arrays.append(np.array(frame.convert("RGB")))
        else:
            arrays.append(np.asarray(frame))
    return arrays


def handler(job: dict[str, Any]) -> dict[str, Any]:
    global _last_request_at, _failure_count
    job_input = job.get("input") or {}
    if _pipeline is None:
        wait_start = time.time()
        while _pipeline is None and _load_error is None:
            if time.time() - wait_start > 600:
                return {"error": "Model failed to load within timeout.", "status": "loading"}
            time.sleep(5)
        if _load_error:
            return {"error": f"Model load failed: {_load_error}", "status": "failed"}
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing or empty 'prompt' in input", "status": "failed"}
    try:
        _last_request_at = time.time()
        started_at = time.time()
        output = _pipeline(
            prompt=str(prompt),
            num_inference_steps=int(job_input.get("num_inference_steps", 16)),
            guidance_scale=float(job_input.get("guidance_scale", 7.5)),
            num_frames=int(job_input.get("num_frames", 16)),
        )
        frames = _extract_frames(output)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            imageio.mimwrite(tmp.name, frames, fps=int(job_input.get("fps", 8)))
            tmp.flush()
            with open(tmp.name, "rb") as fh:
                video_bytes = fh.read()
        artifact = upload_bytes(video_bytes, job, job_input, content_type="video/mp4", extension="mp4")
        os.remove(tmp.name)
        result: dict[str, Any] = {
            "model_id": HF_MODEL_ID,
            "frame_count": len(frames),
            "execution_duration_ms": max(int((time.time() - started_at) * 1000), 0),
        }
        if artifact:
            result["artifact"] = artifact
        _failure_count = 0
        return result
    except Exception as exc:
        _failure_count += 1
        log_tunnel("ERROR", f"Video inference failed: {exc}")
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