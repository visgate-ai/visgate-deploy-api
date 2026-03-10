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
    MODEL_LOAD_WAIT_TIMEOUT_SECONDS,
    VISGATE_WEBHOOK,
)
from app.loader import resolve_model_source
from app.runtime_common import log_tunnel, post_json, request_cleanup, upload_bytes, download_to_tempfile
from PIL import Image

_pipeline: Optional[Any] = None
_load_error: Optional[str] = None
_last_request_at: float = 0.0
_failure_count: int = 0


def _runtime_video_model_id(model_id: str) -> str:
    aliases = {
        "Wan-AI/Wan2.1-T2V-1.3B": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    }
    return aliases.get(model_id, model_id)


def _notify_orchestrator(
    status: str,
    message: str | None = None,
    timings: dict[str, float | bool | None] | None = None,
) -> None:
    if not VISGATE_WEBHOOK or not VISGATE_WEBHOOK.strip():
        return
    payload = {"status": status}
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


def _load_model_background() -> None:
    global _pipeline, _load_error
    from diffusers import DiffusionPipeline

    try:
        effective_model_id = _runtime_video_model_id(HF_MODEL_ID)
        model_source, _, t_r2_sync_s, loaded_from_cache = resolve_model_source(effective_model_id)
        log_tunnel("INFO", f"Loading video model: requested={HF_MODEL_ID} runtime={effective_model_id}")
        _notify_orchestrator("loading_model", "Video model loading started")
        t0 = time.time()
        _pipeline = DiffusionPipeline.from_pretrained(model_source, torch_dtype=_torch_dtype(), token=HF_TOKEN)
        load_elapsed = time.time() - t0
        if DEVICE.startswith("cuda"):
            t1 = time.time()
            _pipeline.to(DEVICE)
            gpu_elapsed = time.time() - t1
            log_tunnel("INFO", f"Pipeline to GPU: {gpu_elapsed:.1f}s")
            load_elapsed += gpu_elapsed
        log_tunnel("INFO", f"Video pipeline loaded in {load_elapsed:.1f}s (from_pretrained)")
        _notify_orchestrator(
            "ready",
            "Video model loaded successfully",
            timings={
                "t_r2_sync_s": round(t_r2_sync_s, 3) if t_r2_sync_s is not None else None,
                "t_model_load_s": round(load_elapsed, 3),
                "loaded_from_cache": loaded_from_cache,
            },
        )
        log_tunnel("INFO", "Video model loaded successfully")
    except Exception as exc:
        _load_error = str(exc)
        log_tunnel("ERROR", f"Video model load failed: {exc}")
        _notify_orchestrator("failed", _load_error)
        request_cleanup("startup_failure")


def _extract_frames(output: Any) -> list[np.ndarray]:
    frames_raw = None
    if hasattr(output, "frames") and output.frames is not None:
        frames_raw = output.frames
    elif isinstance(output, dict) and output.get("frames") is not None:
        frames_raw = output["frames"]
    elif hasattr(output, "images") and output.images is not None:
        frames_raw = output.images

    if frames_raw is None:
        raise RuntimeError("Video pipeline returned no frames")

    # WAN diffusers 0.33: frames is np.ndarray (batch, num_frames, H, W, C)
    # or List[List[PIL]] or List[np.ndarray]
    if isinstance(frames_raw, np.ndarray):
        if frames_raw.ndim == 5:
            frames_raw = frames_raw[0]  # (num_frames, H, W, C)
        arrays = [frames_raw[i] for i in range(len(frames_raw))]
    elif isinstance(frames_raw, list):
        # Batch wrapper: [[frame, frame, ...]]
        if frames_raw and isinstance(frames_raw[0], (list, np.ndarray)) and not hasattr(frames_raw[0], "convert"):
            if isinstance(frames_raw[0], np.ndarray) and frames_raw[0].ndim >= 3:
                # List[np.ndarray] where each is (H, W, C) — already flat
                arrays = [np.asarray(f) for f in frames_raw]
            else:
                frames_raw = frames_raw[0]
                arrays = []
                for frame in frames_raw:
                    arrays.append(np.array(frame.convert("RGB")) if hasattr(frame, "convert") else np.asarray(frame))
        else:
            arrays = []
            for frame in frames_raw:
                arrays.append(np.array(frame.convert("RGB")) if hasattr(frame, "convert") else np.asarray(frame))
    else:
        raise RuntimeError(f"Unexpected frames type: {type(frames_raw)}")

    # Ensure uint8 [0,255] for imageio
    result = []
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        result.append(arr)
    return result


def handler(job: dict[str, Any]) -> dict[str, Any]:
    global _last_request_at, _failure_count
    job_input = job.get("input") or {}
    if _pipeline is None:
        wait_start = time.time()
        while _pipeline is None and _load_error is None:
            if time.time() - wait_start > MODEL_LOAD_WAIT_TIMEOUT_SECONDS:
                return {"error": "Model failed to load within timeout.", "status": "loading"}
            time.sleep(2)
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