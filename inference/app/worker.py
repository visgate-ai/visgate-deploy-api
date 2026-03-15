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
    HF_TOKEN,
    HF_MODEL_ID,
    MODEL_LOAD_WAIT_TIMEOUT_SECONDS,
    VISGATE_WEBHOOK,
    VISGATE_INTERNAL_SECRET,
)
from app.loader import resolve_model_source
from app.runtime_common import (
    download_r2_artifact_to_tempfile,
    download_to_tempfile,
    log_tunnel,
    post_json,
    request_cleanup,
    upload_bytes,
)
from app.task_detector import detect_task
from PIL import Image

_pipeline: Optional[Any] = None
_load_error: Optional[str] = None
_last_request_at: float = 0.0
_failure_count: int = 0
_task_kind: str = "text2img"
_state_lock = threading.RLock()
_runtime_device: str = DEVICE


def _job_id(job: dict[str, Any]) -> str:
    return str(job.get("id") or job.get("jobId") or job.get("requestId") or "unknown")


def _log_request(job: dict[str, Any], level: str, message: str) -> None:
    job_id = _job_id(job)
    log_tunnel(level, f"job_id={job_id} {message}")


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
                "t_model_load_s": timings.get("t_model_load_s"),
                "loaded_from_cache": timings.get("loaded_from_cache"),
            }
        )
    
    headers = {}
    if VISGATE_INTERNAL_SECRET:
        headers["X-Visgate-Internal-Secret"] = VISGATE_INTERNAL_SECRET
        
    for attempt in range(5):
        try:
            post_json(VISGATE_WEBHOOK, payload, headers=headers)
            return
        except Exception:
            if attempt < 4:
                time.sleep(2 ** attempt)


def _resolve_runtime_device(preferred: str) -> str:
    if not preferred.startswith("cuda"):
        return preferred
    if not torch.cuda.is_available():
        log_tunnel("WARNING", "CUDA requested but no CUDA device is available; falling back to CPU")
        return "cpu"

    try:
        capability = torch.cuda.get_device_capability(0)
        arch = f"sm_{capability[0]}{capability[1]}"
        supported_arches = {value for value in torch.cuda.get_arch_list() if value.startswith("sm_")}
        if supported_arches and arch not in supported_arches:
            gpu_name = torch.cuda.get_device_name(0)
            log_tunnel(
                "WARNING",
                (
                    f"GPU architecture {arch} ({gpu_name}) is unsupported by current PyTorch build "
                    "— falling back to CPU"
                ),
            )
            return "cpu"
    except Exception as exc:
        log_tunnel("WARNING", f"Could not verify CUDA architecture compatibility: {exc}")
    return preferred


def _torch_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device.startswith("cuda") else torch.float32


def _runtime_video_model_id(model_id: str) -> str:
    aliases = {
        "Wan-AI/Wan2.1-T2V-1.3B": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    }
    return aliases.get(model_id, model_id)


def _load_speech_to_text_pipeline(model_source: str, device: str) -> Any:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    use_local_files = os.path.isdir(model_source)

    model_kwargs: dict[str, Any] = {
        "token": HF_TOKEN,
        "torch_dtype": _torch_dtype(device),
        "low_cpu_mem_usage": True,
        "local_files_only": use_local_files,
    }
    processor_kwargs: dict[str, Any] = {"token": HF_TOKEN, "local_files_only": use_local_files}

    loaded_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_source, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)

    if device.startswith("cuda"):
        loaded_model.to(device)

    return pipeline(
        "automatic-speech-recognition",
        model=loaded_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=_torch_dtype(device),
        device=0 if device.startswith("cuda") else -1,
    )


def _load_model_background() -> None:
    global _pipeline, _load_error, _task_kind, _runtime_device

    try:
        with _state_lock:
            _task_kind = detect_task(HF_MODEL_ID, os.environ.get("TASK", ""))
        log_tunnel("INFO", f"Detected task: {_task_kind} for model: {HF_MODEL_ID}")

        with _state_lock:
            task_kind = _task_kind
            _runtime_device = _resolve_runtime_device(DEVICE)
            runtime_device = _runtime_device
        log_tunnel("INFO", f"Using runtime device: {runtime_device}")
        effective_model_id = _runtime_video_model_id(HF_MODEL_ID) if task_kind == "text2video" else HF_MODEL_ID
        model_source, use_local, t_r2_sync_s, loaded_from_cache = resolve_model_source(effective_model_id)
        log_tunnel("INFO", f"Loading model source: {model_source}")

        _notify_orchestrator("loading_model", f"Model loading started ({task_kind})")
        t0 = time.time()

        loaded_pipeline: Optional[Any] = None
        if task_kind == "text2img":
            from pipelines.registry import load_pipeline
            loaded_pipeline = load_pipeline(model_id=model_source, token=HF_TOKEN, device=runtime_device)
        
        elif task_kind == "text2video":
            from diffusers import DiffusionPipeline
            loaded_pipeline = DiffusionPipeline.from_pretrained(
                model_source,
                torch_dtype=_torch_dtype(runtime_device),
                token=HF_TOKEN,
                local_files_only=os.path.isdir(model_source),
            )
            if runtime_device.startswith("cuda"):
                loaded_pipeline.to(runtime_device)
                
        elif task_kind == "speech_to_text":
            loaded_pipeline = _load_speech_to_text_pipeline(model_source, runtime_device)

        elif task_kind == "text_to_speech":
            from transformers import pipeline

            loaded_pipeline = pipeline(
                "text-to-audio",
                model=model_source,
                token=HF_TOKEN,
                device=0 if runtime_device.startswith("cuda") else -1,
            )

        with _state_lock:
            _pipeline = loaded_pipeline
            _load_error = None
            
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
        with _state_lock:
            _load_error = str(exc)
        log_tunnel("ERROR", f"Model load failed: {exc}")
        _notify_orchestrator("failed", _load_error)
        request_cleanup("startup_failure")
        traceback.print_exc()


def _handle_image(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    prompt = job_input.get("prompt")
    staged_image = job_input.get("input_image_r2") or job_input.get("image_r2")
    if not prompt and not job_input.get("input_image_url") and not staged_image:
        return {"error": "Missing 'prompt', 'input_image_url', or staged R2 image input"}

    kwargs = {}
    tmp_img = None
    if staged_image:
        try:
            _log_request(job, "INFO", "reading staged image input from R2")
            tmp_img = download_r2_artifact_to_tempfile(staged_image, ".png")
            kwargs["image"] = Image.open(tmp_img).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to download staged input image: {e}"}
    elif job_input.get("input_image_url"):
        try:
            _log_request(job, "INFO", f"reading image input from url={job_input['input_image_url']}")
            tmp_img = download_to_tempfile(job_input["input_image_url"], ".png")
            kwargs["image"] = Image.open(tmp_img).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to download input image: {e}"}

    try:
        _log_request(job, "INFO", "image processing started")
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
            artifact = upload_bytes(
                data,
                job,
                job_input,
                content_type="image/png",
                extension="png",
                required=True,
            )
            result["artifact"] = artifact
            result.pop("image_base64", None)  # Force remove base64
        elif result.get("file_path"):
            with open(result["file_path"], "rb") as f:
                data = f.read()
            artifact = upload_bytes(
                data,
                job,
                job_input,
                content_type=result.get("content_type", "image/png"),
                extension=result.get("file_extension", "png").lstrip("."),
                required=True,
            )
            result["artifact"] = artifact
            # Cleanup temp file from pipeline
            if os.path.exists(result["file_path"]):
                os.remove(result["file_path"])
        _log_request(job, "INFO", f"image processing finished artifact_key={(result.get('artifact') or {}).get('key')}")
        return result
    finally:
        if tmp_img and os.path.exists(tmp_img):
            os.remove(tmp_img)


def _handle_video(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing or empty 'prompt' in input", "status": "failed"}

    _log_request(job, "INFO", "video processing started")
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
            frames_iter = frames_raw[0]
        else:
            frames_iter = frames_raw
    else:
        if isinstance(frames_raw, list) and frames_raw and isinstance(frames_raw[0], list):
            frames_raw = frames_raw[0]
        frames_iter = frames_raw

    frame_count = 0
    tmp_name = ""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        with imageio.get_writer(tmp_name, fps=int(job_input.get("fps", 8))) as writer:
            for frame in frames_iter:
                arr = np.array(frame.convert("RGB")) if hasattr(frame, "convert") else np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = (arr * 255 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
                writer.append_data(arr)
                frame_count += 1

        with open(tmp_name, "rb") as fh:
            video_bytes = fh.read()
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.remove(tmp_name)

    artifact = upload_bytes(
        video_bytes,
        job,
        job_input,
        content_type="video/mp4",
        extension="mp4",
        required=True,
    )

    payload: dict[str, Any] = {
        "model_id": HF_MODEL_ID,
        "frame_count": frame_count,
        "artifact": artifact,
    }
    _log_request(job, "INFO", f"video processing finished frame_count={frame_count} artifact_key={(artifact or {}).get('key')}")
    return payload


def _handle_audio(job: dict[str, Any], job_input: dict[str, Any]) -> dict[str, Any]:
    if _task_kind == "text_to_speech":
        text = job_input.get("text") or job_input.get("prompt")
        if not text:
            return {"error": "Missing 'text' or 'prompt' in input", "status": "failed"}

        _log_request(job, "INFO", "text-to-speech processing started")
        output = _pipeline(str(text))
        audio = output.get("audio")
        sampling_rate = int(output.get("sampling_rate", 22050))
        if audio is None:
            return {"error": "Audio pipeline returned no audio", "status": "failed"}

        buffer = io.BytesIO()
        sf.write(buffer, audio, sampling_rate, format="WAV")
        raw_bytes = buffer.getvalue()
        artifact = upload_bytes(
            raw_bytes,
            job,
            job_input,
            content_type="audio/wav",
            extension="wav",
            required=True,
        )
        return {
            "model_id": HF_MODEL_ID,
            "task": "text_to_speech",
            "sampling_rate": sampling_rate,
            "artifact": artifact,
        }
    else:
        # Speech to text
        staged_audio = job_input.get("audio_r2")
        audio_url = job_input.get("audio_url") or job_input.get("audioUrl")
        temp_path = None
        try:
            if staged_audio:
                _log_request(job, "INFO", "reading staged audio input from R2")
                temp_path = download_r2_artifact_to_tempfile(staged_audio, ".wav")
                _log_request(job, "INFO", "speech-to-text processing started")
                output = _pipeline(temp_path)
            elif audio_url:
                _log_request(job, "INFO", f"reading audio input from url={audio_url}")
                temp_path = download_to_tempfile(str(audio_url), suffix=".wav")
                _log_request(job, "INFO", "speech-to-text processing started")
                output = _pipeline(temp_path)
            else:
                return {"error": "Missing 'audio_url' or staged audio input in input (base64 deprecated)", "status": "failed"}

            result = {
                "text": output.get("text") if isinstance(output, dict) else str(output),
                "model_id": HF_MODEL_ID,
                "task": "speech_to_text",
            }
            _log_request(job, "INFO", f"speech-to-text processing finished text_chars={len(result['text'])}")
            return result
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    global _failure_count
    job_input = job.get("input") or {}
    request_received_at = time.time()
    _log_request(
        job,
        "INFO",
        (
            f"request received task={_task_kind} keys={sorted(job_input.keys())} "
            f"has_staged_r2_input={bool(job_input.get('input_image_r2') or job_input.get('image_r2') or job_input.get('audio_r2'))}"
        ),
    )
    
    if job_input.get("debug") is True:
        with _state_lock:
            return {
                "status": "ok",
                "pipeline_loaded": _pipeline is not None,
                "task": _task_kind,
                "error": _load_error,
            }
        
    with _state_lock:
        pipeline = _pipeline
        load_error = _load_error

    if pipeline is None:
        wait_start = time.time()
        while True:
            with _state_lock:
                pipeline = _pipeline
                load_error = _load_error
            if pipeline is not None or load_error is not None:
                break
            if time.time() - wait_start > MODEL_LOAD_WAIT_TIMEOUT_SECONDS:
                return {"error": "Model failed to load within timeout.", "status": "loading"}
            time.sleep(2)
        if load_error:
            return {"error": f"Model load failed: {load_error}", "status": "failed"}

    try:
        global _last_request_at
        with _state_lock:
            _last_request_at = time.time()
            task_kind = _task_kind
            current_pipeline = _pipeline
        started_at = time.time()

        if current_pipeline is None:
            return {"error": "Pipeline was unloaded during request", "status": "failed"}

        if task_kind == "text2img":
            result = _handle_image(job, job_input)
        elif task_kind == "text2video":
            result = _handle_video(job, job_input)
        else:
            result = _handle_audio(job, job_input)

        if "error" not in result:
            result["execution_duration_ms"] = max(int((time.time() - started_at) * 1000), 0)
            total_duration_ms = max(int((time.time() - request_received_at) * 1000), 0)
            _log_request(
                job,
                "INFO",
                (
                    f"request completed task={task_kind} execution_duration_ms={result['execution_duration_ms']} "
                    f"total_duration_ms={total_duration_ms} artifact_key={(result.get('artifact') or {}).get('key')}"
                ),
            )
            with _state_lock:
                _failure_count = 0
        else:
            total_duration_ms = max(int((time.time() - request_received_at) * 1000), 0)
            _log_request(
                job,
                "ERROR",
                f"request returned error task={task_kind} total_duration_ms={total_duration_ms} error={result.get('error')}",
            )

        return result
    except Exception as exc:
        import traceback as _tb
        with _state_lock:
            _failure_count += 1
            failure_count = _failure_count
        total_duration_ms = max(int((time.time() - request_received_at) * 1000), 0)
        _log_request(
            job,
            "ERROR",
            f"request failed task={_task_kind} total_duration_ms={total_duration_ms} error={exc} failure_count={failure_count}",
        )
        if failure_count >= CLEANUP_FAILURE_THRESHOLD:
            request_cleanup("inference_failure_threshold")
        # Sanitize traceback to avoid leaking secrets
        raw_tb = _tb.format_exc()
        import re
        sanitized_tb = re.sub(r'(hf_|rpa_|sk-|token=)[A-Za-z0-9_]+', r'\1***', raw_tb)
        return {"error": str(exc), "traceback": sanitized_tb, "model_id": HF_MODEL_ID}
    finally:
        # VRAM cleanup between requests to prevent fragmentation and OOM
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _idle_watchdog() -> None:
    while True:
        time.sleep(15)
        with _state_lock:
            pipeline = _pipeline
            last_request_at = _last_request_at
        if pipeline is None or last_request_at == 0.0:
            continue
        if time.time() - last_request_at > CLEANUP_IDLE_TIMEOUT_SECONDS:
            log_tunnel("WARNING", "Idle timeout reached, requesting cleanup")
            request_cleanup("idle_timeout")
            return


# ── HTTP server mode for non-RunPod providers (Vast.ai, etc.) ──────────────

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _run_job_async(job_id: str, job: dict) -> None:
    """Execute an inference job in a background thread and store the result."""
    try:
        result = handler(job)
        with _jobs_lock:
            _jobs[job_id] = {"status": "COMPLETED", "output": result}
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id] = {"status": "FAILED", "error": str(exc)}


def _run_http_server(port: int = 8000) -> None:
    """Standalone HTTP server for non-RunPod providers (Vast.ai, etc.).

    Routes:
        POST /run        — async: queues job, returns job ID
        POST /runsync    — sync: runs job and returns result
        GET  /status/<id> — poll job status
        GET  /health     — worker health check
    """
    import json
    import uuid as _uuid
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class InferenceHTTPHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            path = self.path.rstrip("/")
            if path in ("/run", "/v2/run"):
                self._handle_run_async()
            elif path in ("/runsync", "/v2/runsync"):
                self._handle_runsync()
            else:
                self._respond(404, {"error": "not found"})

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/health":
                self._handle_health()
            elif path.startswith("/status/"):
                job_id = path.split("/status/", 1)[1]
                self._handle_status(job_id)
            else:
                self._respond(404, {"error": "not found"})

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length))

        def _handle_run_async(self):
            body = self._read_body()
            job_input = body.get("input", body)
            job_id = str(_uuid.uuid4())
            job = {"id": job_id, "input": job_input}
            with _jobs_lock:
                _jobs[job_id] = {"status": "IN_PROGRESS"}
            threading.Thread(target=_run_job_async, args=(job_id, job), daemon=True).start()
            self._respond(200, {"id": job_id, "status": "IN_QUEUE"})

        def _handle_runsync(self):
            body = self._read_body()
            job_input = body.get("input", body)
            job_id = str(_uuid.uuid4())
            job = {"id": job_id, "input": job_input}
            try:
                result = handler(job)
                self._respond(200, {"id": job_id, "status": "COMPLETED", "output": result})
            except Exception as exc:
                self._respond(500, {"id": job_id, "status": "FAILED", "error": str(exc)})

        def _handle_health(self):
            with _state_lock:
                ready = _pipeline is not None
                error = _load_error
            workers = {
                "ready": 1 if ready else 0,
                "idle": 1 if (ready and _last_request_at == 0.0) else 0,
                "initializing": 1 if (not ready and error is None) else 0,
                "running": 0,
            }
            self._respond(200, {"workers": workers, "error": error})

        def _handle_status(self, job_id: str):
            with _jobs_lock:
                entry = _jobs.get(job_id)
            if entry is None:
                self._respond(404, {"id": job_id, "status": "NOT_FOUND", "error": "Job not found"})
                return
            resp: dict = {"id": job_id, "status": entry["status"]}
            if "output" in entry:
                resp["output"] = entry["output"]
            if "error" in entry:
                resp["error"] = entry["error"]
            self._respond(200, resp)

        def _respond(self, code: int, data: dict):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, fmt, *args):
            # Suppress default stderr logging to keep output clean
            pass

    print(f"[visgate] Starting HTTP inference server on port {port}")
    HTTPServer(("0.0.0.0", port), InferenceHTTPHandler).serve_forever()


def main() -> None:
    mode = os.environ.get("WORKER_MODE", "runpod").lower()

    threading.Thread(target=_load_model_background, daemon=True).start()
    threading.Thread(target=_idle_watchdog, daemon=True).start()

    if mode == "http":
        http_port = int(os.environ.get("HTTP_PORT", "8000"))
        _run_http_server(port=http_port)
    else:
        import runpod
        runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
