"""
Runpod serverless worker: load HF model at startup, notify orchestrator when ready, handle inference jobs.
"""

import os
import sys
import threading
import time
from typing import Any, Optional

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEVICE,
    HF_MODEL_ID,
    HF_TOKEN,
    VISGATE_WEBHOOK,
    VISGATE_LOG_TUNNEL,
    VISGATE_INTERNAL_SECRET,
    VISGATE_DEPLOYMENT_ID,
    CLEANUP_FAILURE_THRESHOLD,
    CLEANUP_IDLE_TIMEOUT_SECONDS,
    OUTPUT_S3_URL,
    CDN_BASE_URL,
    RETURN_BASE64,
)
from app.loader import load_pipeline_optimized as load_pipeline


# Global pipeline (loaded once at startup)
_pipeline: Optional[Any] = None
_load_error: Optional[str] = None
_last_request_at: float = 0.0
_failure_count: int = 0


def _mask_sensitive(text: str) -> str:
    if not text:
        return text
    for token in ("hf_", "rpa_", "sk_", "api_key", "token", "secret"):
        if token in text.lower():
            return "***REDACTED***"
    return text


def _post_json(url: str, payload: dict[str, Any]) -> None:
    import json
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if VISGATE_INTERNAL_SECRET:
        headers["X-Visgate-Internal-Secret"] = VISGATE_INTERNAL_SECRET
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=10):
        return


def _log_tunnel(level: str, message: str) -> None:
    if not VISGATE_LOG_TUNNEL or not VISGATE_LOG_TUNNEL.strip():
        return
    try:
        _post_json(VISGATE_LOG_TUNNEL, {"level": level, "message": _mask_sensitive(message)})
    except Exception:
        return


def _request_cleanup(reason: str) -> None:
    if not VISGATE_DEPLOYMENT_ID or not VISGATE_WEBHOOK:
        return
    cleanup_url = VISGATE_WEBHOOK.replace("/deployment-ready/", "/cleanup/")
    try:
        _post_json(cleanup_url, {"reason": reason})
    except Exception:
        return


def _maybe_upload_output(image_base64: str) -> Optional[str]:
    """Upload base64 output to S3 via s5cmd and return CDN URL if configured."""
    if not OUTPUT_S3_URL:
        return None
    import base64
    import os
    import subprocess
    import tempfile
    try:
        subprocess.run(["s5cmd", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None

    key_name = f"visgate/{int(time.time())}.png"
    target = f"{OUTPUT_S3_URL.rstrip('/')}/{key_name}"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(base64.b64decode(image_base64))
            tmp_path = tmp.name
        subprocess.run(["s5cmd", "cp", tmp_path, target], check=True)
        if CDN_BASE_URL:
            return f"{CDN_BASE_URL.rstrip('/')}/{key_name}"
        return target
    except Exception:
        return None
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _notify_orchestrator(status: str, message: str | None = None) -> None:
    """POST worker lifecycle status to orchestrator internal webhook."""
    if not VISGATE_WEBHOOK or not VISGATE_WEBHOOK.strip():
        return
    
    import json
    import time
    import urllib.request
    import urllib.error

    print(f"[worker] Sending status={status} to {VISGATE_WEBHOOK}...", flush=True)

    payload = {"status": status}
    if message:
        payload["message"] = message
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if VISGATE_INTERNAL_SECRET:
        headers["X-Visgate-Internal-Secret"] = VISGATE_INTERNAL_SECRET
    req = urllib.request.Request(VISGATE_WEBHOOK, data=data, headers=headers, method="POST")
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    print("[worker] Webhook delivered successfully", flush=True)
                    return
                print(f"[worker] Webhook returned status {resp.status}", flush=True)
        except urllib.error.HTTPError as e:
             print(f"[worker] Webhook failed (attempt {attempt+1}): HTTP {e.code} {e.reason}", flush=True)
        except Exception as e:
            print(f"[worker] Webhook failed (attempt {attempt+1}): {e}", flush=True)
        
        if attempt < 2:
            time.sleep(2 ** attempt)

    print("[worker] ERROR: Failed to notify orchestrator after retries", flush=True)


def _load_model_background() -> None:
    global _pipeline, _load_error
    if not HF_MODEL_ID or not HF_MODEL_ID.strip():
        _load_error = "HF_MODEL_ID environment variable is required"
        print(f"[worker] ERROR: {_load_error}", flush=True)
        return

    print(f"[worker] Loading model in background: {HF_MODEL_ID}", flush=True)
    _log_tunnel("INFO", f"Loading model: {HF_MODEL_ID}")
    try:
        _notify_orchestrator("loading_model", "Model loading started")
        _pipeline = load_pipeline(model_id=HF_MODEL_ID, token=HF_TOKEN, device=DEVICE)
        print("[worker] Model loaded successfully", flush=True)
        _log_tunnel("INFO", "Model loaded successfully")
        # Notify orchestrator
        _notify_orchestrator("ready", "Model loaded successfully")
    except Exception as e:
        import traceback
        _load_error = str(e)
        print(f"[worker] Failed to load model: {e}", flush=True)
        _log_tunnel("ERROR", f"Model load failed: {e}")
        _notify_orchestrator("failed", _load_error)
        _request_cleanup("startup_failure")
        traceback.print_exc()


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    Runpod job handler.
    """
    global _pipeline, _load_error
    job_input = job.get("input") or {}
    
    # Allow debug env dump
    if job_input.get("debug") is True:
        import os
        safe_env = {}
        for k, v in os.environ.items():
            if any(s in k.upper() for s in ("TOKEN", "SECRET", "KEY", "AUTH")):
                safe_env[k] = "***REDACTED***"
            else:
                safe_env[k] = v
        return {
            "env": safe_env,
            "status": "ok",
            "pipeline_loaded": _pipeline is not None,
            "HF_MODEL_ID_PYTHON": HF_MODEL_ID,
            "load_error": _load_error,
        }

    if _pipeline is None:
        return {"error": "Model is still loading, please retry in a few seconds.", "status": "loading"}

    prompt = job_input.get("prompt")
    if not prompt or not str(prompt).strip():
        return {"error": "Missing or empty 'prompt' in input"}
    try:
        global _last_request_at, _failure_count
        _last_request_at = time.time()
        _log_tunnel("INFO", "Inference started")
        result = _pipeline.run(
            prompt=str(prompt).strip(),
            num_inference_steps=int(job_input.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)),
            guidance_scale=float(job_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)),
            height=job_input.get("height"),
            width=job_input.get("width"),
            seed=job_input.get("seed"),
        )
        if OUTPUT_S3_URL and result.get("image_base64"):
            upload_info = _maybe_upload_output(result["image_base64"])
            if upload_info:
                result["cdn_url"] = upload_info
                if RETURN_BASE64.lower() != "true":
                    result.pop("image_base64", None)
        _log_tunnel("INFO", "Inference completed")
        _failure_count = 0
        return result
    except Exception as e:
        import traceback
        _failure_count += 1
        _log_tunnel("ERROR", f"Inference failed: {e}")
        if _failure_count >= CLEANUP_FAILURE_THRESHOLD:
            _request_cleanup("inference_failure_threshold")
        return {"error": str(e), "traceback": traceback.format_exc(), "model_id": HF_MODEL_ID}


def main() -> None:
    import traceback
    print("[worker] Starting Inference Worker v1.7 (Background Load)", flush=True)
    try:
        import runpod

        def _idle_watchdog() -> None:
            global _last_request_at
            while True:
                time.sleep(15)
                if _pipeline is None:
                    continue
                if _last_request_at == 0.0:
                    continue
                idle_seconds = time.time() - _last_request_at
                if idle_seconds > CLEANUP_IDLE_TIMEOUT_SECONDS:
                    _log_tunnel("WARNING", f"Idle timeout reached: {idle_seconds:.0f}s")
                    _request_cleanup("idle_timeout")
                    return
        
        # Start model loading in background thread
        threading.Thread(target=_load_model_background, daemon=True).start()
        threading.Thread(target=_idle_watchdog, daemon=True).start()
        
        print("[worker] Starting Runpod listener immediately...", flush=True)
        runpod.serverless.start({"handler": handler})
    except Exception:
        print("[worker] CRITICAL ERROR during startup:", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
