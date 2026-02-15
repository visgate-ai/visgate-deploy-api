"""
Runpod serverless worker: load HF model at startup, notify orchestrator when ready, handle inference jobs.
"""

import os
import sys
import threading
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
)
from app.loader import load_pipeline_optimized as load_pipeline


# Global pipeline (loaded once at startup)
_pipeline: Optional[Any] = None
_load_error: Optional[str] = None


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
    req = urllib.request.Request(
        VISGATE_WEBHOOK,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
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
    try:
        _notify_orchestrator("loading_model", "Model loading started")
        _pipeline = load_pipeline(model_id=HF_MODEL_ID, token=HF_TOKEN, device=DEVICE)
        print("[worker] Model loaded successfully", flush=True)
        # Notify orchestrator
        _notify_orchestrator("ready", "Model loaded successfully")
    except Exception as e:
        import traceback
        _load_error = str(e)
        print(f"[worker] Failed to load model: {e}", flush=True)
        _notify_orchestrator("failed", _load_error)
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
        return {
            "env": dict(os.environ), 
            "status": "ok", 
            "pipeline_loaded": _pipeline is not None,
            "HF_MODEL_ID_PYTHON": HF_MODEL_ID,
            "load_error": _load_error
        }

    if _pipeline is None:
        return {"error": "Model is still loading, please retry in a few seconds.", "status": "loading"}

    prompt = job_input.get("prompt")
    if not prompt or not str(prompt).strip():
        return {"error": "Missing or empty 'prompt' in input"}
    try:
        result = _pipeline.run(
            prompt=str(prompt).strip(),
            num_inference_steps=int(job_input.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)),
            guidance_scale=float(job_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)),
            height=job_input.get("height"),
            width=job_input.get("width"),
            seed=job_input.get("seed"),
        )
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "model_id": HF_MODEL_ID}


def main() -> None:
    import traceback
    print("[worker] Starting Inference Worker v1.7 (Background Load)", flush=True)
    try:
        import runpod
        
        # Start model loading in background thread
        threading.Thread(target=_load_model_background, daemon=True).start()
        
        print("[worker] Starting Runpod listener immediately...", flush=True)
        runpod.serverless.start({"handler": handler})
    except Exception:
        print("[worker] CRITICAL ERROR during startup:", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
