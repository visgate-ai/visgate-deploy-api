"""
Runpod serverless worker: load HF model at startup, notify orchestrator when ready, handle inference jobs.
"""

import os
import sys
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
from pipelines.registry import load_pipeline

# Global pipeline (loaded once at startup)
_pipeline: Optional[Any] = None


def _notify_ready() -> None:
    """POST to orchestrator internal webhook when model is loaded."""
    if not VISGATE_WEBHOOK or not VISGATE_WEBHOOK.strip():
        return
    
    import json
    import time
    import urllib.request
    import urllib.error

    print(f"[worker] Sending ready signal to {VISGATE_WEBHOOK}...", flush=True)
    
    data = json.dumps({"status": "ready"}).encode("utf-8")
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


def _load_model() -> None:
    global _pipeline
    if not HF_MODEL_ID or not HF_MODEL_ID.strip():
        raise RuntimeError("HF_MODEL_ID environment variable is required")
    print(f"[worker] Loading model: {HF_MODEL_ID}", flush=True)
    _pipeline = load_pipeline(model_id=HF_MODEL_ID, token=HF_TOKEN, device=DEVICE)
    print("[worker] Model loaded successfully", flush=True)
    _notify_ready()


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    Runpod job handler. Input: { "prompt": str, "num_inference_steps": int?, "guidance_scale": float?, "height": int?, "width": int?, "seed": int? }.
    Output: { "image_base64": str, "model_id": str, ... } or { "error": str }.
    """
    global _pipeline
    if _pipeline is None:
        return {"error": "Model not loaded"}
    job_input = job.get("input") or {}
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
        return {"error": str(e), "model_id": HF_MODEL_ID}


def main() -> None:
    print("[worker] Starting Inference Worker v1.2", flush=True)
    _load_model()
    import runpod
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
