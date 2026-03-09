#!/usr/bin/env python3
"""
End-to-end smoke test for all three modalities: image, audio, video.

Runs them sequentially (RunPod quota is 5 workers max).
For each modality:
  1. Create deployment → poll until ready
  2. Submit inference job → poll until completed
  3. Verify the artifact / output
  4. Delete deployment

Usage:
    RUNPOD_API_KEY="rpa_..." python scripts/e2e_multimodal_smoke.py

Optional env:
    API_BASE          (default: production URL)
    GCP_PROJECT_ID    (default: visgate)
    SKIP_MODALITIES   comma-separated list to skip, e.g. "audio,video"
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from typing import Any

import requests

API = os.environ.get(
    "API_BASE", "https://visgate-deploy-api-wxup7pxrsa-ey.a.run.app"
).rstrip("/")
RUNPOD_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
PROJECT = os.environ.get("GCP_PROJECT_ID", "visgate")
SKIP = {s.strip().lower() for s in os.environ.get("SKIP_MODALITIES", "").split(",") if s.strip()}

HEADERS = {"Authorization": f"Bearer {RUNPOD_KEY}", "Content-Type": "application/json"}

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"
B = "\033[1m"; X = "\033[0m"


# ── helpers ──────────────────────────────────────────────────────────────────

def read_secret(name: str) -> str:
    return subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest",
         f"--secret={name}", "--project", PROJECT],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def clip(obj: Any, limit: int = 500) -> str:
    s = json.dumps(obj, ensure_ascii=True)
    return s if len(s) <= limit else s[:limit] + "…"


def section(title: str) -> None:
    print(f"\n{B}{'─'*64}\n  {title}\n{'─'*64}{X}")


def ok(msg: str) -> None:
    print(f"  {G}✅  {msg}{X}")


def fail(msg: str) -> None:
    print(f"  {R}❌  {msg}{X}")


def info(msg: str) -> None:
    print(f"  {C}ℹ   {msg}{X}")


# ── deployment lifecycle ──────────────────────────────────────────────────────

def create_deployment(model_id: str, task: str, cache_scope: str = "shared") -> str:
    r = requests.post(
        f"{API}/v1/deployments",
        headers=HEADERS,
        json={"hf_model_id": model_id, "task": task, "cache_scope": cache_scope},
        timeout=90,
    )
    r.raise_for_status()
    dep_id = r.json()["deployment_id"]
    info(f"Deployment created: {dep_id}")
    return dep_id


def wait_ready(dep_id: str, *, timeout_s: int = 720) -> dict:
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        r = requests.get(
            f"{API}/v1/deployments/{dep_id}",
            headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
            timeout=60,
        )
        r.raise_for_status()
        d = r.json()
        status = d.get("status")
        elapsed = int(time.time() - (deadline - timeout_s))
        print(f"    poll {attempt:>3} [{elapsed:>4}s] {status}", flush=True)
        if status in {"ready", "failed", "deleted", "webhook_failed"}:
            return d
        time.sleep(10)
    raise TimeoutError(f"Deployment {dep_id} did not become ready in {timeout_s}s")


def delete_deployment(dep_id: str) -> None:
    try:
        r = requests.delete(
            f"{API}/v1/deployments/{dep_id}",
            headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
            timeout=60,
        )
        info(f"DELETE {dep_id} → {r.status_code}")
    except Exception as exc:
        info(f"DELETE error: {exc}")


# ── inference ─────────────────────────────────────────────────────────────────

def submit_job(dep_id: str, inp: dict, s3_cfg: dict | None = None) -> str:
    body: dict = {"deployment_id": dep_id, "input": inp}
    if s3_cfg:
        body["s3Config"] = s3_cfg
    r = requests.post(
        f"{API}/v1/inference/jobs",
        headers=HEADERS,
        json=body,
        timeout=90,
    )
    r.raise_for_status()
    job_id = r.json()["job_id"]
    info(f"Job submitted: {job_id}")
    return job_id


def wait_job(job_id: str, *, timeout_s: int = 300) -> dict:
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        r = requests.get(
            f"{API}/v1/inference/jobs/{job_id}",
            headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
            timeout=60,
        )
        r.raise_for_status()
        d = r.json()
        status = d.get("status")
        metrics = d.get("metrics") or {}
        elapsed = int(time.time() - (deadline - timeout_s))
        print(
            f"    job {attempt:>3} [{elapsed:>4}s] {status}"
            f" exec={metrics.get('execution_ms', '?')}ms",
            flush=True,
        )
        if status in {"completed", "failed", "cancelled", "expired"}:
            return d
        time.sleep(5)
    raise TimeoutError(f"Job {job_id} did not complete in {timeout_s}s")


# ── modality smoke functions ───────────────────────────────────────────────────

def smoke_image(s3_cfg: dict) -> bool:
    section("1/3  IMAGE  —  stabilityai/sdxl-turbo  →  text_to_image")
    dep_id = None
    passed = False
    try:
        dep_id = create_deployment("stabilityai/sdxl-turbo", "text_to_image")
        d = wait_ready(dep_id)
        if d.get("status") != "ready":
            fail(f"Deployment failed: {d.get('error')}")
            return False
        ok(f"Deployment ready — GPU: {d.get('gpu_allocated', '?')}")

        job_id = submit_job(
            dep_id,
            {"prompt": "a golden retriever puppy on a sunny beach", "num_inference_steps": 1},
            s3_cfg,
        )
        job = wait_job(job_id)

        if job.get("status") != "completed":
            fail(f"Job failed: {job.get('error') or clip(job)}")
            return False

        artifact = (job.get("artifact") or job.get("output", {}).get("artifact") or {})
        if artifact.get("url") and artifact.get("bytes", 0) > 0:
            ok(f"Image artifact: {artifact['bytes']:,} bytes → {artifact.get('key')}")
            ok(f"Execution: {(job.get('metrics') or {}).get('execution_ms')}ms")
            passed = True
        else:
            # fall back to base64
            output = job.get("output") or {}
            b64 = output.get("image_base64") or ""
            if len(b64) > 100:
                ok(f"Image returned as base64 ({len(b64)} chars)")
                passed = True
            else:
                fail(f"No image artifact in output: {clip(job.get('output'))}")
    except Exception as exc:
        fail(f"Exception: {exc}")
    finally:
        if dep_id:
            delete_deployment(dep_id)
    return passed


def _generate_speech_wav_b64() -> str:
    """Generate a 2-second 440 Hz sine wave WAV at 16 kHz, base64-encoded.

    Whisper will not produce meaningful text, but the pipeline executes
    end-to-end and the job reaches `completed` — which is what we verify.
    """
    import base64
    import io
    import math
    import struct
    import wave

    sample_rate = 16000
    duration = 2.0
    freq = 440.0
    n_samples = int(sample_rate * duration)
    samples = [int(32767 * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n_samples)]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def smoke_audio(s3_cfg: dict) -> bool:
    section("2/3  AUDIO  —  openai/whisper-large-v3  →  speech_to_text")
    dep_id = None
    passed = False
    try:
        dep_id = create_deployment("openai/whisper-large-v3", "speech_to_text")
        d = wait_ready(dep_id)
        if d.get("status") != "ready":
            fail(f"Deployment failed: {d.get('error')}")
            return False
        ok(f"Deployment ready — GPU: {d.get('gpu_allocated', '?')}")

        # Use inline base64 WAV to avoid any external URL dependency
        audio_b64 = _generate_speech_wav_b64()
        job_id = submit_job(dep_id, {"audio_base64": audio_b64}, s3_cfg or None)
        job = wait_job(job_id)

        if job.get("status") != "completed":
            fail(f"Job failed: {job.get('error') or clip(job)}")
            return False

        output = job.get("output") or {}
        text = output.get("text") or ""
        # Whisper may return empty or hallucinated text for a tone — success
        # is defined by the pipeline completing without error.
        ok(f"Speech-to-text completed — transcription: «{text[:120] or '(empty)'}»")
        ok(f"Execution: {(job.get('metrics') or {}).get('execution_ms')}ms")
        passed = True
    except Exception as exc:
        fail(f"Exception: {exc}")
    finally:
        if dep_id:
            delete_deployment(dep_id)
    return passed


def smoke_video(s3_cfg: dict) -> bool:
    section("3/3  VIDEO  —  Wan-AI/Wan2.1-T2V-1.3B  →  text_to_video")
    dep_id = None
    passed = False
    try:
        dep_id = create_deployment("Wan-AI/Wan2.1-T2V-1.3B", "text_to_video")
        d = wait_ready(dep_id)
        if d.get("status") != "ready":
            fail(f"Deployment failed: {d.get('error')}")
            return False
        ok(f"Deployment ready — GPU: {d.get('gpu_allocated', '?')}")

        job_id = submit_job(
            dep_id,
            {
                "prompt": "A red car driving on a coastal road at sunset",
                "num_inference_steps": 2,
                "num_frames": 2,
                "fps": 2,
                "guidance_scale": 4.0,
            },
            s3_cfg,
        )
        job = wait_job(job_id, timeout_s=300)

        if job.get("status") != "completed":
            fail(f"Job failed: {job.get('error') or clip(job)}")
            return False

        artifact = (job.get("artifact") or job.get("output", {}).get("artifact") or {})
        if artifact.get("url") and artifact.get("bytes", 0) > 0:
            ok(f"Video artifact: {artifact['bytes']:,} bytes → {artifact.get('key')}")
            ok(f"Execution: {(job.get('metrics') or {}).get('execution_ms')}ms")
            passed = True
        else:
            fail(f"No video artifact in output: {clip(job.get('output'))}")
    except Exception as exc:
        fail(f"Exception: {exc}")
    finally:
        if dep_id:
            delete_deployment(dep_id)
    return passed


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    if not RUNPOD_KEY:
        raise SystemExit("Set RUNPOD_API_KEY")

    print(f"{B}Visgate E2E Multimodal Smoke Test{X}")
    print(f"API:     {API}")
    print(f"Project: {PROJECT}")
    if SKIP:
        print(f"Skip:    {', '.join(SKIP)}")

    # Load R2 credentials once
    r2_cfg: dict = {}
    try:
        r2_cfg = {
            "accessId": read_secret("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_RW"),
            "accessSecret": read_secret("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_RW"),
            "bucketName": read_secret("VISGATE_DEPLOY_API_INFERENCE_R2_BUCKET_NAME"),
            "endpointUrl": read_secret("VISGATE_DEPLOY_API_S3_API_R2"),
            "keyPrefix": "smoke-tests/e2e",
        }
        info("R2 credentials loaded")
    except Exception as exc:
        info(f"R2 credentials unavailable ({exc}) — artifacts won't be stored")

    results: dict[str, bool] = {}

    if "image" not in SKIP:
        results["image"] = smoke_image(r2_cfg)
    else:
        info("Skipping image")

    if "audio" not in SKIP:
        results["audio"] = smoke_audio(r2_cfg)
    else:
        info("Skipping audio")

    if "video" not in SKIP:
        results["video"] = smoke_video(r2_cfg)
    else:
        info("Skipping video")

    # ── summary ──────────────────────────────────────────────────────────────
    section("Summary")
    all_passed = True
    for mod, passed in results.items():
        if passed:
            ok(mod)
        else:
            fail(mod)
            all_passed = False

    if all_passed:
        print(f"\n{G}{B}ALL PASSED{X}\n")
        return 0
    else:
        print(f"\n{R}{B}SOME FAILED{X}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
