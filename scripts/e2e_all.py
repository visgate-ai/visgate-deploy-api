#!/usr/bin/env python3
"""Unified E2E test runner for ALL modalities × ALL providers.

Dynamic, modular design:
  - Models/tasks are configured via a lightweight registry (env-overridable)
  - GPU selection is fully delegated to the backend (dynamic)
  - Providers are iterable (vast, runpod, or both)
  - Modalities are iterable (image, audio, video, or all)
  - Runs sequentially to conserve credit and ensure clean state
  - Cleanup is always performed, even on failure
  - Results are persisted to JSON for CI/CD integration

Usage:
    python scripts/e2e_all.py                          # all modalities × all providers
    python scripts/e2e_all.py -m image -p runpod       # image on runpod only
    python scripts/e2e_all.py -m audio -m video -p vast # audio+video on vast
    python scripts/e2e_all.py --retry 2                 # retry failed tests up to 2 times

Environment:
    RUNPOD_API_KEY   RunPod API key
    HF_TOKEN         HuggingFace token
    VAST_API_KEY     Vast.ai API key (needed for vast provider header pass-through)
    API_BASE         Override deploy API base URL
    E2E_MODEL_IMAGE  Override image model (default: stabilityai/sd-turbo)
    E2E_MODEL_AUDIO  Override audio model (default: openai/whisper-large-v3-turbo)
    E2E_MODEL_VIDEO  Override video model (default: Wan-AI/Wan2.1-T2V-1.3B)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

# ── Configuration (env-overridable) ──────────────────────────────────────────

API_BASE = os.environ.get(
    "API_BASE",
    "https://visgate-deploy-api-93820292919.europe-west3.run.app/deployapi",
).rstrip("/")

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
VAST_API_KEY = os.environ.get("VAST_API_KEY", "")

DEPLOYMENT_TIMEOUT = int(os.environ.get("E2E_DEPLOY_TIMEOUT", "900"))  # 15 min
JOB_TIMEOUT = int(os.environ.get("E2E_JOB_TIMEOUT", "600"))  # 10 min

PROVIDERS = ["runpod", "vast"]
MODALITIES = ["image", "audio", "video"]


def _build_modality_registry() -> dict[str, dict[str, Any]]:
    """Build modality config dynamically. Models are env-overridable."""
    return {
        "image": {
            "model": os.environ.get("E2E_MODEL_IMAGE", "stabilityai/sd-turbo"),
            "task": "text_to_image",
            "job_input": {
                "prompt": "a cute orange cat sitting on a windowsill, digital art",
                "num_inference_steps": 2,
            },
        },
        "audio": {
            "model": os.environ.get("E2E_MODEL_AUDIO", "openai/whisper-large-v3-turbo"),
            "task": "speech_to_text",
            "job_input": {
                "audio_url": "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/0_george_0.wav",
            },
        },
        "video": {
            "model": os.environ.get("E2E_MODEL_VIDEO", "Wan-AI/Wan2.1-T2V-1.3B"),
            "task": "text_to_video",
            "job_input": {
                "prompt": "a red cube slowly rotating on a white table, studio lighting",
                "num_inference_steps": 4,
                "num_frames": 8,
                "fps": 8,
            },
        },
    }


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _headers() -> dict[str, str]:
    h: dict[str, str] = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    # Pass through provider-specific keys the API needs
    if RUNPOD_API_KEY:
        h["x-runpod-api-key"] = RUNPOD_API_KEY
    if HF_TOKEN:
        h["x-hf-token"] = HF_TOKEN
    if VAST_API_KEY:
        h["x-vast-api-key"] = VAST_API_KEY
    return h


def _request(
    method: str, url: str, payload: dict[str, Any] | None = None, *, timeout: int = 120
) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, method=method, data=data, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {method} {url}: {body[:500]}") from exc


def _poll(
    url: str,
    terminal: set[str],
    *,
    timeout: int,
    label: str = "",
) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    prev_status = ""
    t0 = time.time()
    while time.time() < deadline:
        try:
            last = _request("GET", url)
        except Exception as e:
            elapsed = int(time.time() - t0)
            print(f"  [{elapsed:>4}s] {label} poll error: {e}")
            time.sleep(5)
            continue
        st = last.get("status", "")
        if st != prev_status:
            elapsed = int(time.time() - t0)
            print(f"  [{elapsed:>4}s] {label} status -> {st}")
            prev_status = st
        if st in terminal:
            return last
        time.sleep(5)
    raise TimeoutError(
        f"Timeout after {timeout}s for {label}; last={json.dumps(last, indent=2)[:500]}"
    )


def _delete_deployment(dep_id: str) -> None:
    try:
        req = urllib.request.Request(
            f"{API_BASE}/v1/deployments/{dep_id}",
            method="DELETE",
            headers=_headers(),
        )
        with urllib.request.urlopen(req, timeout=30):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code not in (204, 404):
            print(f"  WARN: delete {dep_id} HTTP {exc.code}")
    except Exception as exc:
        print(f"  WARN: delete {dep_id} failed: {exc}")


# ── Single test case ─────────────────────────────────────────────────────────

def run_test(provider: str, modality: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run a single E2E test: deploy -> wait -> infer -> wait -> cleanup."""
    tag = f"{provider}/{modality}"
    result: dict[str, Any] = {
        "provider": provider,
        "modality": modality,
        "model": config["model"],
        "task": config["task"],
        "success": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    deployment_id = None
    t_start = time.time()

    try:
        # ── 1. Create deployment ─────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"[{tag}] Creating deployment: {config['model']} ({config['task']})")
        print(f"[{tag}] Provider: {provider} | API: {API_BASE}")
        dep_payload = {
            "hf_model_id": config["model"],
            "task": config["task"],
            "provider": provider,
            "hf_token": HF_TOKEN,
            "user_runpod_key": RUNPOD_API_KEY,
        }
        dep_resp = _request("POST", f"{API_BASE}/v1/deployments", dep_payload)
        deployment_id = dep_resp.get("deployment_id") or dep_resp.get("id")
        result["deployment_id"] = deployment_id
        print(f"  deployment_id = {deployment_id}")
        print(f"  initial status = {dep_resp.get('status')}")

        # ── 2. Poll deployment until ready ───────────────────────────────
        print(f"  Polling deployment (timeout {DEPLOYMENT_TIMEOUT}s)...")
        dep_final = _poll(
            f"{API_BASE}/v1/deployments/{deployment_id}",
            {"ready", "failed", "webhook_failed", "deleted"},
            timeout=DEPLOYMENT_TIMEOUT,
            label=tag,
        )
        result["deployment_status"] = dep_final.get("status")
        t_deploy = time.time() - t_start
        result["t_deploy_s"] = round(t_deploy, 1)

        if dep_final["status"] != "ready":
            err = dep_final.get("error", "unknown")
            result["error"] = f"Deployment {dep_final['status']}: {err}"
            # Print logs for debugging
            for log in (dep_final.get("logs") or [])[-5:]:
                print(f"    {log.get('level','?')} {log.get('message','')[:120]}")
            print(f"  FAIL: {result['error']}")
            return result

        endpoint_url = dep_final.get("endpoint_url")
        gpu = dep_final.get("gpu_allocated", "?")
        print(f"  READY in {t_deploy:.0f}s | GPU: {gpu} | endpoint: {endpoint_url}")

        # ── 3. Submit inference job ──────────────────────────────────────
        print(f"  Submitting inference job...")
        job_payload = {
            "deployment_id": deployment_id,
            "task": config["task"],
            "input": config["job_input"],
        }
        job_resp = _request("POST", f"{API_BASE}/v1/inference/jobs", job_payload)
        job_id = job_resp.get("job_id") or job_resp.get("id")
        result["job_id"] = job_id
        print(f"  job_id = {job_id}")

        # ── 4. Poll job until done ───────────────────────────────────────
        print(f"  Polling job (timeout {JOB_TIMEOUT}s)...")
        t_job_start = time.time()
        job_final = _poll(
            f"{API_BASE}/v1/inference/jobs/{job_id}",
            {"completed", "failed", "cancelled", "expired"},
            timeout=JOB_TIMEOUT,
            label=f"{tag}/job",
        )
        t_job = time.time() - t_job_start
        result["job_status"] = job_final.get("status")
        result["t_job_s"] = round(t_job, 1)

        if job_final.get("status") != "completed":
            result["error"] = f"Job {job_final.get('status')}: {job_final.get('error', 'unknown')}"
            print(f"  FAIL: {result['error']}")
            return result

        # ── 5. Validate output ───────────────────────────────────────────
        artifact = job_final.get("artifact")
        output = job_final.get("output") or job_final.get("output_preview")
        if artifact:
            result["artifact_key"] = artifact.get("key")
            result["artifact_bytes"] = artifact.get("bytes")
            result["artifact_content_type"] = artifact.get("content_type")
            print(f"  artifact: {artifact.get('content_type')} ({artifact.get('bytes')} bytes)")
        if output:
            out_str = json.dumps(output) if isinstance(output, dict) else str(output)
            result["output_preview"] = out_str[:300]
            print(f"  output: {out_str[:200]}")

        t_total = time.time() - t_start
        result["t_total_s"] = round(t_total, 1)
        result["success"] = True
        print(f"\n  OK [{tag}] total={t_total:.0f}s (deploy={t_deploy:.0f}s, job={t_job:.0f}s)")
        return result

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  FAIL: {result['error']}")
        return result
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        if deployment_id:
            print(f"  Cleaning up {deployment_id}...")
            _delete_deployment(deployment_id)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Unified E2E test: all modalities × all providers")
    parser.add_argument(
        "-m", "--modality",
        action="append",
        choices=MODALITIES,
        help="Modality to test (repeatable); omit for all",
    )
    parser.add_argument(
        "-p", "--provider",
        action="append",
        choices=PROVIDERS,
        help="Provider to test (repeatable); omit for all",
    )
    parser.add_argument("--retry", type=int, default=1, help="Max retry attempts per test (default: 1)")
    parser.add_argument("-o", "--output", default="e2e-results.json", help="Output JSON file")
    args = parser.parse_args()

    modalities = args.modality or MODALITIES
    providers = args.provider or PROVIDERS
    max_retries = max(1, args.retry)
    registry = _build_modality_registry()

    # Build test matrix
    test_matrix = [(p, m) for p in providers for m in modalities]

    print(f"{'='*70}")
    print(f"Visgate E2E Test Runner")
    print(f"{'='*70}")
    print(f"  API:        {API_BASE}")
    print(f"  Providers:  {', '.join(providers)}")
    print(f"  Modalities: {', '.join(modalities)}")
    print(f"  Tests:      {len(test_matrix)}")
    print(f"  Retry:      {max_retries}x")
    print(f"  Deploy timeout: {DEPLOYMENT_TIMEOUT}s | Job timeout: {JOB_TIMEOUT}s")
    print(f"{'='*70}")

    all_results: list[dict[str, Any]] = []

    for idx, (provider, modality) in enumerate(test_matrix, 1):
        config = registry[modality]
        tag = f"{provider}/{modality}"
        print(f"\n[{idx}/{len(test_matrix)}] {tag}")

        result = None
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                print(f"\n  Retry {attempt}/{max_retries} for {tag}...")
                time.sleep(5)  # Brief pause between retries
            result = run_test(provider, modality, config)
            if result.get("success"):
                break
            # Don't retry on certain permanent failures
            err = (result.get("error") or "").lower()
            if any(k in err for k in ("insufficient credit", "model not found", "unauthorized")):
                print(f"  Permanent failure — skipping retries for {tag}")
                break

        result["attempt"] = attempt
        all_results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("E2E TEST SUMMARY")
    print(f"{'='*70}")
    passed = 0
    for r in all_results:
        tag = f"{r['provider']}/{r['modality']}"
        if r.get("success"):
            t = r.get("t_total_s", "?")
            print(f"  PASS  {tag:<20} {t}s  (deploy={r.get('t_deploy_s','?')}s, job={r.get('t_job_s','?')}s)")
            passed += 1
        else:
            err = (r.get("error") or "FAIL")[:60]
            print(f"  FAIL  {tag:<20} {err}")

    print(f"\n  {passed}/{len(all_results)} passed")
    print(f"{'='*70}")

    # ── Save results ─────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results: {args.output}")

    return 0 if passed == len(all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
