#!/usr/bin/env python3
"""End-to-end test for Vast.ai Serverless provider via Cloud Run API.

Usage:
    python scripts/vast_e2e.py [--modality image|audio|video] [--all]

Defaults to 'image' (sd-turbo, text_to_image) which is fastest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

API_BASE = os.environ.get(
    "API_BASE",
    "https://visgate-deploy-api-93820292919.europe-west3.run.app/deployapi",
).rstrip("/")

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODALITY_CONFIGS: dict[str, dict[str, Any]] = {
    "image": {
        "deployment": {
            "hf_model_id": "stabilityai/sd-turbo",
            "task": "text_to_image",
            "provider": "vast",
        },
        "job": {
            "task": "text_to_image",
            "input": {
                "prompt": "a cute orange cat sitting on a windowsill, digital art",
                "num_inference_steps": 2,
            },
        },
    },
    "audio": {
        "deployment": {
            "hf_model_id": "openai/whisper-large-v3-turbo",
            "task": "speech_to_text",
            "provider": "vast",
        },
        "job": {
            "task": "speech_to_text",
            "input": {
                "audio_url": "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/0_george_0.wav",
            },
        },
    },
    "video": {
        "deployment": {
            "hf_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
            "task": "text_to_video",
            "provider": "vast",
        },
        "job": {
            "task": "text_to_video",
            "input": {
                "prompt": "a red cube slowly rotating on a white table, studio lighting",
                "num_inference_steps": 4,
                "num_frames": 8,
                "fps": 8,
            },
        },
    },
}

DEPLOYMENT_TIMEOUT = 600  # 10 min
JOB_TIMEOUT = 600  # 10 min


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def _request(method: str, url: str, payload: dict[str, Any] | None = None, *, timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, method=method, data=data, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {method} {url}: {body[:500]}") from exc


def _poll(url: str, terminal: set[str], *, timeout: int, label: str = "") -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] | None = None
    prev_status = ""
    while time.time() < deadline:
        try:
            last = _request("GET", url)
        except Exception as exc:
            elapsed = int(time.time() - (deadline - timeout))
            print(f"  [{elapsed:>4}s] {label} poll request error (retrying): {exc}")
            time.sleep(10)
            continue
        st = last.get("status", "")
        if st != prev_status:
            elapsed = int(time.time() - (deadline - timeout))
            print(f"  [{elapsed:>4}s] {label} status → {st}")
            prev_status = st
        if st in terminal:
            return last
        time.sleep(5)
    raise TimeoutError(f"Timeout after {timeout}s for {label}; last={json.dumps(last, indent=2)[:500]}")


def _delete_deployment(dep_id: str) -> None:
    try:
        req = urllib.request.Request(f"{API_BASE}/v1/deployments/{dep_id}", method="DELETE", headers=_headers())
        with urllib.request.urlopen(req, timeout=30):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code not in (204, 404):
            print(f"  WARN: delete {dep_id} HTTP {exc.code}")
    except Exception as exc:
        print(f"  WARN: delete {dep_id} failed: {exc}")


def run_modality(modality: str) -> dict[str, Any]:
    config = MODALITY_CONFIGS[modality]
    result: dict[str, Any] = {"modality": modality, "success": False}
    deployment_id = None

    try:
        # 1. Create deployment
        print(f"\n{'='*60}")
        print(f"[{modality.upper()}] Creating Vast.ai deployment...")
        dep_payload = {
            **config["deployment"],
            "hf_token": HF_TOKEN,
            "user_runpod_key": RUNPOD_API_KEY,
        }
        dep_resp = _request("POST", f"{API_BASE}/v1/deployments", dep_payload)
        deployment_id = dep_resp.get("deployment_id") or dep_resp.get("id")
        result["deployment_id"] = deployment_id
        result["deployment_response"] = dep_resp
        print(f"  deployment_id = {deployment_id}")
        print(f"  initial status = {dep_resp.get('status')}")

        # 2. Poll until ready
        print(f"  Polling deployment (timeout {DEPLOYMENT_TIMEOUT}s)...")
        dep_final = _poll(
            f"{API_BASE}/v1/deployments/{deployment_id}",
            {"ready", "failed", "webhook_failed", "deleted"},
            timeout=DEPLOYMENT_TIMEOUT,
            label=f"{modality}/deployment",
        )
        result["deployment_final"] = dep_final

        if dep_final["status"] != "ready":
            result["error"] = f"Deployment ended with status={dep_final['status']}: {dep_final.get('error')}"
            print(f"  FAIL: {result['error']}")
            return result

        endpoint_url = dep_final.get("endpoint_url")
        print(f"  endpoint_url = {endpoint_url}")
        result["endpoint_url"] = endpoint_url

        # Print logs
        logs = dep_final.get("logs", [])
        if logs:
            print(f"  Last 3 deployment logs:")
            for log in logs[-3:]:
                print(f"    {log.get('level', 'INFO')} {log.get('message', '')[:120]}")

        # 3. Submit inference job
        print(f"\n  Submitting inference job...")
        job_payload = {
            **config["job"],
            "deployment_id": deployment_id,
        }
        job_resp = _request("POST", f"{API_BASE}/v1/inference/jobs", job_payload)
        job_id = job_resp.get("job_id") or job_resp.get("id")
        result["job_id"] = job_id
        result["job_response"] = job_resp
        print(f"  job_id = {job_id}")

        # 4. Poll job until done
        print(f"  Polling job (timeout {JOB_TIMEOUT}s)...")
        job_final = _poll(
            f"{API_BASE}/v1/inference/jobs/{job_id}",
            {"completed", "failed", "cancelled", "expired"},
            timeout=JOB_TIMEOUT,
            label=f"{modality}/job",
        )
        result["job_final"] = job_final

        if job_final.get("status") != "completed":
            result["error"] = f"Job ended with status={job_final.get('status')}: {job_final.get('error')}"
            print(f"  FAIL: {result['error']}")
            return result

        # 5. Check output
        artifact = job_final.get("artifact")
        result["artifact"] = artifact
        output = job_final.get("output") or job_final.get("output_preview")
        result["output_preview"] = output
        print(f"  Job completed!")
        if artifact:
            print(f"  artifact.key = {artifact.get('key')}")
            print(f"  artifact.url = {artifact.get('url')}")
            print(f"  artifact.content_type = {artifact.get('content_type')}")
            print(f"  artifact.bytes = {artifact.get('bytes')}")
        if output:
            out_str = json.dumps(output) if isinstance(output, dict) else str(output)
            print(f"  output_preview = {out_str[:200]}")

        result["success"] = True
        print(f"\n  ✓ [{modality.upper()}] E2E SUCCESS")
        return result

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  FAIL: {result['error']}")
        return result
    finally:
        if deployment_id:
            print(f"  Cleaning up deployment {deployment_id}...")
            _delete_deployment(deployment_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Vast.ai E2E smoke test")
    parser.add_argument("--modality", "-m", default="image", choices=list(MODALITY_CONFIGS))
    parser.add_argument("--all", "-a", action="store_true", help="Run all modalities")
    parser.add_argument("--keep", action="store_true", help="Don't delete deployments after test")
    parser.add_argument("--output", "-o", default="vast-e2e-results.json")
    args = parser.parse_args()

    modalities = list(MODALITY_CONFIGS) if args.all else [args.modality]

    print(f"Vast.ai E2E Test — API: {API_BASE}")
    print(f"Modalities: {', '.join(modalities)}")
    print(f"Provider: vast (serverless)")

    results: list[dict[str, Any]] = []
    for mod in modalities:
        results.append(run_modality(mod))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = 0
    for r in results:
        icon = "✓" if r.get("success") else "✗"
        print(f"  {icon} {r['modality']}: {'PASS' if r.get('success') else r.get('error', 'FAIL')[:80]}")
        if r.get("success"):
            passed += 1

    print(f"\n  {passed}/{len(results)} passed")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {args.output}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
