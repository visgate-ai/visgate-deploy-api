#!/usr/bin/env python3
"""
E2E test: Istanbul image via API with timing metrics.
Measures:
  T1: Deploy request -> Pod ready (seconds)
  T2: Inference request -> Response received (seconds)
  Total: T1 + T2
"""
import os
import sys
import time
import requests
import base64
from typing import Any

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
# SD-Turbo: fastest model, 2 steps, good quality
MODEL = "stabilityai/sd-turbo"
GPU_TIER = "A10"  # AMPERE_24 - optimal cost/speed for SD
ISTANBUL_PROMPT = (
    "A beautiful panoramic view of Istanbul with the Bosphorus strait, "
    "Hagia Sophia and minarets, golden hour, photorealistic, 4k"
)
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "https://httpbin.org/post")
MAX_DEPLOY_WAIT_SECONDS = int(os.environ.get("MAX_DEPLOY_WAIT_SECONDS", "1800"))


def _api_headers(runpod_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {runpod_key}",
        "X-Runpod-Api-Key": runpod_key,
    }


def main():
    print("=" * 60)
    print("E2E: Istanbul Image - Timed Deployment + Inference")
    print("=" * 60)

    runpod_key = os.environ.get("RUNPOD")
    hf_token = os.environ.get("HF")
    if not runpod_key:
        print("Error: RUNPOD env var required")
        sys.exit(1)

    payload = {
        "hf_model_id": MODEL,
        "gpu_tier": GPU_TIER,
        "hf_token": hf_token,
        "user_webhook_url": WEBHOOK_URL,
    }

    headers = _api_headers(runpod_key)

    # --- Phase 1: Deploy ---
    print("\n[1] Triggering deployment...")
    t_deploy_start = time.perf_counter()
    r = requests.post(
        f"{API_BASE}/v1/deployments",
        json=payload,
        headers=headers,
        timeout=30,
    )
    if r.status_code >= 400:
        print(f"Deploy failed: {r.text}")
        sys.exit(1)
    data = r.json()
    deploy_id = data.get("deployment_id")
    if not deploy_id:
        print(f"No deployment_id: {data}")
        sys.exit(1)
    print(f"    Deployment ID: {deploy_id}")

    # --- Phase 2: Wait for Ready ---
    print("\n[2] Waiting for endpoint ready (poll every 10s)...")
    status_url = f"{API_BASE}/v1/deployments/{deploy_id}"
    endpoint_url = None
    deploy_deadline = time.perf_counter() + MAX_DEPLOY_WAIT_SECONDS
    while True:
        if time.perf_counter() > deploy_deadline:
            print(f"    Timeout waiting for ready after {MAX_DEPLOY_WAIT_SECONDS}s")
            sys.exit(1)
        r = requests.get(status_url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"    Poll error: {r.text}")
            time.sleep(10)
            continue
        d = r.json()
        st = d.get("status", "")
        elapsed = time.perf_counter() - t_deploy_start
        print(f"    [{int(elapsed)}s] status={st}")
        if st == "ready":
            endpoint_url = d.get("endpoint_url")
            break
        if st == "failed":
            print(f"    Failed: {d.get('error', 'unknown')}")
            sys.exit(1)
        time.sleep(10)

    t_deploy_end = time.perf_counter()
    t_deploy_sec = round(t_deploy_end - t_deploy_start, 2)
    print(f"\n    Pod READY in {t_deploy_sec}s")
    print(f"    Endpoint: {endpoint_url}")

    # --- Phase 3: Inference ---
    print("\n[3] Sending Istanbul inference request...")
    endpoint_root = endpoint_url[:-4] if endpoint_url.endswith("/run") else endpoint_url
    run_url = f"{endpoint_root}/runsync"
    inference_payload = {
        "input": {
            "prompt": ISTANBUL_PROMPT,
            "num_inference_steps": 2,
        }
    }
    run_headers = {"Authorization": f"Bearer {runpod_key}", "Content-Type": "application/json"}

    t_inference_start = time.perf_counter()
    inference_done = False
    output = None

    while not inference_done:
        if "runsync" in run_url:
            r = requests.post(run_url, json=inference_payload, headers=run_headers, timeout=300)
        else:
            r = requests.get(run_url, headers=run_headers, timeout=60)
        r.raise_for_status()
        inf = r.json()
        job_status = inf.get("status")
        job_id = inf.get("id")

        if job_status == "COMPLETED":
            inference_done = True
            output = inf.get("output", {})
        elif job_status in ("IN_QUEUE", "IN_PROGRESS", "RUNNING"):
            elapsed = time.perf_counter() - t_inference_start
            print(f"    [{int(elapsed)}s] {job_status} job={job_id}")
            time.sleep(5)
            run_url = f"{endpoint_root}/status/{job_id}"
            continue
        elif job_status == "FAILED":
            err = str(inf.get("error", ""))
            if "still loading" in err.lower():
                print("    Model loading, waiting 10s...")
                time.sleep(10)
                continue
            print(f"Inference failed: {err}")
            sys.exit(1)
        else:
            inference_done = True
            output = inf.get("output", {})

    t_inference_end = time.perf_counter()
    t_inference_sec = round(t_inference_end - t_inference_start, 2)

    # --- Phase 4: Save Image ---
    image_b64 = output.get("image_base64") or output.get("image")
    if not image_b64:
        print(f"No image in output: {list(output.keys())}")
        sys.exit(1)
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]
    out_path = os.path.join(os.getcwd(), "istanbul_final.png")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    print(f"\n    Image saved: {out_path}")

    # --- Summary ---
    total_sec = round(t_deploy_sec + t_inference_sec, 2)
    print("\n" + "=" * 60)
    print("TIMING RESULTS")
    print("=" * 60)
    print(f"  T1 (deploy request -> pod ready):  {t_deploy_sec} s")
    print(f"  T2 (inference request -> response): {t_inference_sec} s")
    print(f"  Total:                              {total_sec} s")
    print("=" * 60)
    print("SUCCESS")

    timing_summary: dict[str, Any] = {
        "deployment_id": deploy_id,
        "model": MODEL,
        "gpu_tier": GPU_TIER,
        "t_deploy_seconds": t_deploy_sec,
        "t_inference_seconds": t_inference_sec,
        "t_total_seconds": total_sec,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_json = os.path.join(os.getcwd(), "e2e_timing_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        import json

        json.dump(timing_summary, f, ensure_ascii=False, indent=2)
    print(f"Timing summary saved: {out_json}")


if __name__ == "__main__":
    main()
