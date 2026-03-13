"""Local end-to-end smoke test for the Docker Compose local stack.

Requires:
- API_BASE (default http://localhost:8080)
- RUNPOD_API_KEY (any non-empty value when DEV_MODE=true)
- HF_TOKEN (required by the API)
"""

from __future__ import annotations

import os
import sys
import time

import requests


API_BASE = os.environ.get("API_BASE", "http://localhost:8080").rstrip("/")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = os.environ.get("HF_MODEL_ID", "stabilityai/sd-turbo")


def main() -> int:
    if not RUNPOD_API_KEY or not HF_TOKEN:
        print("Set RUNPOD_API_KEY and HF_TOKEN in the environment.", file=sys.stderr)
        return 2

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    health = requests.get(f"{API_BASE}/health", timeout=20)
    health.raise_for_status()
    print("health:", health.json())

    create = requests.post(
        f"{API_BASE}/v1/deployments",
        headers=headers,
        json={
            "hf_model_id": MODEL_ID,
            "hf_token": HF_TOKEN,
            "task": "text_to_image",
        },
        timeout=60,
    )
    create.raise_for_status()
    deployment = create.json()
    deployment_id = deployment["deployment_id"]
    print("deployment:", deployment)

    status_payload = None
    for _ in range(180):
        time.sleep(5)
        resp = requests.get(f"{API_BASE}/v1/deployments/{deployment_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        status_payload = resp.json()
        print("status:", status_payload["status"])
        if status_payload["status"] in {"ready", "failed", "webhook_failed"}:
            break

    if not status_payload or status_payload["status"] != "ready":
        print("deployment did not become ready", file=sys.stderr)
        return 1

    job = requests.post(
        f"{API_BASE}/v1/inference/jobs",
        headers=headers,
        json={
            "deployment_id": deployment_id,
            "task": "text_to_image",
            "input": {"prompt": "A red cube on a wooden table", "num_inference_steps": 2},
        },
        timeout=60,
    )
    job.raise_for_status()
    job_payload = job.json()
    print("job:", job_payload)

    job_id = job_payload["job_id"]
    final_payload = None
    for _ in range(120):
        time.sleep(3)
        resp = requests.get(f"{API_BASE}/v1/inference/jobs/{job_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        final_payload = resp.json()
        print("job status:", final_payload["status"])
        if final_payload["status"] in {"completed", "failed", "cancelled", "expired"}:
            break

    print("final job:", final_payload)
    return 0 if final_payload and final_payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())