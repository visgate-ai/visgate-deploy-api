#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import time
import urllib.request

from google.cloud import firestore


API_BASE = os.environ.get("API_BASE", "https://visgate-deploy-api-wxup7pxrsa-ey.a.run.app")
RUNPOD_KEY = os.environ.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD") or ""
FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "visgate_deploy_api_deployments")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "visgate")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "5"))
POLL_ATTEMPTS = int(os.environ.get("POLL_ATTEMPTS", "30"))

MODELS = [
    ("image", "stabilityai/sdxl-turbo"),
    ("audio", "openai/whisper-large-v3"),
    ("video", "Wan-AI/Wan2.1-T2V-1.3B"),
]


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_KEY}",
        "X-Runpod-Api-Key": RUNPOD_KEY,
    }


def _request(url: str, *, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=_headers(), method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload) if payload else {}


def main() -> int:
    if not RUNPOD_KEY:
        raise SystemExit("Set RUNPOD_API_KEY or RUNPOD")

    client = firestore.Client(project=PROJECT_ID)
    results: list[dict] = []

    for expected_profile, model_id in MODELS:
        created = _request(
            f"{API_BASE.rstrip('/')}/v1/deployments",
            method="POST",
            body={"hf_model_id": model_id, "user_webhook_url": "https://httpbin.org/post"},
        )
        deployment_id = created["deployment_id"]
        result = {
            "expected_profile": expected_profile,
            "model_id": model_id,
            "deployment_id": deployment_id,
        }
        try:
            for _ in range(POLL_ATTEMPTS):
                data = client.collection(FIRESTORE_COLLECTION).document(deployment_id).get().to_dict() or {}
                result.update(
                    {
                        "status": data.get("status"),
                        "error": data.get("error"),
                        "endpoint_url": data.get("endpoint_url"),
                        "runpod_endpoint_id": data.get("runpod_endpoint_id"),
                        "logs": (data.get("logs") or [])[-8:],
                    }
                )
                if data.get("runpod_endpoint_id") or data.get("status") in {"ready", "failed", "loading_model"}:
                    break
                time.sleep(POLL_SECONDS)
        finally:
            try:
                _request(f"{API_BASE.rstrip('/')}/v1/deployments/{deployment_id}", method="DELETE")
            except Exception:
                pass
        results.append(result)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())