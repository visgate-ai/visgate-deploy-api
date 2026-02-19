#!/usr/bin/env python3
"""
Sends a request to our GCP API; we do not touch HF, Runpod, or Docker Hub directly from here.
Input: keys (Runpod, optional HF) + HF model ID. The response is received via webhook.

Usage:
  python3 scripts/deploy_via_api.py
  # Reads RUNPOD and optional HF from .env.local.
  # Provide HF model: hf_model_id or model_name (e.g. FLUX.1-schnell).
"""
import json
import os
import sys
import time
import urllib.request

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ENV_LOCAL = os.path.join(REPO_ROOT, ".env.local")

# Our GCP API (the only entry point)
API_BASE = os.environ.get("VISGATE_API_URL", "https://visgate-deploy-api-93820292919.europe-west1.run.app")
# Default model (HF model ID or model_name) - sdxl-turbo: smaller, faster
DEFAULT_HF_MODEL = os.environ.get("HF_MODEL", "stabilityai/sdxl-turbo")


def load_env():
    env = {}
    if os.path.isfile(ENV_LOCAL):
        with open(ENV_LOCAL) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    return env


def create_webhook_url():
    """Get a single-use URL from Webhook.site."""
    req = urllib.request.Request(
        "https://webhook.site/token",
        data=b"{}",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    uuid = data.get("uuid")
    if not uuid:
        raise RuntimeError("Could not get Webhook.site token: " + str(data))
    return f"https://webhook.site/{uuid}", uuid


def api_post(path, body, runpod_key: str):
    """POST to our GCP API."""
    url = f"{API_BASE.rstrip('/')}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {runpod_key}",
            "X-Runpod-Api-Key": runpod_key,
            "User-Agent": "Visgate-Deploy-API/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def api_get(path, runpod_key: str):
    """GET from our GCP API."""
    url = f"{API_BASE.rstrip('/')}{path}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {runpod_key}",
            "X-Runpod-Api-Key": runpod_key,
            "User-Agent": "Visgate-Deploy-API/1.0",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def get_webhook_requests(token_uuid):
    """Get requests received at Webhook.site (newest first)."""
    url = f"https://webhook.site/token/{token_uuid}/requests?sorting=newest&per_page=5"
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode())
    return data.get("data") or []


def main():
    env = load_env()
    runpod_key = env.get("RUNPOD", "").strip()
    hf_token = env.get("HF", "").strip() or None

    if not runpod_key:
        print("ERROR: Set RUNPOD=... in .env.local.", file=sys.stderr)
        sys.exit(1)

    # Webhook URL (orchestrator will POST the response here)
    webhook_url, token_uuid = create_webhook_url()
    print("Webhook URL (response will arrive here):", webhook_url)

    # Model: use hf_model_id (Direct HF model ID)
    hf_model = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
    body = {
        "hf_model_id": hf_model,
        "user_webhook_url": webhook_url,
    }
    gpu_tier = os.environ.get("GPU_TIER")
    if gpu_tier:
        body["gpu_tier"] = gpu_tier
    if hf_token:
        body["hf_token"] = hf_token

    print("Sending to GCP API (Inference Orchestrator)...")
    try:
        resp = api_post("/v1/deployments", body, runpod_key)
    except urllib.error.HTTPError as e:
        print("API Error:", e.code, e.read().decode()[:500], file=sys.stderr)
        sys.exit(1)

    deployment_id = resp.get("deployment_id")
    if not deployment_id:
        print("Deployment ID missing in response:", resp, file=sys.stderr)
        sys.exit(1)

    print("Deployment created:", deployment_id)
    print("Model:", resp.get("model_id"), "| Webhook:", resp.get("webhook_url"))
    print("Waiting for response via webhook (or status poll)...")

    # 1) Wait for webhook POST from orchestrator when ready
    # 2) Poll status from API until deadline
    deadline = time.monotonic() + 600  # 10 min
    last_status = None
    webhook_received = None

    while time.monotonic() < deadline:
        # Check for new requests at Webhook.site
        for req in get_webhook_requests(token_uuid):
            try:
                content = req.get("content") or ""
                payload = json.loads(content)
                if payload.get("deployment_id") == deployment_id and payload.get("status") == "ready":
                    webhook_received = payload
                    break
            except (json.JSONDecodeError, TypeError):
                pass
        if webhook_received:
            print("\n--- Response received via Webhook ---")
            print(json.dumps(webhook_received, indent=2, ensure_ascii=False))
            print("Endpoint URL (for inference):", webhook_received.get("endpoint_url"))
            return

        # Fallback: poll status from API
        try:
            doc = api_get(f"/v1/deployments/{deployment_id}", runpod_key)
            status = doc.get("status")
            if status != last_status:
                print("  status:", status)
                last_status = status
            if status == "ready":
                print("\n--- Response received via API (ready) ---")
                print("endpoint_url:", doc.get("endpoint_url"))
                return
            if status == "failed":
                print("Deployment failed:", doc.get("error"), file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print("  poll error:", e)
        time.sleep(5)

    print("Timeout (10 min). Webhook not received. Is INTERNAL_WEBHOOK_BASE_URL set on Cloud Run?", file=sys.stderr)
    print("The endpoint might still be active: GET", f"{API_BASE}/v1/deployments/{deployment_id}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
