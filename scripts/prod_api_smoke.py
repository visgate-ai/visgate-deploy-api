#!/usr/bin/env python3
"""Production smoke + timing test for deploy API.

Usage:
  API_BASE=https://... RUNPOD=... HF=... python3 scripts/prod_api_smoke.py

If RUNPOD key is missing, only public endpoints are tested.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict

import requests

API_BASE = os.environ.get("API_BASE", "https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app").rstrip("/")
RUNPOD_KEY = os.environ.get("RUNPOD")
HF_TOKEN = os.environ.get("HF")
MODEL = os.environ.get("HF_MODEL", "stabilityai/sd-turbo")
GPU_TIER = os.environ.get("GPU_TIER", "A10")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "https://httpbin.org/post")
DEPLOY_TIMEOUT_SECONDS = int(os.environ.get("DEPLOY_TIMEOUT_SECONDS", "1800"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "10"))
USER_S3_URL = os.environ.get("USER_S3_URL")
USER_AWS_ACCESS_KEY_ID = os.environ.get("USER_AWS_ACCESS_KEY_ID")
USER_AWS_SECRET_ACCESS_KEY = os.environ.get("USER_AWS_SECRET_ACCESS_KEY")
USER_AWS_ENDPOINT_URL = os.environ.get("USER_AWS_ENDPOINT_URL")


@dataclass
class Result:
    health_ok: bool = False
    metrics_ok: bool = False
    deployment_attempted: bool = False
    deployment_id: str | None = None
    create_status_code: int | None = None
    create_latency_ms: float | None = None
    ready_latency_sec: float | None = None
    final_status: str | None = None
    delete_status_code: int | None = None
    errors: list[str] | None = None


def _headers() -> dict[str, str]:
    if not RUNPOD_KEY:
        return {}
    return {"Authorization": f"Bearer {RUNPOD_KEY}", "X-Runpod-Api-Key": RUNPOD_KEY}


def _fail(result: Result, message: str) -> None:
    if result.errors is None:
        result.errors = []
    result.errors.append(message)


def main() -> int:
    result = Result(errors=[])

    # 1) health
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=20)
        result.health_ok = resp.status_code == 200 and resp.json().get("status") == "ok"
        if not result.health_ok:
            _fail(result, f"health failed: {resp.status_code} {resp.text[:300]}")
    except Exception as exc:
        _fail(result, f"health exception: {exc}")

    # 2) metrics
    try:
        resp = requests.get(f"{API_BASE}/metrics", timeout=20)
        result.metrics_ok = resp.status_code == 200
        if not result.metrics_ok:
            _fail(result, f"metrics failed: {resp.status_code} {resp.text[:300]}")
    except Exception as exc:
        _fail(result, f"metrics exception: {exc}")

    # 3) create/poll/delete only when RUNPOD key exists
    if not RUNPOD_KEY:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        print("RUNPOD missing: skipped create/get/delete flow.")
        return 0 if result.health_ok and result.metrics_ok else 1

    result.deployment_attempted = True
    payload = {
        "hf_model_id": MODEL,
        "gpu_tier": GPU_TIER,
        "hf_token": HF_TOKEN,
        "user_webhook_url": WEBHOOK_URL,
        "cache_scope": os.environ.get("CACHE_SCOPE", "off"),
    }
    if payload["cache_scope"] == "private":
        if not USER_S3_URL or not USER_AWS_ACCESS_KEY_ID or not USER_AWS_SECRET_ACCESS_KEY:
            _fail(
                result,
                "private cache requires USER_S3_URL, USER_AWS_ACCESS_KEY_ID, USER_AWS_SECRET_ACCESS_KEY",
            )
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            return 1
        payload["user_s3_url"] = USER_S3_URL
        payload["user_aws_access_key_id"] = USER_AWS_ACCESS_KEY_ID
        payload["user_aws_secret_access_key"] = USER_AWS_SECRET_ACCESS_KEY
        if USER_AWS_ENDPOINT_URL:
            payload["user_aws_endpoint_url"] = USER_AWS_ENDPOINT_URL

    started = time.perf_counter()
    create_resp = requests.post(
        f"{API_BASE}/v1/deployments",
        json=payload,
        headers=_headers(),
        timeout=45,
    )
    result.create_status_code = create_resp.status_code
    result.create_latency_ms = round((time.perf_counter() - started) * 1000, 2)

    if create_resp.status_code >= 400:
        _fail(result, f"create failed: {create_resp.status_code} {create_resp.text[:500]}")
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 1

    create_data = create_resp.json()
    deployment_id = create_data.get("deployment_id")
    if not deployment_id:
        _fail(result, f"create missing deployment_id: {create_data}")
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 1

    result.deployment_id = deployment_id

    # poll
    poll_start = time.perf_counter()
    deadline = poll_start + DEPLOY_TIMEOUT_SECONDS
    while time.perf_counter() < deadline:
        status_resp = requests.get(
            f"{API_BASE}/v1/deployments/{deployment_id}", headers=_headers(), timeout=30
        )
        if status_resp.status_code != 200:
            _fail(result, f"poll failed: {status_resp.status_code} {status_resp.text[:300]}")
            time.sleep(POLL_SECONDS)
            continue

        body = status_resp.json()
        status = body.get("status", "unknown")
        result.final_status = status
        if status == "ready":
            result.ready_latency_sec = round(time.perf_counter() - poll_start, 2)
            break
        if status in {"failed", "webhook_failed", "deleted"}:
            _fail(result, f"terminal non-ready status: {status} error={body.get('error')}")
            break
        time.sleep(POLL_SECONDS)
    else:
        _fail(result, f"deploy timeout after {DEPLOY_TIMEOUT_SECONDS}s")

    # delete best-effort
    if deployment_id:
        try:
            delete_resp = requests.delete(
                f"{API_BASE}/v1/deployments/{deployment_id}", headers=_headers(), timeout=30
            )
            result.delete_status_code = delete_resp.status_code
            if delete_resp.status_code not in {200, 202, 204}:
                _fail(result, f"delete failed: {delete_resp.status_code} {delete_resp.text[:300]}")
        except Exception as exc:
            _fail(result, f"delete exception: {exc}")

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
