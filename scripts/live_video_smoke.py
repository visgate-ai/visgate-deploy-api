#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import requests


API = os.environ.get("API_BASE", "https://visgate-deploy-api-wxup7pxrsa-ey.a.run.app").rstrip("/")
RUNPOD_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
PROJECT = os.environ.get("GCP_PROJECT_ID", "visgate")
HEADERS = {"Authorization": f"Bearer {RUNPOD_KEY}", "Content-Type": "application/json"}


def read_secret(name: str) -> str:
    result = subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest", f"--secret={name}", "--project", PROJECT],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def clip(payload: object, limit: int = 2000) -> str:
    text = json.dumps(payload, ensure_ascii=True)
    return text if len(text) <= limit else f"{text[:limit]}..."


def main() -> int:
    if not RUNPOD_KEY:
        raise SystemExit("Set RUNPOD_API_KEY")

    deployment_id: str | None = None
    try:
        r2_access_id = read_secret("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_RW")
        r2_secret = read_secret("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_RW")
        bucket_name = read_secret("VISGATE_DEPLOY_API_INFERENCE_R2_BUCKET_NAME")
        r2_endpoint = read_secret("VISGATE_DEPLOY_API_S3_API_R2")

        create_resp = requests.post(
            f"{API}/v1/deployments",
            headers=HEADERS,
            json={
                "hf_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "task": "text_to_video",
                "cache_scope": "shared",
            },
            timeout=90,
        )
        create_data = create_resp.json()
        print("DEPLOY_CREATE", create_resp.status_code, clip(create_data), flush=True)
        create_resp.raise_for_status()
        deployment_id = create_data["deployment_id"]

        status_data = None
        for attempt in range(1, 201):
            resp = requests.get(
                f"{API}/v1/deployments/{deployment_id}",
                headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
                timeout=60,
            )
            resp.raise_for_status()
            status_data = resp.json()
            print(
                "DEPLOY_STATUS",
                attempt,
                status_data.get("status"),
                status_data.get("worker_profile"),
                status_data.get("worker_image"),
                flush=True,
            )
            if status_data.get("status") in {"ready", "failed", "webhook_failed"}:
                break
            time.sleep(10)

        if status_data.get("status") != "ready":
            raise RuntimeError(f"deployment did not become ready: {clip(status_data)}")

        job_resp = requests.post(
            f"{API}/v1/inference/jobs",
            headers=HEADERS,
            json={
                "deployment_id": deployment_id,
                "input": {
                    "prompt": "A red car driving on a coastal road at sunset",
                    "num_inference_steps": 2,
                    "num_frames": 2,
                    "fps": 2,
                    "guidance_scale": 4.0,
                },
                "s3Config": {
                    "accessId": r2_access_id,
                    "accessSecret": r2_secret,
                    "bucketName": bucket_name,
                    "endpointUrl": r2_endpoint,
                    "keyPrefix": "smoke-tests/video-results",
                },
            },
            timeout=90,
        )
        job_data = job_resp.json()
        print("JOB_CREATE", job_resp.status_code, clip(job_data), flush=True)
        job_resp.raise_for_status()
        job_id = job_data["job_id"]

        final_job = None
        for attempt in range(1, 121):
            resp = requests.get(
                f"{API}/v1/inference/jobs/{job_id}",
                headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
                timeout=60,
            )
            resp.raise_for_status()
            final_job = resp.json()
            metrics = final_job.get("metrics") or {}
            print(
                "JOB_STATUS",
                attempt,
                final_job.get("status"),
                metrics.get("queue_ms"),
                metrics.get("execution_ms"),
                flush=True,
            )
            if final_job.get("status") in {"completed", "failed", "cancelled", "expired"}:
                break
            time.sleep(10)

        print("FINAL_JOB", clip(final_job), flush=True)
        return 0 if final_job and final_job.get("status") == "completed" else 1
    finally:
        if deployment_id:
            try:
                delete_resp = requests.delete(
                    f"{API}/v1/deployments/{deployment_id}",
                    headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
                    timeout=60,
                )
                print("DELETE", delete_resp.status_code, flush=True)
            except Exception as exc:
                print("DELETE_ERROR", repr(exc), flush=True)


if __name__ == "__main__":
    sys.exit(main())