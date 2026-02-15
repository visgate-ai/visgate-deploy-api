#!/usr/bin/env python3
"""
E2E test: Istanbul image via API with timing metrics.
Measures:
  T1: Deploy request -> Pod ready (seconds)
  T2: Inference request -> Response received (seconds)
  Total: T1 + T2

Local dev: When webhook can't reach localhost, use --dev-fallback to
auto-detect Runpod endpoint and mark Firestore ready.
"""
import os
import sys
import time
import requests
import base64

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
BEARER = os.environ.get("BEARER", "visgate")
# SD-Turbo: fastest model, 2 steps, good quality
MODEL = "stabilityai/sd-turbo"
GPU_TIER = "A10"  # AMPERE_24 - optimal cost/speed for SD
ISTANBUL_PROMPT = (
    "A beautiful panoramic view of Istanbul with the Bosphorus strait, "
    "Hagia Sophia and minarets, golden hour, photorealistic, 4k"
)
DEV_FALLBACK_WAIT = 120  # seconds before trying Runpod+Firestore bypass
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "https://httpbin.org/post")


def _runpod_find_endpoint(runpod_key: str, deploy_id: str) -> str | None:
    """Query Runpod for endpoint named visgate-{deploy_id}. Return endpoint id or None."""
    url = "https://api.runpod.io/graphql"
    query = 'query { myself { endpoints { id name } } }'
    r = requests.post(url, params={"api_key": runpod_key}, json={"query": query}, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    endpoints = data.get("data", {}).get("myself", {}).get("endpoints", []) or []
    name = f"visgate-{deploy_id}"
    for ep in endpoints:
        if ep.get("name") == name:
            return ep.get("id")
    return None


def _firestore_mark_ready(deploy_id: str, endpoint_id: str, project: str, collection: str) -> bool:
    """Update Firestore deployment doc to ready."""
    try:
        from google.cloud import firestore
        client = firestore.Client(project=project)
        endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        client.collection(collection).document(deploy_id).update({
            "status": "ready",
            "endpoint_url": endpoint_url,
            "runpod_endpoint_id": endpoint_id,
            "ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return True
    except Exception as e:
        print(f"    [dev-fallback] Firestore update failed: {e}")
        return False


def main():
    dev_fallback = "--dev-fallback" in sys.argv
    print("=" * 60)
    print("E2E: Istanbul Image - Timed Deployment + Inference")
    if dev_fallback:
        print("(dev-fallback enabled: will bypass webhook if stuck)")
    print("=" * 60)

    runpod_key = os.environ.get("RUNPOD")
    hf_token = os.environ.get("HF")
    if not runpod_key:
        print("Error: RUNPOD env var required")
        sys.exit(1)

    gcp_project = os.environ.get("GCP_PROJECT_ID", "visgate")
    firestore_coll = os.environ.get("FIRESTORE_COLLECTION_DEPLOYMENTS", "visgate_deploy_api_deployments")

    payload = {
        "hf_model_id": MODEL,
        "gpu_tier": GPU_TIER,
        "user_runpod_key": runpod_key,
        "hf_token": hf_token,
        "user_webhook_url": WEBHOOK_URL,
    }

    headers = {"Authorization": f"Bearer {BEARER}"}

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
    dev_fallback_attempted = False
    while True:
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
        # Dev fallback: if stuck at creating_endpoint, detect Runpod endpoint and mark Firestore
        if dev_fallback and st == "creating_endpoint" and elapsed >= DEV_FALLBACK_WAIT and not dev_fallback_attempted:
            dev_fallback_attempted = True
            print(f"    [dev-fallback] Stuck >{DEV_FALLBACK_WAIT}s, checking Runpod...")
            ep_id = _runpod_find_endpoint(runpod_key, deploy_id)
            if ep_id:
                print(f"    [dev-fallback] Found endpoint {ep_id}, marking Firestore ready")
                if _firestore_mark_ready(deploy_id, ep_id, gcp_project, firestore_coll):
                    endpoint_url = f"https://api.runpod.ai/v2/{ep_id}"
                    break
            else:
                print("    [dev-fallback] Endpoint not found yet")
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


if __name__ == "__main__":
    main()
