#!/usr/bin/env python3
"""Minimal E2E test: deploy minimal models, run inference, download results."""

import json
import os
import sys
import time
import urllib.request
import urllib.error

API_BASE = "https://visgate-deploy-api-93820292919.europe-west3.run.app/deployapi"
RUNPOD_KEY = open(os.path.join(os.path.dirname(__file__), "..", "keys")).read().strip().split("\n")[0]
HF_TOKEN = open(os.path.join(os.path.dirname(__file__), "..", "keys")).read().strip().split("\n")[1]

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_KEY}",
    "Content-Type": "application/json",
}

# Minimal models for each modality
TESTS = {
    "text_to_image": {
        "hf_model_id": "stabilityai/sd-turbo",
        "task": "text_to_image",
        "input": {
            "prompt": "A red apple on a white table, studio lighting",
            "num_inference_steps": 2,
            "width": 512,
            "height": 512,
        },
    },
    "speech_to_text": {
        "hf_model_id": "openai/whisper-large-v3-turbo",
        "task": "speech_to_text",
        "input": {
            "audio_url": "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings/0_george_0.wav",
        },
    },
    "text_to_video": {
        "hf_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "task": "text_to_video",
        "input": {
            "prompt": "a red cube rotating on white table",
            "num_inference_steps": 4,
            "num_frames": 8,
            "fps": 8,
        },
    },
    "prompt2audio": {
        "hf_model_id": "cvssp/audioldm2",
        "task": "text_to_speech",
        "input": {
            "prompt": "a dog barking in a park",
            "num_inference_steps": 4,
        },
    },
    "image2video": {
        "hf_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "task": "text_to_video",
        "input": {
            "prompt": "a cat walking on grass, slow motion",
            "num_inference_steps": 4,
            "num_frames": 8,
            "fps": 8,
        },
    },
    "image_prompt2image": {
        "hf_model_id": "stabilityai/sd-turbo",
        "task": "image_to_image",
        "input": {
            "prompt": "a red sports car parked on a sunny street",
            "input_image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
            "num_inference_steps": 2,
        },
    },
}


def api_request(method, path, body=None):
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            if not raw:
                return {}, resp.status
            return json.loads(raw), resp.status
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")[:500]
        print(f"  HTTP {e.code}: {body_text}")
        return {"error": body_text}, e.code


def deploy_and_wait(test_name, config, timeout=600):
    print(f"\n{'='*60}")
    print(f"[{test_name}] Deploying {config['hf_model_id']}...")
    body = {
        "hf_model_id": config["hf_model_id"],
        "task": config["task"],
        "hf_token": HF_TOKEN,
    }
    resp, code = api_request("POST", "/v1/deployments", body)
    if code >= 400:
        print(f"  FAIL: Deployment create failed: {resp}")
        return None
    dep_id = resp.get("deployment_id")
    print(f"  deployment_id={dep_id}")

    start = time.time()
    last_status = ""
    while time.time() - start < timeout:
        resp, _ = api_request("GET", f"/v1/deployments/{dep_id}")
        status = resp.get("status", "unknown")
        if status != last_status:
            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s] status={status}")
            last_status = status
        if status == "ready":
            return dep_id
        if status == "failed":
            print(f"  FAIL: {resp.get('error', 'unknown')}")
            return None
        time.sleep(5)
    print(f"  TIMEOUT after {timeout}s")
    return None


def run_inference(test_name, dep_id, config, timeout=300):
    print(f"[{test_name}] Submitting inference job...")
    body = {
        "deployment_id": dep_id,
        "task": config["task"],
        "input": config["input"],
    }
    resp, code = api_request("POST", "/v1/inference/jobs", body)
    if code >= 400:
        print(f"  FAIL: Job submit failed: {resp}")
        return None
    job_id = resp.get("job_id")
    print(f"  job_id={job_id}")

    start = time.time()
    last_status = ""
    while time.time() - start < timeout:
        resp, _ = api_request("GET", f"/v1/inference/jobs/{job_id}")
        status = resp.get("status", "unknown")
        if status != last_status:
            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s] job_status={status}")
            last_status = status
        if status in ("COMPLETED", "completed"):
            output = resp.get("output") or resp.get("result") or {}
            artifact = output.get("artifact") if isinstance(output, dict) else None
            print(f"  SUCCESS! output keys: {list(output.keys()) if isinstance(output, dict) else type(output)}")
            if artifact:
                print(f"  artifact: {artifact}")
            return resp
        if status in ("FAILED", "failed", "CANCELLED"):
            print(f"  FAIL: {resp.get('error', resp.get('output', 'unknown'))}")
            return resp
        time.sleep(5)
    print(f"  TIMEOUT after {timeout}s")
    return None


def download_artifact(result, test_name):
    """Download artifact URL to local file."""
    if not result:
        return
    output = result.get("output") or result.get("result") or {}
    if not isinstance(output, dict):
        print(f"  [{test_name}] No artifact to download (text output)")
        return
    artifact = output.get("artifact")
    if not artifact:
        print(f"  [{test_name}] No artifact in output")
        return
    url = artifact.get("url")
    if not url:
        print(f"  [{test_name}] No artifact URL")
        return

    ext_map = {"image/png": "png", "video/mp4": "mp4", "audio/wav": "wav"}
    ext = ext_map.get(artifact.get("content_type", ""), "bin")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "downloads")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{test_name}_result.{ext}")

    print(f"  [{test_name}] Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, out_path)
        size = os.path.getsize(out_path)
        print(f"  [{test_name}] Saved to {out_path} ({size} bytes)")
    except Exception as e:
        print(f"  [{test_name}] Download failed: {e}")


def cleanup(dep_id, test_name):
    print(f"  [{test_name}] Deleting deployment {dep_id}...")
    resp, code = api_request("DELETE", f"/v1/deployments/{dep_id}")
    if code < 300:
        print(f"  [{test_name}] Deleted.")
    else:
        print(f"  [{test_name}] Delete returned {code}")


def main():
    modalities = sys.argv[1:] if len(sys.argv) > 1 else list(TESTS.keys())
    results = {}
    deployment_ids = {}

    for test_name in modalities:
        if test_name not in TESTS:
            print(f"Unknown test: {test_name}")
            continue
        config = TESTS[test_name]
        dep_id = deploy_and_wait(test_name, config, timeout=600)
        if dep_id:
            deployment_ids[test_name] = dep_id
            result = run_inference(test_name, dep_id, config, timeout=600)
            results[test_name] = result
            download_artifact(result, test_name)
        else:
            results[test_name] = None

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    for name, result in results.items():
        if result is None:
            status = "FAILED (deploy)"
        elif isinstance(result, dict):
            s = result.get("status", "unknown")
            status = "OK" if s in ("COMPLETED", "completed") else f"FAILED ({s})"
        else:
            status = "UNKNOWN"
        print(f"  {name}: {status}")

    # Cleanup
    print(f"\n{'='*60}")
    print("CLEANUP:")
    for test_name, dep_id in deployment_ids.items():
        cleanup(dep_id, test_name)

    # Exit code
    all_ok = all(
        isinstance(r, dict) and r.get("status") in ("COMPLETED", "completed")
        for r in results.values()
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
