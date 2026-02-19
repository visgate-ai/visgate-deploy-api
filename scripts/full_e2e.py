#!/usr/bin/env python3
import os
import sys
import time
import requests
import base64

API_BASE = os.environ.get("API_BASE", "https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app")
MODEL = "stabilityai/sd-turbo"
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "https://httpbin.org/post")
MAX_DEPLOY_WAIT_SECONDS = int(os.environ.get("MAX_DEPLOY_WAIT_SECONDS", "1800"))

def main():
    print(f"--- Starting Full E2E Flow ---")
    runpod_key = os.environ.get("RUNPOD")
    if not runpod_key:
        print("Error: RUNPOD environment variable not set.")
        sys.exit(1)
    
    # 1. Trigger Deployment
    deploy_url = f"{API_BASE}/v1/deployments"
    payload = {
        "hf_model_id": MODEL,
        "gpu_tier": "A10", # AMPERE_24
        "hf_token": os.environ.get("HF"),
        "user_webhook_url": WEBHOOK_URL
    }

    print(f"Triggering deployment for {MODEL}...")
    headers = {"Authorization": f"Bearer {runpod_key}", "X-Runpod-Api-Key": runpod_key}
    r = requests.post(deploy_url, json=payload, headers=headers)
    if r.status_code >= 400:
        print(f"Failed to trigger: {r.text}")
        sys.exit(1)
    
    deploy_data = r.json()
    if "deployment_id" not in deploy_data:
        print(f"No deployment_id in response: {deploy_data}")
        sys.exit(1)
    deploy_id = deploy_data["deployment_id"]
    print(f"Deployment ID: {deploy_id}")

    # 2. Poll Status
    status_url = f"{API_BASE}/v1/deployments/{deploy_id}"
    ready = False
    endpoint_url = None
    
    print("Waiting for endpoint to be ready (this can take 2-5 mins)...")
    start_time = time.time()
    deadline = start_time + MAX_DEPLOY_WAIT_SECONDS
    while not ready:
        if time.time() > deadline:
            print(f"Timed out after {MAX_DEPLOY_WAIT_SECONDS}s waiting for ready")
            sys.exit(1)
        elapsed = time.time() - start_time
        r = requests.get(status_url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            status = data.get("status")
            print(f"  [{int(elapsed)}s] Status: {status}")
            
            if status == "ready":
                ready = True
                endpoint_url = data.get("endpoint_url")
            elif status == "failed":
                print(f"Deployment failed: {data.get('error')}")
                sys.exit(1)
        else:
            print(f"Error polling: {r.text}")
        
        if not ready:
            time.sleep(10)
    
    print(f"Endpoint READY: {endpoint_url}")

    # 3. Trigger Inference (using e2e_istanbul logic)
    print("Sending Istanbul prompt...")
    endpoint_root = endpoint_url[:-4] if endpoint_url.endswith("/run") else endpoint_url
    run_url = f"{endpoint_root}/runsync"
    inference_payload = {
        "input": {
            "prompt": "A beautiful panoramic view of Istanbul with the Bosphorus strait, Hagia Sophia and minarets, golden hour, photorealistic, 4k",
            "num_inference_steps": 2
        }
    }
    
    headers = {"Authorization": f"Bearer {runpod_key}"}
    
    # Poll for inference result if queued
    inference_done = False
    while not inference_done:
        r = requests.post(run_url, json=inference_payload, headers=headers)
        r.raise_for_status()
        inf_data = r.json()
        
        job_status = inf_data.get("status")
        job_id = inf_data.get("id")
        
        if job_status == "COMPLETED":
            inference_done = True
            output = inf_data["output"]
        elif job_status in ["IN_QUEUE", "IN_PROGRESS"]:
            print(f"  Inference {job_status}... polling job {job_id}")
            time.sleep(5)
            # Correct Runpod status URL: v2/{id}/status/{job_id}
            run_url = f"{endpoint_root}/status/{job_id}"
            continue
        elif job_status == "FAILED":
            error_msg = str(inf_data.get('error', ''))
            if "still loading" in error_msg.lower():
                print("  Model still loading, waiting 10s...")
                time.sleep(10)
                continue
            print(f"Inference FAILED: {error_msg}")
            sys.exit(1)
        else:
            # If COMPLETED on first try
            inference_done = True
            output = inf_data.get("output", {})

    # 4. Save Image
    image_b64 = output.get("image_base64") or output.get("image")
    if not image_b64:
        print(f"No image in output: {output}")
        sys.exit(1)
    
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",")[1]
        
    out_path = os.path.join(os.getcwd(), "istanbul_final.png")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(image_b64))
    
    print(f"SUCCESS! Image saved to {out_path}")
    
    # 5. Cleanup
    print("Cleaning up Runpod endpoint...")
    # We could send a DELETE but for now we'll just report success
    print("Done.")

if __name__ == "__main__":
    main()
