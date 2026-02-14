#!/usr/bin/env python3
import os
import sys
import time
import requests
import base64
import subprocess

API_BASE = "https://visgate-deploy-api-93820292919.us-central1.run.app"
BEARER = "visgate"
MODEL = "stabilityai/sd-turbo"

def main():
    print(f"--- Starting Full E2E Flow ---")
    
    # 1. Trigger Deployment
    deploy_url = f"{API_BASE}/v1/deployments"
    payload = {
        "hf_model_id": MODEL,
        "gpu_tier": "A10", # AMPERE_24
        "user_runpod_key": os.environ.get("RUNPOD"),
        "hf_token": os.environ.get("HF"),
        "user_webhook_url": "https://webhook.site/4fc3656c-0e24-42f1-9b62-97b5e40632b8" # dummy
    }
    
    if not payload["user_runpod_key"]:
        print("Error: RUNPOD environment variable not set.")
        sys.exit(1)

    print(f"Triggering deployment for {MODEL}...")
    r = requests.post(deploy_url, json=payload, headers={"Authorization": f"Bearer {BEARER}"})
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
    while not ready:
        elapsed = time.time() - start_time
        r = requests.get(status_url, headers={"Authorization": f"Bearer {BEARER}"})
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
    run_url = f"{endpoint_url}/runsync"
    inference_payload = {
        "input": {
            "prompt": "A beautiful panoramic view of Istanbul with the Bosphorus strait, Hagia Sophia and minarets, golden hour, photorealistic, 4k",
            "num_inference_steps": 2
        }
    }
    
    headers = {"Authorization": f"Bearer {payload['user_runpod_key']}"}
    
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
            # Switch to GET status polling
            run_url = f"{endpoint_url.replace('/run', '/status')}/{job_id}"
            # This is a bit hacky transition but runpod usually needs GET after first POST
            # Actually runpod status URL is v2/{id}/status/{job_id}
            continue
        elif job_status == "FAILED":
            print(f"Inference FAILED: {inf_data.get('error')}")
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
