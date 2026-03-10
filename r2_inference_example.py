import os
import time
import requests

API_URL = "http://localhost:8000"
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "rpa_...")
HF_MODEL_ID = "stabilityai/sdxl-turbo"

# R2 Credentials (for the worker to upload the result)
R2_ACCESS_ID = os.environ.get("VISGATE_DEPLOY_API_INFERENCE_R2_ACCESS_KEY_ID_RW", "")
R2_SECRET = os.environ.get("VISGATE_DEPLOY_API_INFERENCE_R2_SECRET_ACCESS_KEY_RW", "")
R2_ENDPOINT = os.environ.get("VISGATE_DEPLOY_API_S3_API_R2", "https://<account>.r2.cloudflarestorage.com")
R2_BUCKET = os.environ.get("VISGATE_DEPLOY_API_INFERENCE_R2_BUCKET_NAME", "visgate-inference")

headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}

def main():
    print(f"Deploying {HF_MODEL_ID}...")
    # 1. Create deployment
    res = requests.post(f"{API_URL}/v1/deployments", headers=headers, json={
        "hf_model_id": HF_MODEL_ID,
        "task": "text_to_image"
    })
    dep_id = res.json()["deployment_id"]

    # 2. Wait for deployment readiness
    for i in range(60):
        print(f"Waiting... [{i}/60]")
        res = requests.get(f"{API_URL}/v1/deployments/{dep_id}", headers=headers)
        if res.json()["status"] == "ready":
            print("Deployment Ready!")
            break
        elif res.json()["status"] == "failed":
            print("Deployment Failed:", res.json()["error"])
            return
        time.sleep(5)

    # 3. Create Inference Job (Results written to R2)
    print("Submitting inference job with S3 config...")
    job_res = requests.post(f"{API_URL}/v1/inference/jobs", headers=headers, json={
        "deployment_id": dep_id,
        "input": {
            "prompt": "A futuristic city in the clouds, high quality",
            "num_inference_steps": 2
        },
        "s3Config": {
            "accessId": R2_ACCESS_ID,
            "accessSecret": R2_SECRET,
            "bucketName": R2_BUCKET,
            "endpointUrl": R2_ENDPOINT,
            "keyPrefix": "my-results"
        }
    })
    
    print("Inference Job response:", job_res.status_code)
    print(job_res.json())

if __name__ == "__main__":
    main()
