#!/usr/bin/env python3
"""
End-to-end test: send an Istanbul prompt to the Runpod endpoint, save the image, and open it.
Supports polling if the request is queued.
"""
import base64
import json
import os
import subprocess
import sys
import time

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)

# Repo root .env.local
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ENV_LOCAL = os.path.join(REPO_ROOT, ".env.local")

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

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "mxa0he79ljwapv")
RUNPOD_BASE = "https://api.runpod.ai/v2"
PROMPT = "A beautiful panoramic view of Istanbul with the Bosphorus strait, Hagia Sophia and minarets, golden hour, photorealistic, 4k"

def main():
    env = load_env()
    api_key = env.get("RUNPOD", "").strip()
    if not api_key:
        print("RUNPOD not found. Set RUNPOD=... in .env.local.", file=sys.stderr)
        sys.exit(1)

    url = f"{RUNPOD_BASE}/{ENDPOINT_ID}/runsync"
    payload = {"input": {"prompt": PROMPT, "num_inference_steps": 28}}
    print(f"Sending prompt to {ENDPOINT_ID}...")
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    r = requests.post(url, json=payload, headers=headers, timeout=180)
    r.raise_for_status()
    data = r.json()

    job_id = data.get("id")
    status = data.get("status")

    if status in ["IN_QUEUE", "IN_PROGRESS"]:
        print(f"Job {job_id} is {status}, polling...")
        done = False
        while not done:
            status_url = f"{RUNPOD_BASE}/{ENDPOINT_ID}/status/{job_id}"
            r = requests.get(status_url, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            status = data.get("status")
            if status == "COMPLETED":
                done = True
            elif status == "FAILED":
                print("Job failed:", data.get("error"))
                sys.exit(1)
            else:
                print(f"  status: {status}...")
                time.sleep(10)

    if "output" not in data:
        print("Output not in response:", json.dumps(data, indent=2)[:500])
        sys.exit(1)

    out = data["output"]
    if isinstance(out, dict) and "error" in out:
        print("Error in output:", out["error"])
        sys.exit(1)
    
    # Some workers return direct URL or base64
    b64 = None
    if isinstance(out, dict):
        b64 = out.get("image_base64") or out.get("image")
    
    if not b64:
        print("Could not find image in output:", out)
        sys.exit(1)

    if b64.startswith("data:image"):
        b64 = b64.split(",")[1]

    out_path = os.path.join(REPO_ROOT, "output.png")
    raw = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(raw)
    print("Image saved:", out_path)

    # Show image
    try:
        if sys.platform == "linux":
            subprocess.run(["xdg-open", out_path], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", out_path], check=False)
        else:
            subprocess.run(["start", "", out_path], shell=True, check=False)
        print("Image opened.")
    except Exception:
        print("Could not open image viewer automatically.")

if __name__ == "__main__":
    main()
