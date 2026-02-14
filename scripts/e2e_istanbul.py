#!/usr/bin/env python3
"""
End-to-end test: send an Istanbul prompt to the Runpod endpoint, save the image, and open it.
Reads RUNPOD from .env.local; uses RUNPOD_ENDPOINT_ID env var or default.
"""
import base64
import json
import os
import subprocess
import sys

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
    print("Sending prompt:", PROMPT[:60], "...")
    print("Endpoint:", url)

    r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=180)
    r.raise_for_status()
    data = r.json()

    if "output" not in data:
        print("Output not in response:", json.dumps(data, indent=2)[:500])
        sys.exit(1)

    out = data["output"]
    if "error" in out:
        print("Error:", out["error"])
        sys.exit(1)

    b64 = out.get("image_base64")
    if not b64:
        print("image_base64 missing:", list(out.keys()))
        sys.exit(1)

    out_path = os.path.join(REPO_ROOT, "istanbul.png")
    raw = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(raw)
    print("Image saved:", out_path)

    # Show image
    if sys.platform == "linux":
        subprocess.run(["xdg-open", out_path], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", out_path], check=False)
    else:
        subprocess.run(["start", "", out_path], shell=True, check=False)
    print("Image opened.")

if __name__ == "__main__":
    main()
