#!/usr/bin/env python3
"""
Uçtan uca test: Runpod endpoint'e İstanbul prompt'u gönder, resmi kaydet ve göster.
.env.local'dan RUNPOD okunur; ENDPOINT_ID ortam değişkeni veya varsayılan kullanılır.
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
        print("RUNPOD bulunamadı. .env.local içinde RUNPOD=... tanımlayın.", file=sys.stderr)
        sys.exit(1)

    url = f"{RUNPOD_BASE}/{ENDPOINT_ID}/runsync"
    payload = {"input": {"prompt": PROMPT, "num_inference_steps": 28}}
    print("Prompt gönderiliyor:", PROMPT[:60], "...")
    print("Endpoint:", url)

    r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=180)
    r.raise_for_status()
    data = r.json()

    if "output" not in data:
        print("Yanıtta output yok:", json.dumps(data, indent=2)[:500])
        sys.exit(1)

    out = data["output"]
    if "error" in out:
        print("Hata:", out["error"])
        sys.exit(1)

    b64 = out.get("image_base64")
    if not b64:
        print("image_base64 yok:", list(out.keys()))
        sys.exit(1)

    out_path = os.path.join(REPO_ROOT, "istanbul.png")
    raw = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(raw)
    print("Resim kaydedildi:", out_path)

    # Göster (Linux/Mac/Windows)
    if sys.platform == "linux":
        subprocess.run(["xdg-open", out_path], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", out_path], check=False)
    else:
        subprocess.run(["start", "", out_path], shell=True, check=False)
    print("Resim açıldı.")

if __name__ == "__main__":
    main()
