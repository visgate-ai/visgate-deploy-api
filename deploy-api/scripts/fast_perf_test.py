import requests
import time
import os
import sys

API_URL = "https://visgate-deploy-api-wxup7pxrsa-ey.a.run.app"
RUNPOD_KEY = os.environ.get("RUNPOD_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

MODELS = [
    {"label": "IMAGE", "model_id": "stabilityai/sdxl-turbo", "task": "text2img"},
    {"label": "AUDIO", "model_id": "openai/whisper-large-v3", "task": "speech_to_text"},
    {"label": "VIDEO", "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "task": "text2video"}
]

headers = {"Authorization": f"Bearer {RUNPOD_KEY}"}

def run_test(cfg):
    print(f"[{cfg['label']}] Requesting {cfg['model_id']} ({cfg['task']})...")
    t0 = time.time()
    
    resp = requests.post(f"{API_URL}/v1/deployments", json={
        "hf_model_id": cfg["model_id"],
        "task": cfg["task"],
        "hf_token": HF_TOKEN,
    }, headers=headers)
    
    if resp.status_code != 202:
        print(f"[{cfg['label']}] Failed. {resp.text}")
        return {"label": cfg['label'], "duration": 0, "status": "failed_post"}
    
    dep_id = resp.json()["deployment_id"]
    print(f"[{cfg['label']}] Dep ID: {dep_id}. Polling...")
    
    fail_count = 0
    while True:
        try:
            poll = requests.get(f"{API_URL}/v1/deployments/{dep_id}", headers=headers).json()
            status = poll["status"]
            if status in ("ready", "failed"):
                t_r2 = poll.get("t_r2_sync_s", "-")
                t_load = poll.get("t_model_load_s", "-")
                dur = time.time() - t0
                print(f"[{cfg['label']}] {status.upper()} in {dur:.2f}s (R2: {t_r2}s, Load: {t_load}s)")
                
                # Delete it
                if status == "ready":
                    print(f"[{cfg['label']}] Tearing down {dep_id}...")
                    requests.delete(f"{API_URL}/v1/deployments/{dep_id}", headers=headers)
                return {"label": cfg['label'], "duration": dur, "status": status, "r2": t_r2, "load": t_load}
        except Exception as e:
            fail_count += 1
            if fail_count > 5:
                return {"label": cfg['label'], "duration": 0, "status": "poll_error"}
        time.sleep(2)

import concurrent.futures

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as x:
    futures = [x.submit(run_test, m) for m in MODELS]
    for f in concurrent.futures.as_completed(futures):
        results.append(f.result())

print("\n\n=== HIZLI BAŞLATMA PERFORMANS TABLOSU ===")
print(f"{'MODALİTE':<10} | {'TOPLAM SÜRE':<15} | {'R2 İNDİRME':<12} | {'VRAM YÜKLEME':<12} | {'DURUM':<10}")
print("-" * 75)
for r in results:
    s = f"{r['duration']:.2f}s"
    print(f"{r['label']:<10} | {s:<15} | {str(r['r2'])+'s':<12} | {str(r['load'])+'s':<12} | {r['status']:<10}")

