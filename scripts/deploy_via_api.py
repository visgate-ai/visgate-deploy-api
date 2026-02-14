#!/usr/bin/env python3
"""
Sadece GCP API'mize istek atar; HF, Runpod, Docker Hub'a biz dokunmayız.
API'ye: keyler (Runpod, isteğe bağlı HF) + HF model adı. Cevabı webhook ile alırız.

Kullanım:
  python3 scripts/deploy_via_api.py
  # .env.local'dan RUNPOD, isteğe bağlı HF okunur.
  # HF model: hf_model_id veya model_name (örn. FLUX.1-schnell) verebilirsiniz.
"""
import json
import os
import sys
import time
import urllib.request

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ENV_LOCAL = os.path.join(REPO_ROOT, ".env.local")

# Bizim GCP API (tek giriş noktası)
API_BASE = os.environ.get("VISGATE_API_URL", "https://deployment-orchestrator-93820292919.europe-west1.run.app")
# Bearer token (API herhangi bir token kabul ediyor)
API_BEARER = os.environ.get("VISGATE_API_BEARER", "visgate")

# Varsayılan model (HF model ID veya model_name) - sdxl-turbo: daha küçük, hızlı
DEFAULT_HF_MODEL = os.environ.get("HF_MODEL", "stabilityai/sdxl-turbo")


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


def create_webhook_url():
    """Webhook.site ile tek kullanımlık URL al."""
    req = urllib.request.Request(
        "https://webhook.site/token",
        data=b"{}",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
    uuid = data.get("uuid")
    if not uuid:
        raise RuntimeError("Webhook.site token alınamadı: " + str(data))
    return f"https://webhook.site/{uuid}", uuid


def api_post(path, body):
    """GCP API'mize POST at."""
    url = f"{API_BASE.rstrip('/')}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_BEARER}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def api_get(path):
    """GCP API'den GET."""
    url = f"{API_BASE.rstrip('/')}{path}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {API_BEARER}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def get_webhook_requests(token_uuid):
    """Webhook.site'ta gelen istekleri al (en yeni)."""
    url = f"https://webhook.site/token/{token_uuid}/requests?sorting=newest&per_page=5"
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode())
    return data.get("data") or []


def main():
    env = load_env()
    runpod_key = env.get("RUNPOD", "").strip()
    hf_token = env.get("HF", "").strip() or None

    if not runpod_key:
        print("HATA: .env.local içinde RUNPOD=... tanımlayın.", file=sys.stderr)
        sys.exit(1)

    # Webhook URL (orchestrator cevabı buraya POST edecek)
    webhook_url, token_uuid = create_webhook_url()
    print("Webhook URL (cevap buraya gelecek):", webhook_url)

    # Model: hf_model_id kullan (doğrudan HF model ID)
    hf_model = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
    body = {
        "hf_model_id": hf_model,
        "user_runpod_key": runpod_key,
        "user_webhook_url": webhook_url,
    }
    if hf_token:
        body["hf_token"] = hf_token

    print("GCP API'ye gönderiliyor (sadece bizim API; HF/Runpod/Docker Hub API'ye gidilmiyor)...")
    try:
        resp = api_post("/v1/deployments", body)
    except urllib.error.HTTPError as e:
        print("API hatası:", e.code, e.read().decode()[:500], file=sys.stderr)
        sys.exit(1)

    deployment_id = resp.get("deployment_id")
    if not deployment_id:
        print("API yanıtında deployment_id yok:", resp, file=sys.stderr)
        sys.exit(1)

    print("Deployment oluşturuldu:", deployment_id)
    print("Model:", resp.get("model_id"), "| Webhook:", resp.get("webhook_url"))
    print("Cevap webhook ile bekleniyor (veya API'den status poll)...")

    # 1) Webhook.site'tan gelen isteği bekle (orchestrator ready olunca POST atar)
    # 2) Zaman aşımına kadar poll: önce webhook, yoksa GET /v1/deployments/{id}
    deadline = time.monotonic() + 600  # 10 dk
    last_status = None
    webhook_received = None

    while time.monotonic() < deadline:
        # Webhook'ta yeni istek var mı?
        for req in get_webhook_requests(token_uuid):
            try:
                content = req.get("content") or ""
                payload = json.loads(content)
                if payload.get("deployment_id") == deployment_id and payload.get("status") == "ready":
                    webhook_received = payload
                    break
            except (json.JSONDecodeError, TypeError):
                pass
        if webhook_received:
            print("\n--- Webhook ile gelen cevap ---")
            print(json.dumps(webhook_received, indent=2, ensure_ascii=False))
            print("Endpoint URL (inference için):", webhook_received.get("endpoint_url"))
            return

        # Fallback: API'den status oku
        try:
            doc = api_get(f"/v1/deployments/{deployment_id}")
            status = doc.get("status")
            if status != last_status:
                print("  status:", status)
                last_status = status
            if status == "ready":
                print("\n--- API'den durum (ready) ---")
                print("endpoint_url:", doc.get("endpoint_url"))
                return
            if status == "failed":
                print("Deployment failed:", doc.get("error"), file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print("  poll:", e)
        time.sleep(5)

    print("Zaman aşımı (10 dk). Webhook gelmedi; INTERNAL_WEBHOOK_BASE_URL Cloud Run'da set mi?", file=sys.stderr)
    print("Yine de endpoint oluşmuş olabilir: GET", f"{API_BASE}/v1/deployments/{deployment_id}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
