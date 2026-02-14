# Visgate Deploy API

**Visgate Serverless**, Hugging Face üzerindeki diffusion modellerini (Flux, SDXL vb.) **Runpod Serverless** üzerinde çalıştırmanızı sağlayan açık kaynaklı bir araçtır.

Bizim sağladığımız **ücretsiz Orchestrator API** sayesinde, kendi sunucunuzu veya bulut altyapınızı kurmanıza gerek kalmadan modelleri deploy edebilirsiniz. Sadece Runpod API Key'iniz yeterlidir.

---

## 🚀 Nasıl Çalışır?

1.  **İstek Atın:** Bizim API endpoint'imize (`POST /v1/deployments`) istediğiniz modeli ve Runpod key'inizi gönderin.
2.  **Orchestrator İşlesin:** Sistemimiz Runpod hesabınızda gerekli ayarları yapar ve modeli hazırlar.
3.  **Webhook Bekleyin:** Model hazır olduğunda, belirttiğiniz URL'e bir webhook göndeririz.
4.  **Kullanın:** Webhook ile gelen endpoint adresine istek atarak görsel üretmeye başlayın.

---

## 🔌 API Kullanımı (Hosted Service)

Aşağıdaki API adresi herkesin kullanımına açıktır.

**Base URL:** `https://api.visgate.io` (Örnek URL - Deployment sonrası güncellenecek)

### 1. Deployment Oluşturma

**POST** `/v1/deployments`
**Header:** `Authorization: Bearer <VISGATE_API_KEY>` (Discord/Community üzerinden talep edin)

```json
{
  "hf_model_id": "black-forest-labs/FLUX.1-schnell",
  "user_runpod_key": "YOUR_RUNPOD_API_KEY",
  "user_webhook_url": "https://your-server.com/webhook",
  "gpu_tier": "3090" // Opsiyonel (3090, A40, A100 vb.)
}
```

### 2. Webhook Yanıtı (Başarılı)

Model hazır olduğunda `user_webhook_url` adresine şu JSON gelir:

```json
{
  "event": "deployment_ready",
  "deployment_id": "dep_2024_abc123",
  "status": "ready",
  "endpoint_url": "https://api.runpod.ai/v2/xxxx-xxxx/run",
  "model_id": "black-forest-labs/FLUX.1-schnell",
  "gpu_allocated": "RTX 3090",
  "duration_seconds": 120.5,
  "usage_example": {
    "method": "POST",
    "url": "https://api.runpod.ai/v2/xxxx-xxxx/run",
    "headers": {
      "Authorization": "Bearer YOUR_RUNPOD_API_KEY"
    },
    "body": {
      "input": {
        "prompt": "An astronaut riding a horse in photorealistic style",
        "num_inference_steps": 28,
        "guidance_scale": 3.5
      }
    }
  }
}
```

### 3. Görsel Üretme (Runpod)

Webhook'tan gelen URL'e istek atın:

```bash
curl -X POST https://api.runpod.ai/v2/xxxx-xxxx/run \
     -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
           "input": {
               "prompt": "Cyberpunk city with neon lights",
               "num_inference_steps": 25
           }
         }'
```

---

## 🛠️ Kendi Bünyenizde Çalıştırma (Self-Hosting)

Eğer bu servisi kendiniz (GCP Cloud Run üzerinde) barındırmak isterseniz:

1.  **deployment-orchestrator:** GCP Cloud Run'a deploy edin. Firestore, Cloud Tasks ve Secret Manager gerektirir.
2.  **inference:** Docker image'ını build edip Docker Hub'a atın.
3.  Detaylı kurulum rehberi için [deployment-orchestrator/README.md](deployment-orchestrator/README.md) dosyasına bakın.

---

## 📜 Lisans

MIT License.
