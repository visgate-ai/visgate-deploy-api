# Deployment Orchestrator – Kullanım Kılavuzu ve Test Örnekleri

Bu doküman, servisi lokalde çalıştırıp endpoint'leri test etme adımlarını ve gerçek test çıktılarını içerir.

---

## 1. Lokalde Çalıştırma

```bash
cd deployment-orchestrator
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export GCP_PROJECT_ID=visgate
PYTHONPATH=. .venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 8080
```

GCP Firestore kullanmak için:

```bash
gcloud auth application-default login
gcloud config set project visgate
```

Tarayıcıda API dokümantasyonu: **http://127.0.0.1:8080/docs**

---

## 2. Endpoint Testleri ve Örnek Yanıtlar

Tüm örnekler `curl` ile test edilmiştir. Base URL: `http://127.0.0.1:8080`.

### 2.1 Liveness – GET /health

Servisin ayakta olduğunu kontrol eder (p99 <10ms hedeflenir).

**İstek:**

```bash
curl -s http://127.0.0.1:8080/health
```

**Örnek yanıt (200 OK):**

```json
{"status":"ok"}
```

---

### 2.2 Root – GET /

Servis bilgisi ve docs linki.

**İstek:**

```bash
curl -s http://127.0.0.1:8080/
```

**Örnek yanıt (200 OK):**

```json
{"service":"deployment-orchestrator","docs":"/docs"}
```

---

### 2.3 Readiness – GET /readiness

Firestore bağlantısını kontrol eder. Başarısızsa 503 döner.

**İstek:**

```bash
curl -s http://127.0.0.1:8080/readiness
```

**Örnek yanıt (200 OK):**

```json
{"status":"ready"}
```

**503 örnek (Firestore erişilemezse):**

```json
{"status":"unready","error":"..."}
```

---

### 2.4 Deployment Oluşturma – POST /v1/deployments

Yeni bir deployment başlatır. İşlem arka planda devam eder; yanıt hemen 202 ile döner.

**Akış (senin kullanımın):**

1. `model_name = get_models(provider=fal, "veo3")` → Bu bilgi **başka bir endpoint’ten** gelir (orchestrator’dan değil).
2. `hf_name = get_hf_name(model_name, provider)` → Orchestrator içinde `model_name` + `provider` ile HF model ID’ye çevrilir.
3. `deploy_model(hf_name, gpu=auto) -> webhook` → Aynı POST ile deployment başlatılır; hazır olunca `user_webhook_url`’e POST atılır.

**Seçenek A – HF model ID ile (doğrudan):**

```bash
curl -s -X POST http://127.0.0.1:8080/v1/deployments \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "hf_model_id": "black-forest-labs/FLUX.1-schnell",
    "user_runpod_key": "rpa_xxx",
    "user_webhook_url": "https://example.com/webhook",
    "gpu_tier": "A40"
  }'
```

**Seçenek B – model_name + provider ile (get_models’ten gelen isim):**

```bash
curl -s -X POST http://127.0.0.1:8080/v1/deployments \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model_name": "veo3",
    "provider": "fal",
    "user_runpod_key": "rpa_xxx",
    "user_webhook_url": "https://example.com/webhook",
    "gpu_tier": "A40"
  }'
```

`model_name` + `provider` verilirse `get_hf_name(model_name, provider)` ile Hugging Face ID’ye çevrilir (örn. `fal` + `veo3` → `black-forest-labs/FLUX.1-schnell`). Ya **hf_model_id** ya da **model_name** (isteğe bağlı **provider**) verilmeli; ikisi birden verilmez.

**Örnek yanıt (202 Accepted):**

```json
{
  "deployment_id": "dep_2026_c1badc17",
  "status": "validating",
  "model_id": "black-forest-labs/FLUX.1-schnell",
  "estimated_ready_seconds": 180,
  "webhook_url": "https://example.com/webhook",
  "created_at": "2026-02-14T13:28:55.130342Z"
}
```

`model_name` + `provider` kullandığında `model_id` yanıtta çözümlenmiş HF ID olur.

**Opsiyonel alanlar:** `gpu_tier` (yoksa otomatik seçilir), `hf_token` (gated modeller için). Model tarafı: **hf_model_id** veya (**model_name** + isteğe bağlı **provider**).

**401 – Yetkisiz (Bearer yok/geçersiz):**

```bash
curl -s -X POST http://127.0.0.1:8080/v1/deployments \
  -H "Content-Type: application/json" \
  -d '{"hf_model_id":"x","user_runpod_key":"y","user_webhook_url":"https://a.com"}'
```

```json
{"error":"UnauthorizedError","message":"Missing or invalid API key","details":{}}
```

---

### 2.5 Deployment Durumu – GET /v1/deployments/{deployment_id}

Belirtilen deployment'ın anlık durumunu, logları ve hata bilgisini döner.

**İstek:**

```bash
curl -s -H "Authorization: Bearer YOUR_API_KEY" \
  "http://127.0.0.1:8080/v1/deployments/dep_2026_a3686d66"
```

**Örnek yanıt (200 OK) – işlem sürerken (creating_endpoint):**

```json
{
  "deployment_id": "dep_2026_a3686d66",
  "status": "creating_endpoint",
  "runpod_endpoint_id": null,
  "endpoint_url": null,
  "gpu_allocated": null,
  "model_vram_gb": 8,
  "logs": [
    {"timestamp": "2026-02-14T13:28:59.228406Z", "level": "INFO", "message": "Validating Hugging Face model"},
    {"timestamp": "2026-02-14T13:29:00.265060Z", "level": "INFO", "message": "HF model validated"},
    {"timestamp": "2026-02-14T13:29:00.499217Z", "level": "INFO", "message": "Selected GPU: NVIDIA A40"}
  ],
  "error": null,
  "created_at": "2026-02-14T13:28:57.253043Z",
  "ready_at": null
}
```

**Olası status değerleri:** `validating` | `selecting_gpu` | `creating_endpoint` | `downloading_model` | `loading_model` | `ready` | `failed` | `webhook_failed` | `deleted`.

**404 – Deployment bulunamadı:**

```bash
curl -s -H "Authorization: Bearer YOUR_API_KEY" \
  "http://127.0.0.1:8080/v1/deployments/dep_unknown_999"
```

```json
{
  "error": "DeploymentNotFoundError",
  "message": "Deployment not found: dep_unknown_999",
  "details": {"deployment_id": "dep_unknown_999"}
}
```

**401 – Yetkisiz:**

```json
{"error":"UnauthorizedError","message":"Missing or invalid API key","details":{}}
```

---

### 2.6 Deployment Silme – DELETE /v1/deployments/{deployment_id}

Runpod endpoint'ini kapatır ve deployment'ı silinmiş olarak işaretler.

**İstek:**

```bash
curl -s -X DELETE -H "Authorization: Bearer YOUR_API_KEY" \
  "http://127.0.0.1:8080/v1/deployments/dep_2026_xxxxx"
```

**Yanıt:** 204 No Content (body yok).

---

### 2.7 Internal Webhook – POST /internal/deployment-ready/{deployment_id}

Inference container tarafından model yüklendiğinde çağrılır. Status `ready` yapılır ve kullanıcı webhook'u tetiklenir.

**İstek (opsiyonel header: X-Visgate-Internal-Secret):**

```bash
curl -s -X POST "http://127.0.0.1:8080/internal/deployment-ready/dep_2026_a3686d66" \
  -H "Content-Type: application/json" \
  -d '{"status":"ready"}'
```

**Örnek yanıt (200 OK):**

```json
{
  "deployment_id": "dep_2026_a3686d66",
  "status": "ready",
  "webhook_delivered": false
}
```

`webhook_delivered`: Kullanıcı webhook URL'ine POST'un başarılı olup olmadığı.

---

## 3. Hata Yanıtları Özeti

| HTTP | Hata / Durum | Açıklama |
|------|----------------|----------|
| 200 | - | GET başarılı |
| 202 | - | POST /v1/deployments kabul edildi |
| 204 | - | DELETE başarılı |
| 401 | UnauthorizedError | Eksik veya geçersiz Bearer token |
| 404 | DeploymentNotFoundError | deployment_id bulunamadı veya erişim yok |
| 429 | RateLimitError | Dakikada 100 istek limiti aşıldı |
| 503 | - | Readiness: Firestore bağlantı hatası |

---

## 4. Hızlı Test Scripti

Tüm endpoint'leri sırayla denemek için:

```bash
#!/bin/bash
BASE="http://127.0.0.1:8080"
KEY="Bearer test-api-key-12345"

echo "1. Health:" && curl -s "$BASE/health"
echo "\n2. Root:" && curl -s "$BASE/"
echo "\n3. Readiness:" && curl -s "$BASE/readiness"
echo "\n4. POST deployment:"
RESP=$(curl -s -X POST "$BASE/v1/deployments" -H "Content-Type: application/json" -H "Authorization: $KEY" \
  -d '{"hf_model_id":"stabilityai/sdxl-turbo","user_runpod_key":"rpa_test","user_webhook_url":"https://example.com/hook","gpu_tier":"A40"}')
echo "$RESP"
DEP_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('deployment_id',''))")
echo "\n5. GET $DEP_ID:" && curl -s -H "Authorization: $KEY" "$BASE/v1/deployments/$DEP_ID"
echo "\n6. 401 test:" && curl -s -o /dev/null -w "%{http_code}" "$BASE/v1/deployments/$DEP_ID"
echo "\n7. 404 test:" && curl -s -o /dev/null -w "%{http_code}" -H "Authorization: $KEY" "$BASE/v1/deployments/dep_fake_123"
```

---

## 5. OpenAPI ve Otomatik Test

- Swagger UI: http://127.0.0.1:8080/docs  
- ReDoc: http://127.0.0.1:8080/redoc  
- OpenAPI JSON: http://127.0.0.1:8080/openapi.json  

Birim ve entegrasyon testleri:

```bash
cd deployment-orchestrator
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v --tb=short
```

---

*Son test tarihi: 2026-02-14. Örnek yanıtlar lokalde çalışan servisten alınmıştır.*
