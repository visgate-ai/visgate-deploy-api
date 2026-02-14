# Visgate Deploy API – User Guide and Test Examples

This document contains steps for running the service locally, testing endpoints, and real test output examples.

---

## 1. Running Locally

```bash
cd deployment-orchestrator
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export GCP_PROJECT_ID=visgate
PYTHONPATH=. .venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 8080
```

To use GCP Firestore:

```bash
gcloud auth application-default login
gcloud config set project visgate
```

API documentation in browser: **http://127.0.0.1:8080/docs**

---

## 2. Endpoint Tests and Example Responses

All examples are tested with `curl`. Base URL: `http://127.0.0.1:8080`.

### 2.1 Liveness – GET /health

Checks if the service is up (p99 <10ms target).

**Request:**

```bash
curl -s http://127.0.0.1:8080/health
```

**Example response (200 OK):**

```json
{"status":"ok"}
```

---

### 2.2 Root – GET /

Service info and docs link.

**Request:**

```bash
curl -s http://127.0.0.1:8080/
```

**Example response (200 OK):**

```json
{"service":"deployment-orchestrator","docs":"/docs"}
```

---

### 2.3 Readiness – GET /readiness

Checks Firestore connection. Returns 503 if failed.

**Request:**

```bash
curl -s http://127.0.0.1:8080/readiness
```

**Example response (200 OK):**

```json
{"status":"ready"}
```

**503 example (if Firestore is unreachable):**

```json
{"status":"unready","error":"..."}
```

---

### 2.4 Create Deployment – POST /v1/deployments

Starts a new deployment. The process continues in the background; returns 202 immediately.

**Flow:**

1. `model_name = get_models(provider=fal, "veo3")` → This info comes from **another endpoint** (not the orchestrator).
2. `hf_name = get_hf_name(model_name, provider)` → Resolved to HF model ID inside the orchestrator using `model_name` + `provider`.
3. `deploy_model(hf_name, gpu=auto) -> webhook` → Deployment started with the same POST; `user_webhook_url` is POSTed when ready.

**Option A – With HF model ID (Direct):**

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

**Option B – With model_name + provider:**

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

If `model_name` + `provider` are provided, they are converted to Hugging Face ID via `get_hf_name(model_name, provider)` (e.g. `fal` + `veo3` → `black-forest-labs/FLUX.1-schnell`). Provide either **hf_model_id** or **model_name** (+ optional **provider**), not both.

**Example response (202 Accepted):**

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

**401 – Unauthorized (Missing/Invalid Bearer):**

```json
{"error":"UnauthorizedError","message":"Missing or invalid API key","details":{}}
```

---

### 2.5 Deployment Status – GET /v1/deployments/{deployment_id}

Returns current status, logs, and error info for a specific deployment.

**Request:**

```bash
curl -s -H "Authorization: Bearer YOUR_API_KEY" \
  "http://127.0.0.1:8080/v1/deployments/dep_2026_a3686d66"
```

**Example response (200 OK) – while in progress (creating_endpoint):**

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

**Possible status values:** `validating` | `selecting_gpu` | `creating_endpoint` | `downloading_model` | `loading_model` | `ready` | `failed` | `webhook_failed` | `deleted`.

---

### 2.6 Delete Deployment – DELETE /v1/deployments/{deployment_id}

Terminates the Runpod endpoint and marks the deployment as deleted.

**Request:**

```bash
curl -s -X DELETE -H "Authorization: Bearer YOUR_API_KEY" \
  "http://127.0.0.1:8080/v1/deployments/dep_2026_xxxxx"
```

**Response:** 204 No Content.

---

### 2.7 Internal Webhook – POST /internal/deployment-ready/{deployment_id}

Called by the inference container when the model is loaded. Sets status to `ready` and triggers the user webhook.

**Request (Optional header: X-Visgate-Internal-Secret):**

```bash
curl -s -X POST "http://127.0.0.1:8080/internal/deployment-ready/dep_2026_a3686d66" \
  -H "Content-Type: application/json" \
  -d '{"status":"ready"}'
```

**Example response (200 OK):**

```json
{
  "deployment_id": "dep_2026_a3686d66",
  "status": "ready",
  "webhook_delivered": true
}
```

---

## 3. Error Responses Summary

| HTTP | Error / Status | Description |
|------|----------------|-------------|
| 200 | - | GET successful |
| 202 | - | POST /v1/deployments accepted |
| 204 | - | DELETE successful |
| 401 | UnauthorizedError | Missing or invalid Bearer token |
| 404 | DeploymentNotFoundError | deployment_id not found or access denied |
| 429 | RateLimitError | 100 requests per minute limit exceeded |
| 503 | - | Readiness: Firestore connection error |

---

## 4. Quick Test Script

To test all endpoints sequentially:

```bash
#!/bin/bash
BASE="http://127.0.0.1:8080"
KEY="Bearer test-api-key-12345"

echo "1. Health:" && curl -s "$BASE/health"
echo -e "\n2. Root:" && curl -s "$BASE/"
echo -e "\n3. Readiness:" && curl -s "$BASE/readiness"
echo -e "\n4. POST deployment:"
RESP=$(curl -s -X POST "$BASE/v1/deployments" -H "Content-Type: application/json" -H "Authorization: $KEY" \
  -d '{"hf_model_id":"stabilityai/sdxl-turbo","user_runpod_key":"rpa_test","user_webhook_url":"https://example.com/hook","gpu_tier":"A40"}')
echo "$RESP"
DEP_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('deployment_id',''))")
echo -e "\n5. GET $DEP_ID:" && curl -s -H "Authorization: $KEY" "$BASE/v1/deployments/$DEP_ID"
echo -e "\n6. 401 test:" && curl -s -o /dev/null -w "%{http_code}" "$BASE/v1/deployments/$DEP_ID"
echo -e "\n7. 404 test:" && curl -s -o /dev/null -w "%{http_code}" -H "Authorization: $KEY" "$BASE/v1/deployments/dep_fake_123"
```

---

*Last test date: 2026-02-15. Example responses taken from service running locally.*
