# Visgate Deploy API — Agent Reference

Base URL: `https://visgate-deploy-api-wxup7pxrsa-ey.a.run.app`

## Authentication

All requests require a RunPod API key:
```
Authorization: Bearer rpa_...
```

---

## Endpoints

### 1. List supported models
```
GET /v1/models
```
Returns all supported models with VRAM requirements and whether they are already cached in R2 (faster cold start when `cached: true`).

```json
{
  "models": [
    { "model_id": "stabilityai/sdxl-turbo", "tasks": ["text2img"], "gpu_memory_gb": 10, "cached": true },
    { "model_id": "black-forest-labs/FLUX.1-schnell", "tasks": ["text2img"], "gpu_memory_gb": 16, "cached": false }
  ],
  "total": 12,
  "cache_enabled": true
}
```

---

### 2. Create deployment
```
POST /v1/deployments
Content-Type: application/json
```

```json
{
  "hf_model_id": "stabilityai/sdxl-turbo",
  "user_webhook_url": "https://your-app.com/webhook",
  "task": "text2img",
  "cache_scope": "shared",
  "hf_token": "hf_...",
  "gpu_tier": "A40"
}
```

| Field | Required | Notes |
|---|---|---|
| `hf_model_id` | ✅ | Must match a model from `GET /v1/models` (or any valid HF repo) |
| `user_webhook_url` | ✅ | Called with `POST` when deployment reaches `ready` or `failed` |
| `task` | — | `text2img` (default) · `image2img` · `text2video` |
| `cache_scope` | — | `off` (default) · `shared` (platform R2 cache, fast) · `private` (your own S3) |
| `hf_token` | — | Required for gated models |
| `gpu_tier` | — | Auto-selected from VRAM requirements if omitted |

**Response 202:**
```json
{
  "deployment_id": "dep_2026_abc123",
  "status": "accepted_cold",
  "path": "cold",
  "model_id": "stabilityai/sdxl-turbo",
  "estimated_ready_seconds": 180,
  "stream_url": "...",
  "webhook_url": "https://your-app.com/webhook",
  "created_at": "2026-03-07T10:00:00Z"
}
```

---

### 3. Poll deployment status
```
GET /v1/deployments/{deployment_id}
```

**Status progression:**
```
accepted_cold → validating → selecting_gpu → creating_endpoint → loading_model → ready
                                                                               ↘ failed
```

Poll every 8–10 seconds. Stop when status is `ready`, `failed`, or `webhook_failed`.

**Response (ready):**
```json
{
  "deployment_id": "dep_2026_abc123",
  "status": "ready",
  "endpoint_url": "https://api.runpod.ai/v2/{endpoint_id}/run",
  "gpu_allocated": "AMPERE 16GB, AMPERE 24GB, ADA 24GB",
  "model_vram_gb": 10,
  "estimated_remaining_seconds": 0,
  "logs": [...],
  "ready_at": "2026-03-07T10:01:00Z"
}
```

Use `endpoint_url` for inference requests directly against RunPod.

---

### 4. Delete deployment
```
DELETE /v1/deployments/{deployment_id}
```
Returns `204`. Tears down the RunPod endpoint and marks the deployment as deleted. **Always call this when done** to free the worker quota.

---

### 5. Health
```
GET /health     → 200 {"status": "ok"}
GET /readiness  → 200 when Firestore is reachable
```

---

## Typical agent workflow

```python
# 1. (optional) Pick a cached model for fastest cold start
models = GET /v1/models
model = next(m for m in models["models"] if m["cached"])

# 2. Create deployment
resp = POST /v1/deployments  {"hf_model_id": model["model_id"], "cache_scope": "shared", ...}
dep_id = resp["deployment_id"]

# 3. Poll until ready
while True:
    d = GET /v1/deployments/{dep_id}
    if d["status"] == "ready":
        endpoint_url = d["endpoint_url"]
        break
    if d["status"] in ("failed", "deleted"):
        raise RuntimeError(d["error"])
    sleep(8)

# 4. Run inference against RunPod endpoint
POST {endpoint_url}sync  {"input": {"prompt": "...", ...}}

# 5. Cleanup
DELETE /v1/deployments/{dep_id}
```

---

## Performance

| Path | Typical time to ready |
|---|---|
| `cache_scope=shared`, model `cached=true` | ~30–60s |
| `cache_scope=shared`, model `cached=false` | ~120–180s (first run caches it) |
| `cache_scope=off` | ~120–180s always |

---

## Errors

| HTTP | Body `detail` | Cause |
|---|---|---|
| `400` | `HuggingFaceModelNotFoundError` | Invalid or private model without token |
| `400` | `shared cache requires R2_MODEL_BASE_URL…` | Platform cache misconfigured |
| `401` | `Missing or invalid API key` | No / bad `Authorization` header |
| `429` | `Rate limit exceeded` | >100 req/min; back off, use `Retry-After` |
| `404` | `Deployment not found` | Wrong `deployment_id` |
| deployment `status=failed` | `error` field set | See `logs` array for details |
