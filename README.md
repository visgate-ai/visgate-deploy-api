# Visgate Deploy API

Deploy Hugging Face diffusion models to Runpod Serverless through a single API.

`visgate-deploy-api` is an open-source orchestration service that:
- accepts model deployment requests,
- creates and tracks Runpod endpoints,
- updates lifecycle state in Firestore,
- notifies your webhook when the endpoint is ready.

The orchestrator is designed for asynchronous model startup, so your app can stay responsive while pods warm up.

## What You Get

- Asynchronous deployment API with status polling and SSE stream support
- Worker phase tracking (`validating`, `creating_endpoint`, `loading_model`, `ready`, `failed`)
- Runpod endpoint lifecycle management (create/delete)
- Firestore-backed deployment state and logs
- Webhook callback when deployment is ready
- Production deployment path for GCP Cloud Run + GitHub Actions CI/CD

## Repository Layout

- `deploy-api/` - FastAPI orchestrator (Cloud Run service)
- `inference/` - Runpod worker image that loads and serves the model
- `scripts/` - E2E tests and Runpod maintenance helpers

## Implementation Roadmap (for Agents)

- Canonical architecture migration plan for future AI agents:
  - `.cursor/plans/visgate-predictive-stateless-roadmap.md`

## Hosted Endpoint

Current hosted endpoint:

- `https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app`

Use your Runpod key in `Authorization: Bearer <RUNPOD_API_KEY>`.

## Quick Start (Hosted)

1) Create deployment

```bash
curl -X POST "https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app/v1/deployments" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_id": "stabilityai/sd-turbo",
    "gpu_tier": "A10",
    "hf_token": "hf_xxx_optional_for_gated_models",
    "user_webhook_url": "https://your-app.com/webhooks/visgate",
    "cache_scope": "shared"
  }'
```

Cache options:
- `cache_scope: "shared"` uses platform S3 cache.
- `cache_scope: "private"` requires user S3 credentials and base URL.
- `cache_scope: "off"` disables cache (default).

2) Poll deployment status

```bash
curl -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  "https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app/v1/deployments/<deployment_id>"
```

3) Call Runpod inference once ready

```bash
curl -X POST "https://api.runpod.ai/v2/<endpoint_id>/runsync" \
  -H "Authorization: Bearer <YOUR_RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A cinematic view of Istanbul at golden hour",
      "num_inference_steps": 2
    }
  }'
```

## API Summary

- `POST /v1/deployments` - create deployment (returns `202`)
- `GET /v1/deployments/{deployment_id}` - get status, logs, endpoint URL
- `GET /v1/deployments/{deployment_id}/stream` - SSE live status stream
- `DELETE /v1/deployments/{deployment_id}` - delete Runpod endpoint and mark deleted
- `GET /health` - liveness
- `GET /metrics` - deployment and failure counters

## Self-Host on GCP

1) Prepare env and secrets
- create `deploy-api/.env` from `deploy-api/.env.example`
- set Secret Manager values (`RUNPOD_TEMPLATE_ID`, `INTERNAL_WEBHOOK_SECRET`)

2) Build and deploy orchestrator

```bash
cd deploy-api
./deploy_with_keys.sh us-central1
```

3) Build/push worker image for Runpod

```bash
cd inference
./build-and-push.sh
```

For deeper setup details, see:
- `deploy-api/README.md`
- `inference/README.md`

## End-to-End Validation

Use timed E2E flow:

```bash
API_BASE="https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app" \
RUNPOD="<runpod_key>" HF="<hf_token_optional>" \
python3 scripts/e2e_timed_istanbul.py
```

Recent real runs:

| Deployment | Model | Deploy -> Ready | Inference (runsync) | Runpod delay | Runpod execution |
|---|---|---:|---:|---:|---:|
| `dep_2026_95f8d6bf` | `crynux-network/sdxl-turbo` | 256.79s | 17.07s | 0.149s | 15.39s |
| `dep_2026_416968b0` | `unknown` | 72.12s | 8.89s | - | - |
| `dep_2026_24644278` | `unknown` | 37.61s | 7.55s | - | - |

Cache-scope and capacity test snapshot (prod):

| Scope | Deploy status | Deploy duration | Inference | Root cause / note |
|---|---|---:|---:|---|
| `off` | `timeout` | 1203.27s | - | Capacity / queue pressure during provisioning (`dep_2026_31b35de1`) |
| `shared` | `http_400` | - | - | Service config missing `S3_MODEL_URL` |
| `private` | `skipped` | - | - | Missing user S3 credentials in runtime env |

Optimization now in code: GPU provisioning uses cost-ordered multi-candidate fallback (model VRAM + tier compatible) and retries on capacity-style Runpod API errors, so requests can move to the next suitable free GPU automatically.

## Cleanup Helpers

- `python3 scripts/cleanup_runpod.py` - deletes `visgate-*` endpoints from your Runpod account

## Contributing

Contributions are welcome. Open an issue for bugs/features, then submit a PR with:
- clear description of behavior change,
- test evidence (local or Cloud Run),
- updated docs when API behavior changes.

## License

MIT. See `LICENSE`.
