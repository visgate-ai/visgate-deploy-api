# Visgate Deploy API

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Deploy API CI](https://github.com/visgate-ai/visgate-deploy-api/actions/workflows/deploy.yaml/badge.svg)](https://github.com/visgate-ai/visgate-deploy-api/actions/workflows/deploy.yaml)
[![Inference Docker](https://github.com/visgate-ai/visgate-deploy-api/actions/workflows/inference.yaml/badge.svg)](https://github.com/visgate-ai/visgate-deploy-api/actions/workflows/inference.yaml)
[![Docker Hub](https://img.shields.io/docker/v/visgateai/inference?label=visgateai%2Finference)](https://hub.docker.com/r/visgateai/inference)

An open-source orchestration API that deploys Hugging Face diffusion models to [RunPod](https://runpod.io) serverless endpoints — validated, GPU-selected, cached, and webhook-notified.

```
POST /v1/deployments  →  HF validation  →  GPU selection  →  RunPod endpoint
                                                               ↓
       User webhook  ←  status=ready  ←  model loaded  ←  inference worker
```

## What it does

- Accepts a Hugging Face model ID + your RunPod API key
- Validates the model exists and calculates minimum GPU memory (dtype-aware byte estimation)
- Picks the cheapest available GPU that fits the model without OOM
- Creates a RunPod serverless endpoint and waits for the worker to be ready
- Notifies you via webhook when ready; you poll `/v1/deployments/{id}` in the meantime
- Optional S3/R2 model cache: first run downloads from HF, subsequent cold starts load from disk in seconds

## Architecture

```
┌────────────┐   POST /v1/deployments   ┌─────────────────────────────────┐
│   Client   │ ───────────────────────▶ │         FastAPI (Cloud Run)      │
│            │                          │                                  │
│            │ ◀─────────────────────── │  1. HF model validation         │
│  202 +     │    deployment_id          │  2. GPU selection (cost-optimal) │
│  dep_id    │                          │  3. RunPod endpoint creation     │
│            │                          │  4. Readiness monitoring         │
│            │   webhook: status=ready  │  5. User webhook notification    │
│            │ ◀─────────────────────── │                                  │
└────────────┘                          └─────────────────────────────────┘
                                                        │
                                           ┌────────────▼────────────┐
                                           │   RunPod Worker          │
                                           │   (diffusers pipeline)   │
                                           │   + S3 model cache       │
                                           └──────────────────────────┘
```

## Quick Start

### Hosted API (no setup needed)

A production instance is already running — all you need is your [RunPod API key](https://www.runpod.io/console/user/settings).

```bash
API_BASE="https://visgate-deploy-api-wxup7pxrsa-uc.a.run.app"

# 1. Health check
curl "$API_BASE/health"

# 2. Deploy a model (uses your RunPod key — no account registration needed)
curl -X POST "$API_BASE/v1/deployments" \
  -H "Authorization: Bearer <YOUR_RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_id": "stabilityai/sd-turbo",
    "gpu_tier": "A10",
    "user_webhook_url": "https://your-app.com/webhook"
  }'
# → {"deployment_id": "dep_2026_abc123", "status": "validating"}

# 3. Poll status
curl -H "Authorization: Bearer <YOUR_RUNPOD_API_KEY>" \
  "$API_BASE/v1/deployments/dep_2026_abc123"
# → {"status": "ready", "endpoint_url": "https://api.runpod.ai/v2/.../run"}

# 4. Delete when done (stops RunPod billing)
curl -X DELETE -H "Authorization: Bearer <YOUR_RUNPOD_API_KEY>" \
  "$API_BASE/v1/deployments/dep_2026_abc123"
```

> **How auth works:** Your RunPod API key is the auth token — no separate registration.
> The API creates RunPod endpoints under your account and charges go directly to your RunPod balance.

### Self-hosted (local, no GCP needed)

```bash
git clone https://github.com/visgate-ai/visgate-deploy-api
cd visgate-deploy-api/deploy-api

# Configure
cp .env.example .env
# Minimum required for local dev:
#   RUNPOD_TEMPLATE_ID=your-template-id
# Leave GCP_PROJECT_ID empty — in-memory storage is used automatically.

pip install -r requirements.txt
uvicorn src.main:app --reload
# → API running at http://localhost:8000

# Or with Docker
docker build -t visgate-deploy-api .
docker run -p 8080:8080 --env-file .env visgate-deploy-api
```

### Deploy to Cloud Run

```bash
# First-time setup (creates Firestore, enables APIs)
cd deploy-api && ./deploy.sh

# Or use the GitHub Actions workflow (.github/workflows/deploy.yaml)
# Set GCP_CREDENTIALS and GCP_PROJECT_ID secrets in your repo settings
```

## GPU Selection Algorithm

Models are matched to the cheapest GPU that fits:

```
HF model  →  safetensors.parameters {BF16: N, F32: M, ...}
          →  Σ (param_count × bytes_per_dtype)  =  weight_bytes
          →  weight_bytes × 1.35 headroom        =  min_vram_gb
          →  snap to real tier: 6/8/10/12/16/24/28/40/48/80 GB
          →  pick cheapest GPU in DEFAULT_GPU_REGISTRY with vram ≥ min_vram_gb
          →  capacity error → try next candidate (backoff loop)
```

18+ popular models (FLUX, SDXL, SD 1.x/2.x/3.x, PixArt, Kandinsky, Wan, CogVideoX) are pre-registered in [`model_specs_registry.py`](deploy-api/src/models/model_specs_registry.py) with hand-tuned values. Unknown models fall back to the byte-estimation above.

Default GPU registry (cheapest → most expensive):

| GPU | VRAM | RunPod ID |
|-----|------|-----------|
| NVIDIA A16 | 16 GB | `AMPERE_16` |
| NVIDIA A10 / A30 | 24 GB | `AMPERE_24` |
| NVIDIA L40 / RTX 4090 | 24 GB | `ADA_24` |
| NVIDIA A40 | 48 GB | `AMPERE_48` |
| NVIDIA L40S | 48 GB | `ADA_48_PRO` |
| NVIDIA A100 | 80 GB | `AMPERE_80` |
| NVIDIA H100 | 80 GB | `ADA_80_PRO` |

## S3 Model Cache

Cold starts are the main latency cost. With a persistent volume + S3 cache, the _second_ cold start loads from disk in 2–5s instead of 40–80s.

```bash
# Private cache: your own S3/R2/MinIO bucket
POST /v1/deployments
{
  "hf_model_id": "...",
  "cache_scope": "private",
  "user_s3_url": "s3://my-bucket/models/sd-turbo",
  "user_aws_access_key_id": "...",
  "user_aws_secret_access_key": "...",
  "user_aws_endpoint_url": "https://..."   # for R2 / MinIO
}

# Shared cache: platform-managed (read-only for users, allowlist-only)
{
  "cache_scope": "shared"
}
```

Cache sync uses [s5cmd](https://github.com/peak/s5cmd) (50 workers, 50 MB chunks) for fast parallel downloads.

## Cost Model

| What costs money | When |
|-----------------|------|
| RunPod GPU time | Only while a worker is running a job |
| RunPod idle time | `workers_min=1` keeps a worker warm (costs ~$0.35–0.80/hr per endpoint) |
| S3 storage | Per GB stored |
| S3 requests | Per GET/PUT (cheap, but watch egress on large models) |

Default settings (`workers_min=0`, `idle_timeout=120s`) mean **no idle GPU cost** — you pay only for actual inference. First cold start takes 40–80s; subsequent cold starts with volume cache take 2–5s.

Set `RUNPOD_WORKERS_MIN=1` for production deployments where sub-second startup latency matters.

## Configuration

Key environment variables:

```bash
# Required (leave empty for local dev — in-memory storage used automatically)
GCP_PROJECT_ID=your-project

# RunPod
RUNPOD_TEMPLATE_ID=xxxxxxxx          # create in RunPod console → Serverless → Templates
DOCKER_IMAGE=your-org/inference:latest

# Cost tuning
RUNPOD_WORKERS_MIN=0                 # 0 = no idle cost, 1 = always warm
RUNPOD_IDLE_TIMEOUT_SECONDS=120      # how long a worker stays up after last job
RUNPOD_WORKERS_MAX=3

# Optional S3 shared cache
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_ENDPOINT_URL=https://...         # R2 / MinIO endpoint
S3_MODEL_URL=s3://bucket/models

# Shared cache policy
SHARED_CACHE_ALLOWED_MODELS=stabilityai/sd-turbo,black-forest-labs/FLUX.1-schnell,...
SHARED_CACHE_REJECT_UNLISTED=true
```

Full reference: [`deploy-api/src/core/config.py`](deploy-api/src/core/config.py)

## Repository Structure

```
deploy-api/          # FastAPI orchestration service (deploy to Cloud Run)
  src/
    api/routes/      # HTTP endpoints
    services/        # HF validation, GPU selection, RunPod, deployment logic
    models/          # Pydantic schemas, GPU/model registries
    core/            # Config, logging, telemetry, errors
  tests/             # Unit + integration tests (pytest)
  Dockerfile
  cloudbuild.yaml

inference/           # Runpod serverless worker (push to Docker Hub)
  app/
    worker.py        # Job handler + background model loading
    loader.py        # S3 sync + pipeline loading
  pipelines/         # FLUX, SDXL, base pipeline classes

scripts/             # CLI utilities
  deploy_via_api.py          # example: deploy + wait + infer
  prod_api_smoke.py          # health + create + delete smoke test
  cleanup_runpod_endpoints.py  # bulk-delete stale RunPod endpoints
  webhook_receiver.py        # local webhook listener for testing
```

## Development

```bash
cd deploy-api
pip install -r requirements.txt
python -m pytest tests/ -q   # 39 tests, no cloud credentials needed
```

The test suite mocks Firestore and RunPod. For local API development leave `GCP_PROJECT_ID` empty — in-memory storage activates automatically.

## Inference Worker Setup

1. Build and push the inference image:
   ```bash
   cd inference
   docker build -t visgateai/inference:latest .
   docker push visgateai/inference:latest
   ```
   Or fork this repo — GitHub Actions builds and pushes `visgateai/inference:latest` automatically on every push to `inference/`.

2. Create a RunPod Serverless Template pointing to that image (or use the helper script):
   ```bash
   RUNPOD_API_KEY=xxx python deploy-api/scripts/create_runpod_template.py
   ```

3. Set `RUNPOD_TEMPLATE_ID` in your deploy-api environment.

## Contributing

Pull requests welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

Areas where contributions are most useful:

- Additional pipeline types in `inference/pipelines/` (ControlNet, img2img, video models)
- More models in `deploy-api/src/models/model_specs_registry.py`
- Alternative GPU provider implementations in `deploy-api/src/services/` (Vast.ai, Lambda, etc.)
- Quantization support (GGUF, GPTQ) in the inference worker
- Multi-cloud storage backends (GCS, Azure Blob)

## License

MIT — see [LICENSE](LICENSE).
