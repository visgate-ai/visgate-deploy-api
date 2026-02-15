# Visgate HF Inference Worker (Runpod)

Docker image that loads any supported Hugging Face diffusion model and exposes a Runpod serverless endpoint. Used by the **deploy-api** service in this repo: when a deployment is created, Runpod runs this image with `HF_MODEL_ID` and `VISGATE_WEBHOOK`; after the model loads, the worker notifies the orchestrator and handles inference jobs. You send requests to the returned `endpoint_url` and get back images (base64) or errors.

## Supported models (modular)

- **Flux**: `black-forest-labs/FLUX.1-schnell`, `black-forest-labs/FLUX.1-dev`
- **SDXL**: `stabilityai/sdxl-turbo`, `stabilityai/stable-diffusion-xl-base-1.0`, etc.
- **Generic**: Any diffusers `AutoPipelineForText2Image`-compatible model

New model families can be added by adding a pipeline class in `pipelines/` and registering it in `pipelines/registry.py`.

## Environment variables (set by Runpod / orchestrator)

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_MODEL_ID` | Yes | Hugging Face model ID (e.g. `black-forest-labs/FLUX.1-schnell`) |
| `HF_TOKEN` | No | HF token for gated/private models |
| `VISGATE_WEBHOOK` | No | URL to POST when model is ready (orchestrator callback) |
| `DEVICE` | No | Default `cuda` |

## Job input (Runpod request body)

```json
{
  "prompt": "A photo of a cat",
  "num_inference_steps": 28,
  "guidance_scale": 3.5,
  "height": 1024,
  "width": 1024,
  "seed": 42
}
```

All fields except `prompt` are optional.

## Job output

Success:

```json
{
  "image_base64": "<base64-encoded PNG>",
  "model_id": "black-forest-labs/FLUX.1-schnell",
  "seed": 42,
  "height": 1024,
  "width": 1024
}
```

Error:

```json
{
  "error": "Missing or empty 'prompt' in input",
  "model_id": "..."
}
```

## Build and push to Docker Hub

Image is built from a PyTorch base image (no large torch download in build).

```bash
cd inference
./build-and-push.sh
```

Or manually:

```bash
docker build -t visgateai/inference:latest .
docker tag visgateai/inference:latest visgateai/inference:$(date +%Y%m%d)
# Login from a real terminal (PAT from .env.local or Docker Hub):
#   echo YOUR_PAT | docker login -u visgateai --password-stdin docker.io
docker login
docker push visgateai/inference:latest
docker push visgateai/inference:$(date +%Y%m%d)
```

Optional: in repo root `.env.local` set `DOCKER_HUB_PAT=your_token` so `./build-and-push.sh` can log in automatically. Replace `visgateai` with your Docker Hub username if needed.

## Local test (no GPU required for import check)

```bash
cd inference
pip install -r requirements.txt
export HF_MODEL_ID=stabilityai/sdxl-turbo
# Optional: VISGATE_WEBHOOK=http://host.docker.internal:8080/internal/deployment-ready/test
python3 -m app.worker
# In another terminal, or use test_input.json:
# python3 -m app.worker --test_input '{"input":{"prompt":"a cat"}}'
```

With Runpod CLI or a local test input file:

```bash
echo '{"input":{"prompt":"a red apple"}}' > test_input.json
python3 -m app.worker
```

## Runpod template

1. In Runpod Console → Serverless → Templates, create a new template.
2. Docker image: `visgateai/inference:latest` (or your Docker Hub image).
3. Container disk: e.g. 20 GB (for model cache).
4. Environment variables: set at endpoint level (orchestrator will pass `HF_MODEL_ID`, `HF_TOKEN`, `VISGATE_WEBHOOK` when creating the endpoint if Runpod supports per-endpoint env; otherwise bake into template or use a wrapper).

## Sending requests to the Runpod endpoint

After the orchestrator returns `endpoint_url` (e.g. `https://api.runpod.ai/v2/ENDPOINT_ID/run`), send inference requests and get results.

### Async: /run then poll /status

```bash
# 1) Start job
RESP=$(curl -s -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "A photo of a sunset", "num_inference_steps": 28}}')
JOB_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")
echo "Job ID: $JOB_ID"

# 2) Poll until status is COMPLETED
while true; do
  STATUS=$(curl -s -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
    "https://api.runpod.ai/v2/ENDPOINT_ID/status/$JOB_ID")
  echo "$STATUS" | python3 -m json.tool
  S=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
  [[ "$S" == "COMPLETED" || "$S" == "FAILED" ]] && break
  sleep 5
done

# 3) Output is in response: output.image_base64, etc.
```

### Sync: /runsync (wait for result in one call)

```bash
curl -s -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "A photo of a sunset"}}'
```

Response body contains `output` with `image_base64`, `model_id`, etc. Decode base64 to get the PNG image.
