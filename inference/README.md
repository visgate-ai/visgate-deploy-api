# Visgate HF Inference Worker (Runpod)

Docker image that loads any supported Hugging Face diffusion model and exposes a Runpod serverless endpoint. Used by the **deploy-api** service in this repo: when a deployment is created, Runpod runs this image with `HF_MODEL_ID` and `VISGATE_WEBHOOK`; after the model loads, the worker notifies the orchestrator and handles inference jobs. You send requests to the returned `endpoint_url` and get back artifact metadata, optional inline base64, or errors.

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
  "input": {
    "prompt": "A photo of a cat",
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "height": 1024,
    "width": 1024,
    "seed": 42,
    "input_image_r2": {
      "bucket_name": "platform-input-bucket",
      "endpoint_url": "https://<account>.r2.cloudflarestorage.com",
      "key": "inference/inputs/job_123/input_image_url-abc.png"
    }
  }
}
```

All fields except `input.prompt` are optional. In production, deploy-api stages media inputs into platform R2 and injects platform-managed output storage, so callers do not provide `s3Config`.

GitHub Actions publishes separate `latest` images for image, audio, and video worker profiles before the live smoke workflow runs.
The live smoke workflow now pre-caches the smoke models through the internal cache-task endpoint before provisioning deployments.
That avoids repeated cold-start timeouts on GitHub-hosted runners when RunPod has to fetch large model weights from Hugging Face first.
The cache upload path is derived from the configured platform model base URL, so the smoke job warms the same R2 bucket and prefix that production deployments read from.
Webhook validation uses a temporary `cloudflared` receiver so the smoke job can verify the actual callback payloads, while the harness still deletes prior smoke deployments to protect limited RunPod worker quota.

## Job output

Success:

```json
{
  "image_base64": "<base64-encoded PNG>",
  "artifact": {
    "bucket_name": "customer-output-bucket",
    "endpoint_url": "https://s3.example.com",
    "key": "visgate/jobs/1741431234_abcd.png",
    "url": "https://s3.example.com/customer-output-bucket/visgate/jobs/1741431234_abcd.png",
    "content_type": "image/png",
    "bytes": 182736
  },
  "execution_duration_ms": 4821,
  "model_id": "black-forest-labs/FLUX.1-schnell",
  "seed": 42,
  "height": 1024,
  "width": 1024
}
```

If `RETURN_BASE64=false`, the worker removes `image_base64` after a successful upload and returns artifact metadata only.

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

# 3) Output is in response: output.artifact, output.image_base64 (optional), etc.
```

### Sync: /runsync (wait for result in one call)

```bash
curl -s -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "A photo of a sunset"}}'
```

Response body contains `output` with `artifact`, `execution_duration_ms`, and optionally `image_base64`. Decode base64 only when you intentionally keep inline payloads enabled.

