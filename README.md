# Visgate Deploy API

**Visgate Deploy API** is an open-source tool that allows you to run Hugging Face diffusion models (Flux, SDXL, etc.) on **Runpod Serverless** infrastructure with ease.

With our **Hosted Orchestrator API**, you can deploy models without setting up your own server or cloud infrastructure. All you need is your Runpod API Key.

---

## 🚀 How It Works

1.  **Request:** Send the desired model ID and your Runpod key to our API endpoint (`POST /v1/deployments`).
2.  **Orchestration:** Our system configures the Runpod template and endpoint in your account and prepares the model.
3.  **Webhook:** Once the model is loaded and ready, we send a webhook to your specified URL.
4.  **Inference:** Use the provided endpoint URL from the webhook to generate images.

---

## 🔌 API Usage (Hosted Service)

**Base URL:** `https://api.visgate.io` (Example URL - Update after deployment)

### 1. Create Deployment

**POST** `/v1/deployments`
**Header:** `Authorization: Bearer <VISGATE_API_KEY>`

```json
{
  "hf_model_id": "black-forest-labs/FLUX.1-schnell",
  "user_runpod_key": "YOUR_RUNPOD_API_KEY",
  "user_webhook_url": "https://your-server.com/webhook",
  "gpu_tier": "3090" // Optional (3090, A40, A100, etc.)
}
```

### 2. Webhook Response (Ready)

When the model is ready, your `user_webhook_url` will receive this JSON:

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

### 3. Generate Image (Inference)

Send a request to the URL received in the webhook:

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

## 🛠️ Self-Hosting

If you wish to host this service yourself on your own GCP project:

1.  **deployment-orchestrator:** Deploy to GCP Cloud Run. Requires Firestore, Cloud Tasks, and Secret Manager.
2.  **inference:** Build the Docker image and push to Docker Hub.
3.  See [deployment-orchestrator/README.md](deployment-orchestrator/README.md) for detailed setup instructions.

---

## 📜 License

MIT License.
