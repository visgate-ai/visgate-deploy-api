#!/usr/bin/env bash
# Deploy visgate-deploy-api to GCP Cloud Run and set up resources.
# Usage: ./deploy.sh [REGION]

set -euo pipefail

REGION="${1:-europe-west3}"
PROJECT_ID="${GCP_PROJECT_ID:-visgate}"
SERVICE_NAME="visgate-deploy-api"
IMAGE="europe-west3-docker.pkg.dev/${PROJECT_ID}/visgate/${SERVICE_NAME}:latest"

echo "Project: ${PROJECT_ID} Region: ${REGION}"

# Ensure Docker can push to Artifact Registry
gcloud auth configure-docker europe-west3-docker.pkg.dev --quiet 2>/dev/null || true

# Enable APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com firestore.googleapis.com secretmanager.googleapis.com --project="${PROJECT_ID}"

# Build and push using Cloud Build (no local docker required)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
gcloud builds submit --tag "${IMAGE}" deploy-api

# Load only specific optional vars from files without overriding IMAGE/SERVICE_NAME.
load_optional_var() {
  local file="$1"
  local key="$2"
  if [[ -f "${file}" && -z "${!key:-}" ]]; then
    local value
    value="$(awk -F= -v k="${key}" '$1==k {sub(/^[^=]*=/,""); print; exit}' "${file}")"
    if [[ -n "${value}" ]]; then
      export "${key}=${value}"
    fi
  fi
}

for f in "${REPO_ROOT}/deploy-api/.env" "${REPO_ROOT}/.env.local"; do
  load_optional_var "${f}" "RUNPOD_TEMPLATE_ID"
  load_optional_var "${f}" "INTERNAL_WEBHOOK_BASE_URL"
done

CLOUD_TASKS_QUEUE_PATH="projects/${PROJECT_ID}/locations/${REGION}/queues/visgate-orchestrator-queue"
EXISTING_SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --project "${PROJECT_ID}" --format='value(status.url)' 2>/dev/null || true)"
INTERNAL_WEBHOOK_BASE_URL_VALUE="${INTERNAL_WEBHOOK_BASE_URL:-${EXISTING_SERVICE_URL:-}}"

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --update-env-vars "GCP_PROJECT_ID=${PROJECT_ID}" \
  --update-env-vars "FIRESTORE_COLLECTION_DEPLOYMENTS=visgate_deploy_api_deployments" \
  --update-env-vars "FIRESTORE_COLLECTION_LOGS=visgate_deploy_api_logs" \
  --update-env-vars "FIRESTORE_COLLECTION_API_KEYS=visgate_deploy_api_api_keys" \
  --update-env-vars "FIRESTORE_COLLECTION_GPU_REGISTRY=visgate_deploy_api_gpu_registry" \
  --update-env-vars "FIRESTORE_COLLECTION_GPU_TIERS=visgate_deploy_api_gpu_tiers" \
  --update-env-vars "CLOUD_TASKS_QUEUE_PATH=${CLOUD_TASKS_QUEUE_PATH}" \
  --update-env-vars "GCP_LOCATION=${REGION}" \
  --update-env-vars "LOG_LEVEL=INFO" \
  --update-env-vars "DOCKER_IMAGE=visgateai/inference:latest" \
  --update-env-vars "DOCKER_IMAGE_IMAGE=visgateai/inference-image:latest" \
  --update-env-vars "DOCKER_IMAGE_AUDIO=visgateai/inference-audio:latest" \
  --update-env-vars "DOCKER_IMAGE_VIDEO=visgateai/inference-video:latest" \
  --update-env-vars "RUNPOD_WORKERS_MAX=1" \
  --update-env-vars "RUNPOD_WORKERS_MIN=0" \
  --update-env-vars "RUNPOD_WORKERS_MIN_VIDEO=1" \
  --update-env-vars "RUNPOD_EXECUTION_TIMEOUT_MS=300000" \
  --update-env-vars "RUNPOD_EXECUTION_TIMEOUT_MS_VIDEO=900000" \
  --update-env-vars "INTERNAL_WEBHOOK_BASE_URL=${INTERNAL_WEBHOOK_BASE_URL_VALUE}" \
  --remove-env-vars "RUNPOD_TEMPLATE_ID_IMAGE,RUNPOD_TEMPLATE_ID_AUDIO,RUNPOD_TEMPLATE_ID_VIDEO" \
  --port 8080 \
  --project "${PROJECT_ID}"

# Ensure newly created revision receives traffic.
gcloud run services update-traffic "${SERVICE_NAME}" \
  --to-latest \
  --region "${REGION}" \
  --project "${PROJECT_ID}"

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --project "${PROJECT_ID}" --format='value(status.url)' 2>/dev/null || true)
echo "Deployed. API: ${SERVICE_URL:-unknown}"

