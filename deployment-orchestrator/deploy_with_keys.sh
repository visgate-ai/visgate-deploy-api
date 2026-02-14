#!/usr/bin/env bash
# Deploy deployment-orchestrator to GCP Cloud Run and set up resources.
# Usage: ./deploy.sh [REGION]

set -euo pipefail

REGION="${1:-europe-west1}"
PROJECT_ID="${GCP_PROJECT_ID:-visgate}"
SERVICE_NAME="deployment-orchestrator"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "Project: ${PROJECT_ID} Region: ${REGION}"

# Ensure Docker can push to GCR
gcloud auth configure-docker gcr.io --quiet 2>/dev/null || true

# Enable APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com firestore.googleapis.com secretmanager.googleapis.com --project="${PROJECT_ID}"

# Build and push using Cloud Build (no local docker required)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
gcloud builds submit --tag "${IMAGE}" deployment-orchestrator

# Optional: RUNPOD_TEMPLATE_ID (from scripts/create_runpod_template.py), INTERNAL_WEBHOOK_BASE_URL (Cloud Run URL for container callback)
EXTRA_ENV=""
[[ -n "${RUNPOD_TEMPLATE_ID:-}" ]] && EXTRA_ENV="${EXTRA_ENV} --set-env-vars RUNPOD_TEMPLATE_ID=${RUNPOD_TEMPLATE_ID}"
[[ -n "${INTERNAL_WEBHOOK_BASE_URL:-}" ]] && EXTRA_ENV="${EXTRA_ENV} --set-env-vars INTERNAL_WEBHOOK_BASE_URL=${INTERNAL_WEBHOOK_BASE_URL}"

# Deploy Cloud Run
# Load .env.local from parent directory
if [ -f ../.env.local ]; then
  # remove spaces around equals just in case, though .env.local seems clean
  export $(grep -v '^#' ../.env.local | xargs)
fi

CLOUD_TASKS_QUEUE_PATH="projects/${PROJECT_ID}/locations/${REGION}/queues/visgate-orchestrator-queue"

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID}" \
  --set-env-vars "FIRESTORE_COLLECTION_DEPLOYMENTS=deployments" \
  --set-env-vars "FIRESTORE_COLLECTION_LOGS=deployment_logs" \
  --set-env-vars "FIRESTORE_COLLECTION_API_KEYS=api_keys" \
  --set-env-vars "CLOUD_TASKS_QUEUE_PATH=${CLOUD_TASKS_QUEUE_PATH}" \
  --set-env-vars "GCP_LOCATION=${REGION}" \
  --set-env-vars "LOG_LEVEL=INFO" \
  ${EXTRA_ENV} \
  --port 8080 \
  --project "${PROJECT_ID}"

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --project "${PROJECT_ID}" --format='value(status.url)' 2>/dev/null || true)
echo "Deployed. API: ${SERVICE_URL:-unknown}"
if [[ -z "${INTERNAL_WEBHOOK_BASE_URL:-}" && -n "${SERVICE_URL}" ]]; then
  echo "To let Runpod containers callback: Cloud Run Console → deployment-orchestrator → Edit → Variables → INTERNAL_WEBHOOK_BASE_URL=${SERVICE_URL}"
fi
