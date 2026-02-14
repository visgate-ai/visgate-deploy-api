#!/usr/bin/env bash
# Deploy visgate-deploy-api to GCP Cloud Run.
# Usage: ./deploy.sh [REGION]

set -euo pipefail

REGION="${1:-us-central1}"
PROJECT_ID="${GCP_PROJECT_ID:-visgate}"
SERVICE_NAME="visgate-deploy-api"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "Project: ${PROJECT_ID} Region: ${REGION}"

# Enable APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com firestore.googleapis.com secretmanager.googleapis.com --project="${PROJECT_ID}"

# Build and push (from repo root; Dockerfile context is deploy-api)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
docker build -t "${IMAGE}" -f deploy-api/Dockerfile deploy-api
docker push "${IMAGE}"

# Deploy Cloud Run
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID}" \
  --set-env-vars "FIRESTORE_COLLECTION_DEPLOYMENTS=visgate_deploy_api_deployments" \
  --set-env-vars "FIRESTORE_COLLECTION_LOGS=visgate_deploy_api_logs" \
  --set-env-vars "LOG_LEVEL=INFO" \
  --port 8080 \
  --project "${PROJECT_ID}"

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --project "${PROJECT_ID}" --format='value(status.url)' 2>/dev/null || true)
echo "Deployed. API: ${SERVICE_URL:-unknown}"
