#!/usr/bin/env bash
# Build and push visgateai/inference to Docker Hub.
# Usage: ./build-and-push.sh [tag_suffix]
# Optional: set DOCKER_HUB_PAT or put it in ../.env.local for login.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../.env.local" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/../.env.local"
  set +a
fi
# Use DOCKER_HUB_PAT or PAT from .env.local; username from DOCKER_HUB_USER or default visgateai
DOCKER_PAT="${DOCKER_HUB_PAT:-${PAT:-}}"
DOCKER_USER="${DOCKER_HUB_USER:-visgateai}"
if [[ -n "${DOCKER_PAT}" ]]; then
  echo "Logging in to Docker Hub as ${DOCKER_USER}..."
  printf '%s' "${DOCKER_PAT}" | docker login -u "${DOCKER_USER}" --password-stdin docker.io
fi

IMAGE="${IMAGE:-visgateai/inference}"
TAG_SUFFIX="${1:-}"
DATE_TAG=$(date +%Y%m%d)

echo "Building ${IMAGE}:latest ..."
docker build -t "${IMAGE}:latest" .

echo "Tagging ${IMAGE}:${DATE_TAG} ..."
docker tag "${IMAGE}:latest" "${IMAGE}:${DATE_TAG}"

if [[ -n "${TAG_SUFFIX}" ]]; then
  docker tag "${IMAGE}:latest" "${IMAGE}:${TAG_SUFFIX}"
  echo "Tagged ${IMAGE}:${TAG_SUFFIX}"
fi

echo "Pushing to Docker Hub ..."
docker push "${IMAGE}:latest"
docker push "${IMAGE}:${DATE_TAG}"
[[ -n "${TAG_SUFFIX}" ]] && docker push "${IMAGE}:${TAG_SUFFIX}"

echo "Done. Image: ${IMAGE}:latest and ${IMAGE}:${DATE_TAG}"
