#!/usr/bin/env bash
# Build and push the modality-specific RunPod worker images to Docker Hub.
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

TAG_SUFFIX="${1:-}"
DATE_TAG=$(date +%Y%m%d)

build_and_push() {
  local image="$1"
  local dockerfile="$2"
  local legacy_image="${3:-}"

  echo "Building ${image}:latest from ${dockerfile} ..."
  docker build -f "${dockerfile}" -t "${image}:latest" .

  echo "Tagging ${image}:${DATE_TAG} ..."
  docker tag "${image}:latest" "${image}:${DATE_TAG}"

  if [[ -n "${TAG_SUFFIX}" ]]; then
    docker tag "${image}:latest" "${image}:${TAG_SUFFIX}"
    echo "Tagged ${image}:${TAG_SUFFIX}"
  fi

  if [[ -n "${legacy_image}" ]]; then
    docker tag "${image}:latest" "${legacy_image}:latest"
    docker tag "${image}:latest" "${legacy_image}:${DATE_TAG}"
    [[ -n "${TAG_SUFFIX}" ]] && docker tag "${image}:latest" "${legacy_image}:${TAG_SUFFIX}"
  fi

  echo "Pushing ${image} tags ..."
  docker push "${image}:latest"
  docker push "${image}:${DATE_TAG}"
  [[ -n "${TAG_SUFFIX}" ]] && docker push "${image}:${TAG_SUFFIX}"

  if [[ -n "${legacy_image}" ]]; then
    echo "Pushing legacy alias ${legacy_image} tags ..."
    docker push "${legacy_image}:latest"
    docker push "${legacy_image}:${DATE_TAG}"
    [[ -n "${TAG_SUFFIX}" ]] && docker push "${legacy_image}:${TAG_SUFFIX}"
  fi
}

build_and_push "${DOCKER_USER}/inference-image" "Dockerfile" "${DOCKER_USER}/inference"
build_and_push "${DOCKER_USER}/inference-audio" "Dockerfile.audio"
build_and_push "${DOCKER_USER}/inference-video" "Dockerfile.video"

echo "Done. Pushed image, audio, and video worker images."
