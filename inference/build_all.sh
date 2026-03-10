#!/usr/bin/env bash
# Build all three inference images with VISGATE_R2_* fix
# Order: image -> audio (parallel), then video (needs image as base)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Starting parallel build: inference-image + inference-audio ==="
START=$(date +%s)

docker build --platform linux/amd64 -f Dockerfile -t visgateai/inference-image:latest . \
  2>&1 | sed 's/^/[image] /' &
PID_IMAGE=$!

docker build --platform linux/amd64 -f Dockerfile.audio -t visgateai/inference-audio:latest . \
  2>&1 | sed 's/^/[audio] /' &
PID_AUDIO=$!

echo "inference-image pid=$PID_IMAGE"
echo "inference-audio pid=$PID_AUDIO"

# Wait for inference-image (needed as base for video)
wait $PID_IMAGE
IMAGE_RC=$?
IMAGE_END=$(date +%s)
echo "=== inference-image build DONE rc=$IMAGE_RC elapsed=$((IMAGE_END - START))s ==="

if [ $IMAGE_RC -ne 0 ]; then
  echo "ERROR: inference-image build failed"
  kill $PID_AUDIO 2>/dev/null || true
  exit 1
fi

echo "=== Pushing inference-image ==="
docker push visgateai/inference-image:latest 2>&1 | sed 's/^/[image-push] /'
PUSH_IMAGE_RC=$?
echo "=== inference-image push DONE rc=$PUSH_IMAGE_RC ==="

echo "=== Building inference-video (inherits from inference-image) ==="
docker build --platform linux/amd64 -f Dockerfile.video -t visgateai/inference-video:latest . \
  2>&1 | sed 's/^/[video] /'
VIDEO_BUILD_RC=$?
echo "=== inference-video build DONE rc=$VIDEO_BUILD_RC ==="
docker push visgateai/inference-video:latest 2>&1 | sed 's/^/[video-push] /'
VIDEO_PUSH_RC=$?
VIDEO_END=$(date +%s)
echo "=== inference-video push DONE rc=$VIDEO_PUSH_RC elapsed=$((VIDEO_END - START))s ==="

# Wait for audio build
wait $PID_AUDIO
AUDIO_RC=$?
AUDIO_END=$(date +%s)
echo "=== inference-audio build DONE rc=$AUDIO_RC elapsed=$((AUDIO_END - START))s ==="

echo "=== Pushing inference-audio ==="
docker push visgateai/inference-audio:latest 2>&1 | sed 's/^/[audio-push] /'
PUSH_AUDIO_RC=$?
echo "=== inference-audio push DONE rc=$PUSH_AUDIO_RC ==="

TOTAL_END=$(date +%s)
echo ""
echo "=== ALL BUILDS COMPLETE ==="
echo "  inference-image rc=$IMAGE_RC push=$PUSH_IMAGE_RC"
echo "  inference-audio rc=$AUDIO_RC push=$PUSH_AUDIO_RC"
echo "  inference-video rc=$VIDEO_BUILD_RC push=$VIDEO_PUSH_RC"
echo "  total elapsed=$((TOTAL_END - START))s"
