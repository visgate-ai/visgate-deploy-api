"""Inference worker configuration from environment."""

import os
from typing import Optional


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key, default)
    return v.strip() if v else default


# Required: set by Runpod / orchestrator
# Fallback to sd-turbo if template doesn't pass HF_MODEL_ID properly
HF_MODEL_ID: str = get_env("HF_MODEL_ID") or "stabilityai/sd-turbo"
HF_TOKEN: Optional[str] = get_env("HF_TOKEN")

# When ready, POST to this URL (orchestrator internal webhook)
VISGATE_WEBHOOK: Optional[str] = get_env("VISGATE_WEBHOOK")
VISGATE_LOG_TUNNEL: Optional[str] = get_env("VISGATE_LOG_TUNNEL")
VISGATE_INTERNAL_SECRET: Optional[str] = get_env("VISGATE_INTERNAL_SECRET")
VISGATE_DEPLOYMENT_ID: Optional[str] = get_env("VISGATE_DEPLOYMENT_ID")

# Cleanup/watchdog
CLEANUP_IDLE_TIMEOUT_SECONDS: int = int(get_env("CLEANUP_IDLE_TIMEOUT_SECONDS", "900") or "900")
CLEANUP_FAILURE_THRESHOLD: int = int(get_env("CLEANUP_FAILURE_THRESHOLD", "3") or "3")

# Device
DEVICE: str = "cuda"  # Runpod provides GPU

# Default inference
DEFAULT_NUM_INFERENCE_STEPS: int = 28
DEFAULT_GUIDANCE_SCALE: float = 3.5

# Optional output delivery
OUTPUT_S3_URL: Optional[str] = get_env("OUTPUT_S3_URL")
CDN_BASE_URL: Optional[str] = get_env("CDN_BASE_URL")
RETURN_BASE64: str = get_env("RETURN_BASE64", "true") or "true"
