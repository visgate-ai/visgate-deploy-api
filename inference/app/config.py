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

# Device
DEVICE: str = "cuda"  # Runpod provides GPU

# Default inference
DEFAULT_NUM_INFERENCE_STEPS: int = 28
DEFAULT_GUIDANCE_SCALE: float = 3.5
