#!/usr/bin/env python3
"""
Create a Runpod serverless template that uses our inference Docker image.
Run from repo root or deployment-orchestrator with RUNPOD and optionally
DOCKER_IMAGE/IMAGE set (or in ../.env.local). Prints RUNPOD_TEMPLATE_ID for .env.
"""
import asyncio
import os
import sys

# Load .env.local from repo root if present
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_env_local = os.path.join(_root, ".env.local")
if os.path.isfile(_env_local):
    with open(_env_local) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if key and key not in os.environ:
                    os.environ[key] = val

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.runpod import create_serverless_template


TEMPLATE_NAME = "visgate-inference"
DEFAULT_IMAGE = os.environ.get("DOCKER_IMAGE") or os.environ.get("IMAGE") or "visgateai/inference:latest"


async def main() -> None:
    api_key = os.environ.get("RUNPOD", "").strip()
    if not api_key:
        print("Set RUNPOD to your Runpod API key (e.g. in .env.local).", file=sys.stderr)
        sys.exit(1)

    image = DEFAULT_IMAGE
    if ":" not in image:
        image = f"{image}:latest"
    print(f"Using image: {image}", file=sys.stderr)

    try:
        result = await create_serverless_template(
            api_key=api_key,
            name=TEMPLATE_NAME,
            image_name=image,
            container_disk_in_gb=25,
        )
    except Exception as e:
        if "unique" in str(e).lower() or "already" in str(e).lower() or "duplicate" in str(e).lower():
            print("Template with this name may already exist. Create one in Runpod Console (Serverless → Templates) with this image and set RUNPOD_TEMPLATE_ID in .env", file=sys.stderr)
        raise

    tid = result.get("id")
    print(f"Created template: {tid} ({result.get('imageName')})", file=sys.stderr)
    print(f"RUNPOD_TEMPLATE_ID={tid}")


if __name__ == "__main__":
    asyncio.run(main())
