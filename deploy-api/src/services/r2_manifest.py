"""Cloudflare R2 model cache manifest reader/writer.

Reads and writes ``models/manifest.json`` from the configured R2 bucket.
Falls back gracefully whenever R2 is not configured or unreachable.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_MANIFEST_KEY = "models/manifest.json"


def fetch_cached_model_ids(
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    bucket: str = "visgate-models",
) -> set[str]:
    """Return the set of model IDs present in the R2 manifest.

    Returns an empty set on any error so failures are non-fatal.
    """
    try:
        import boto3  # type: ignore
        from botocore.exceptions import BotoCoreError, ClientError  # type: ignore

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )
        response = s3.get_object(Bucket=bucket, Key=_MANIFEST_KEY)
        manifest: dict = json.loads(response["Body"].read())
        model_ids: list[str] = manifest.get("models", [])
        logger.debug("R2 manifest loaded — %d cached models", len(model_ids))
        return set(model_ids)
    except ImportError:
        logger.warning("boto3 not installed; R2 manifest unavailable")
        return set()
    except (BotoCoreError, ClientError) as exc:
        logger.warning("R2 manifest read failed: %s", exc)
        return set()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error reading R2 manifest: %s", exc)
        return set()


def model_s3_url(base_url: str, model_id: str) -> str:
    """Return the per-model S3 URL for a given HuggingFace model ID.

    Example: base_url="s3://visgate-models/models", model_id="black-forest-labs/FLUX.1-dev"
    → "s3://visgate-models/models/black-forest-labs--FLUX.1-dev"
    """
    slug = model_id.replace("/", "--")
    return f"{base_url.rstrip('/')}/{slug}"


def add_model_to_manifest(
    model_id: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    bucket: str = "visgate-models",
) -> bool:
    """Add a model ID to the R2 manifest JSON. Returns True on success."""
    try:
        import boto3  # type: ignore
        from botocore.exceptions import BotoCoreError, ClientError  # type: ignore

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )
        try:
            resp = s3.get_object(Bucket=bucket, Key=_MANIFEST_KEY)
            manifest: dict = json.loads(resp["Body"].read())
        except ClientError:
            manifest = {"models": []}

        models: list[str] = manifest.get("models", [])
        if model_id not in models:
            models.append(model_id)
            manifest["models"] = models
            s3.put_object(
                Bucket=bucket,
                Key=_MANIFEST_KEY,
                Body=json.dumps(manifest, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            logger.info("R2 manifest updated — added %s", model_id)
        return True
    except ImportError:
        logger.warning("boto3 not installed; cannot update R2 manifest")
        return False
    except (BotoCoreError, ClientError) as exc:  # type: ignore[name-defined]
        logger.error("R2 manifest write failed: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error updating R2 manifest: %s", exc)
        return False
