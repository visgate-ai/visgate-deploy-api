"""Cloudflare R2 model cache manifest reader.

Reads ``models/manifest.json`` from the configured R2 bucket and returns
the set of model IDs whose weights are stored on the platform cache.
Falls back to an empty set whenever R2 is not configured or the manifest
is absent / unreadable so callers can treat this as an optional enrichment.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
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
