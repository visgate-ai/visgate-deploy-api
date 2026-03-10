"""Platform-owned Cloudflare R2 helpers for staged inference I/O."""

from __future__ import annotations

import mimetypes
import os
import posixpath
import tempfile
import uuid
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.core.config import Settings

_INPUT_URL_KEYS = (
    "input_image_url",
    "image_url",
    "audio_url",
    "video_url",
)


def _create_s3_client(endpoint_url: str, access_key_id: str, secret_access_key: str):
    import boto3  # type: ignore

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
    )


def _guess_suffix(source_url: str, content_type: str | None = None) -> str:
    parsed = urlparse(source_url)
    suffix = Path(parsed.path).suffix
    if suffix:
        return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
        if guessed:
            return guessed
    return ".bin"


def build_platform_output_s3_config(settings: Settings, deployment_id: str, job_id: str) -> dict[str, Any]:
    if not (
        settings.r2_access_key_id_rw
        and settings.r2_secret_access_key_rw
        and settings.inference_r2_bucket_name_output
        and settings.r2_endpoint_url
    ):
        raise ValueError("Platform R2 output storage is not fully configured")
    return {
        "accessId": settings.r2_access_key_id_rw,
        "accessSecret": settings.r2_secret_access_key_rw,
        "bucketName": settings.inference_r2_bucket_name_output,
        "endpointUrl": settings.r2_endpoint_url,
        "keyPrefix": f"inference/outputs/{deployment_id}/{job_id}",
    }


def sanitize_platform_output_destination(settings: Settings, deployment_id: str, job_id: str) -> dict[str, Any]:
    return {
        "bucket_name": settings.inference_r2_bucket_name_output,
        "endpoint_url": settings.r2_endpoint_url,
        "key_prefix": f"inference/outputs/{deployment_id}/{job_id}",
    }


def _input_object_key(job_id: str, field_name: str, source_url: str, content_type: str | None = None) -> str:
    suffix = _guess_suffix(source_url, content_type)
    return posixpath.join("inference", "inputs", job_id, f"{field_name}-{uuid.uuid4().hex}{suffix}")


def _stage_remote_file(settings: Settings, source_url: str, object_key: str) -> dict[str, Any]:
    if not (
        settings.r2_access_key_id_rw
        and settings.r2_secret_access_key_rw
        and settings.inference_r2_bucket_name_input
        and settings.r2_endpoint_url
    ):
        raise ValueError("Platform R2 input storage is not fully configured")

    tmp_path = ""
    content_type = None
    try:
        with urllib.request.urlopen(source_url, timeout=60) as response:
            content_type = response.headers.get("Content-Type")
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
        client = _create_s3_client(
            settings.r2_endpoint_url,
            settings.r2_access_key_id_rw,
            settings.r2_secret_access_key_rw,
        )
        extra_args = {"ContentType": content_type} if content_type else None
        if extra_args:
            client.upload_file(tmp_path, settings.inference_r2_bucket_name_input, object_key, ExtraArgs=extra_args)
        else:
            client.upload_file(tmp_path, settings.inference_r2_bucket_name_input, object_key)
        return {
            "bucket_name": settings.inference_r2_bucket_name_input,
            "endpoint_url": settings.r2_endpoint_url,
            "key": object_key,
            "url": f"{settings.r2_endpoint_url.rstrip('/')}/{settings.inference_r2_bucket_name_input}/{object_key}",
            "content_type": content_type,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def stage_input_payload_to_r2(settings: Settings, job_id: str, payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    staged_payload = dict(payload)
    staged_inputs: list[dict[str, Any]] = []
    for field_name in _INPUT_URL_KEYS:
        value = staged_payload.get(field_name)
        if not isinstance(value, str) or not value.startswith(("http://", "https://")):
            continue
        object_key = _input_object_key(job_id, field_name, value)
        artifact = _stage_remote_file(settings, value, object_key)
        staged_payload.pop(field_name, None)
        staged_payload[field_name.removesuffix("_url") + "_r2"] = artifact
        staged_inputs.append(artifact)
    return staged_payload, staged_inputs