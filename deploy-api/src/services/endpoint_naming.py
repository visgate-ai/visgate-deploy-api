"""Deterministic endpoint naming for warm discovery."""

from __future__ import annotations


def model_slug(model_id: str) -> str:
    return (model_id or "").strip().replace("/", "--")


def user_endpoint_name(user_hash: str, model_id: str) -> str:
    short_hash = (user_hash or "")[:10]
    return f"visgate-{short_hash}-{model_slug(model_id)}"


def pool_endpoint_name(model_id: str) -> str:
    return f"visgate-pool-{model_slug(model_id)}"
