"""User webhook notification with retries and graceful degradation."""

import asyncio
from typing import Any, Optional

import httpx

from src.core.logging import structured_log
from src.core.telemetry import record_webhook_failure, span


async def notify(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: int = 10,
    retries: int = 3,
    deployment_id: Optional[str] = None,
) -> bool:
    """
    POST payload to user webhook URL with exponential backoff.
    Returns True if delivery succeeded, False otherwise (then callers should set webhook_failed).
    """
    with span("webhook.notify", {"url": url, "deployment_id": deployment_id}):
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    resp = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                if 200 <= resp.status_code < 300:
                    structured_log(
                        "INFO",
                        "Webhook delivered successfully",
                        deployment_id=deployment_id,
                        operation="webhook.notify",
                        metadata={"url": url, "status_code": resp.status_code},
                    )
                    return True
                last_error = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                last_error = e
            if attempt < retries - 1:
                delay = 2 ** attempt
                await asyncio.sleep(delay)
        record_webhook_failure()
        structured_log(
            "WARNING",
            f"Webhook delivery failed after {retries} retries: {last_error}",
            deployment_id=deployment_id,
            operation="webhook.notify",
            metadata={"url": url},
            error={"type": type(last_error).__name__, "message": str(last_error)},
        )
        return False
