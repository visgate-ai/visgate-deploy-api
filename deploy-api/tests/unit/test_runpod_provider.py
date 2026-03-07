from unittest.mock import AsyncMock

import pytest

from src.services.runpod import RunpodProvider


@pytest.mark.asyncio
async def test_submit_job_embeds_s3_config_in_input_for_worker() -> None:
    provider = RunpodProvider()
    provider._endpoint_request = AsyncMock(return_value={"id": "job_1", "status": "IN_QUEUE"})

    await provider.submit_job(
        "https://api.runpod.ai/v2/endpoint_123/run",
        "rpa_test",
        {"prompt": "A futuristic skyline"},
        webhook_url="https://example.com/webhook",
        s3_config={
            "accessId": "key-id",
            "accessSecret": "secret-key",
            "bucketName": "user-results",
            "endpointUrl": "https://storage.example.com",
        },
    )

    payload = provider._endpoint_request.await_args.kwargs["json_payload"]
    assert payload["s3Config"]["bucketName"] == "user-results"
    assert payload["input"]["s3Config"]["bucketName"] == "user-results"
    assert payload["input"]["prompt"] == "A futuristic skyline"