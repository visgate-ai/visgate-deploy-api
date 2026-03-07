"""Unit tests for public root-path stripping middleware."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware.strip_root_path import StripRootPathMiddleware


def test_strip_root_path_middleware_allows_prefixed_health() -> None:
    app = FastAPI()
    app.add_middleware(StripRootPathMiddleware, root_path="/deployapi")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    with TestClient(app) as client:
        resp = client.get("/deployapi/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}