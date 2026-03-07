"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware.strip_root_path import StripRootPathMiddleware
from src.api.routes import (
    deployments_router,
    health_router,
    internal_router,
    models_router,
    providers_router,
    tasks_router,
)
from src.core.config import get_settings
from src.core.errors import OrchestratorError, RateLimitError
from src.core.logging import configure_logging
from src.core.telemetry import init_telemetry, instrument_fastapi


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: configure logging and telemetry."""
    settings = get_settings()
    configure_logging(settings.log_level)
    init_telemetry(project_id=settings.gcp_project_id)
    yield
    # Shutdown: nothing to close for now


app = FastAPI(
    title="Visgate Deploy API",
    description="GCP Cloud Run deployment orchestrator for Hugging Face + Runpod",
    version="1.0.0",
    lifespan=lifespan,
    root_path=get_settings().root_path,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(StripRootPathMiddleware, root_path=get_settings().root_path)

app.include_router(health_router)
app.include_router(models_router)
app.include_router(providers_router)
app.include_router(deployments_router)
app.include_router(internal_router)
app.include_router(tasks_router)

instrument_fastapi(app)


@app.exception_handler(OrchestratorError)
async def orchestrator_error_handler(request: Request, exc: OrchestratorError) -> JSONResponse:
    """Map custom exceptions to JSON response."""
    headers = {}
    if isinstance(exc, RateLimitError):
        retry_after = exc.details.get("retry_after_seconds")
        if retry_after:
            headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
        headers=headers,
    )


@app.get("/")
async def root() -> dict:
    """Root redirect or info."""
    settings = get_settings()
    docs_path = f"{settings.root_path.rstrip('/')}/docs" if settings.root_path else "/docs"
    return {"service": "visgate-deploy-api", "docs": docs_path}
