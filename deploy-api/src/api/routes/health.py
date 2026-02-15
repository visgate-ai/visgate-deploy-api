"""Health and readiness endpoints."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import get_firestore
from src.core.config import get_settings
from src.core.telemetry import get_metrics

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness: minimal check, <10ms."""
    return {"status": "ok"}


@router.get("/readiness", response_model=None)
async def readiness(firestore_client=Depends(get_firestore)):
    """Readiness: verify Firestore connection."""
    settings = get_settings()
    try:
        coll = firestore_client.collection(settings.firestore_collection_deployments)
        _ = list(coll.limit(1).stream())
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unready", "error": str(e)},
        )
    return {"status": "ready"}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int((len(sorted_vals) - 1) * p)
    return round(sorted_vals[idx], 2)


@router.get("/metrics")
async def metrics() -> dict:
    """Simple JSON metrics endpoint for operational visibility."""
    snapshot = get_metrics()
    durations = snapshot.get("deployments_ready_duration_seconds", {}).get("values", [])
    return {
        "deployments_created_total": snapshot.get("deployments_created_total", 0),
        "webhook_delivery_failures_total": snapshot.get("webhook_delivery_failures_total", 0),
        "runpod_api_errors_total": snapshot.get("runpod_api_errors_total", 0),
        "deployments_ready_duration_seconds": {
            "count": len(durations),
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "sum": round(float(sum(durations)), 2),
        },
    }
