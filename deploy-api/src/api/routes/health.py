"""Health and readiness endpoints."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import get_firestore
from src.core.config import get_settings

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
