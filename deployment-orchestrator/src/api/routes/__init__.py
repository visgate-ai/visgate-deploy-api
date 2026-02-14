"""API route modules."""

from src.api.routes.deployments import router as deployments_router
from src.api.routes.health import router as health_router
from src.api.routes.internal import router as internal_router
from src.api.routes.tasks import router as tasks_router

__all__ = ["deployments_router", "health_router", "internal_router", "tasks_router"]
