"""HTTP routes for the web backend."""
from __future__ import annotations

from aura_web.routes.auth import router as auth_router
from aura_web.routes.dashboard import router as dashboard_router

__all__ = ["auth_router", "dashboard_router"]
