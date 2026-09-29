"""HTTP routes for the web backend."""
from __future__ import annotations

from aura_web.routes.auth import router as auth_router
from aura_web.routes.billing import router as billing_router
from aura_web.routes.dashboard import router as dashboard_router
from aura_web.routes.stripe_webhook import router as stripe_webhook_router

__all__ = ["auth_router", "billing_router", "dashboard_router", "stripe_webhook_router"]
