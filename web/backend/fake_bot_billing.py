"""A stand-in for the bot's internal billing API, implementing the same contract in memory.

The web backend's tests use this so they run without the bot's database and
embedding stack, exactly as they run without Discord. The contract it
implements -- the shared secret checked first, event deduplication checked
before the version, a compare-and-swap on the version -- is the one
aura.billing.internal_api implements for real, and tests/test_billing_contract.py
at the repository root drives the web backend's real client against the REAL
bot API to prove the two agree. Without that test, a stand-in could quietly
drift from the thing it stands in for.

Plans are whatever a test sets in `plans`: deciding a plan is the bot's job,
tested in tests/test_billing_entitlement.py, and re-implementing it here would
be a second copy to keep correct.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def free_plan() -> dict[str, Any]:
    """A guild with no subscription on an enforced deployment."""
    return {
        "tier": "free",
        "basis": "subscription",
        "standing": "no_subscription",
        "access_until": None,
        "paid_through": None,
        "in_force_subscription_count": 0,
        "subscriptions": [],
    }


@dataclass
class FakeBotBillingState:
    """The stand-in's memory, plus the failure switches and hooks tests use."""

    secret: str = "fake-bot-internal-api-shared-secret-000000000000"
    versions: dict[str, int] = field(default_factory=dict)
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    processed_events: dict[str, str] = field(default_factory=dict)
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)
    request_log: list[str] = field(default_factory=list)
    applied_snapshots: list[dict[str, Any]] = field(default_factory=list)
    fail_status: int | None = None
    # Awaited inside every apply, before the compare-and-swap -- lets a test
    # commit a competing write at exactly the moment a real race would.
    before_apply: Callable[[dict[str, Any]], Awaitable[None]] | None = None

    def plan_for(self, guild_id: str) -> dict[str, Any]:
        """The plan a test configured for a guild, or Free."""
        return self.plans.get(guild_id, free_plan())

    def commit_competing_write(self, subscription_id: str) -> None:
        """Bump a subscription's version as if another sync had just committed."""
        self.versions[subscription_id] = self.versions.get(subscription_id, 0) + 1


def create_fake_bot_billing(state: FakeBotBillingState) -> FastAPI:
    """Build the ASGI application."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def refuse(request: Request) -> JSONResponse | None:
        presented = request.headers.get("authorization", "")
        if not hmac.compare_digest(presented.encode(), f"Bearer {state.secret}".encode()):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if state.fail_status is not None:
            return JSONResponse({"error": "internal_error"}, status_code=state.fail_status)
        return None

    @app.post("/internal/v1/subscriptions/sync-state")
    async def sync_state(request: Request) -> JSONResponse:
        state.request_log.append("sync-state")
        refusal = refuse(request)
        if refusal is not None:
            return refusal
        body = await request.json()
        return JSONResponse(
            {
                "event_processed": body["event_id"] in state.processed_events,
                "version": state.versions.get(body["subscription_id"], 0),
            }
        )

    @app.post("/internal/v1/subscriptions/apply")
    async def apply(request: Request) -> JSONResponse:
        state.request_log.append("apply")
        refusal = refuse(request)
        if refusal is not None:
            return refusal
        body = await request.json()
        if state.before_apply is not None:
            await state.before_apply(body)
        snapshot = body["snapshot"]
        subscription_id = snapshot["subscription_id"]
        current = state.versions.get(subscription_id, 0)
        if body["event_id"] is not None and body["event_id"] in state.processed_events:
            return JSONResponse({"outcome": "duplicate", "version": current})
        if body["expected_version"] != current:
            return JSONResponse(
                {"outcome": "version_conflict", "version": current}, status_code=409
            )
        state.versions[subscription_id] = current + 1
        state.snapshots[subscription_id] = snapshot
        state.applied_snapshots.append(body)
        if body["event_id"] is not None:
            state.processed_events[body["event_id"]] = body["event_type"]
        return JSONResponse({"outcome": "applied", "version": current + 1})

    @app.post("/internal/v1/guilds/plans")
    async def plans(request: Request) -> JSONResponse:
        state.request_log.append("plans")
        refusal = refuse(request)
        if refusal is not None:
            return refusal
        body = await request.json()
        return JSONResponse(
            {"plans": {guild_id: state.plan_for(guild_id) for guild_id in body["guild_ids"]}}
        )

    return app
