"""aura.billing.internal_api: the bot's billing API, attacked the way an internal network would.

The API sits on an internal network behind a shared secret, and it is the one
path by which a guild's plan changes -- so it is tested as if both of those
protections might someday be the only thing standing, and as if whatever calls
it might be wrong: missing and wrong secrets, a secret compared by length or by
prefix, routes probed without authentication, bodies that are too big, not
JSON, JSON with duplicate keys, and every field of a snapshot with the wrong
type, the wrong shape or an out-of-range value. Each refusal is asserted to
leave the database exactly as it was.
"""
from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import aiohttp
import aiosqlite
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from aura.billing import GracePolicy, PlanGate
from aura.billing.internal_api import (
    MAX_GUILD_IDS_PER_REQUEST,
    MAX_REQUEST_BYTES,
    create_internal_api_app,
    start_internal_api,
)
from aura.db.repository import init_schema
from aura.db.subscriptions import count_processed_events, load_subscription_records

NOW = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)
SECRET = "internal-api-test-secret-" + "x" * 30
POLICY = GracePolicy(renewal_grace=timedelta(hours=72), payment_failure_grace=timedelta(days=7))
GUILD = "100000000000000001"
AUTH = {"Authorization": f"Bearer {SECRET}"}


def snapshot(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "subscription_id": "sub_A1",
        "customer_id": "cus_A1",
        "guild_id": GUILD,
        "purchaser_user_id": "5000",
        "status": "active",
        "cancel_at_period_end": False,
        "cancel_at": None,
        "collection_paused": False,
        "latest_invoice_status": "paid",
        "current_period_start": int(NOW.timestamp()),
        "current_period_end": int((NOW + timedelta(days=30)).timestamp()),
        "livemode": False,
    }
    values.update(overrides)
    return values


def apply_body(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "event_id": "evt_1",
        "event_type": "customer.subscription.updated",
        "expected_version": 0,
        "snapshot": snapshot(),
    }
    values.update(overrides)
    return values


@pytest_asyncio.fixture
async def setup():
    conn = await aiosqlite.connect(":memory:")
    await init_schema(conn)
    gate = PlanGate(enforced=True, policy=POLICY, complimentary_guild_ids=frozenset(), records=[], clock=lambda: NOW)
    app = create_internal_api_app(conn, gate, secret=SECRET, clock=lambda: NOW)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield client, conn, gate
    finally:
        await client.close()
        await conn.close()


async def nothing_was_written(conn: aiosqlite.Connection) -> bool:
    return await load_subscription_records(conn) == [] and await count_processed_events(conn) == 0


class TestAuthentication:
    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Authorization": ""},
            {"Authorization": SECRET},
            {"Authorization": f"Bearer {SECRET[:-1]}"},
            {"Authorization": f"Bearer {SECRET}x"},
            {"Authorization": f"Bearer  {SECRET}"},
            {"Authorization": f"bearer {SECRET}"},
            {"Authorization": f"Basic {SECRET}"},
            {"Authorization": "Bearer " + "x" * len(SECRET)},
        ],
    )
    async def test_anything_but_the_exact_secret_is_refused_and_writes_nothing(self, setup, headers) -> None:
        client, conn, _ = setup

        response = await client.post("/internal/v1/subscriptions/apply", json=apply_body(), headers=headers)

        assert response.status == 401
        assert await response.json() == {"error": "unauthorized"}
        assert await nothing_was_written(conn)

    async def test_two_authorization_headers_are_refused_even_if_one_is_right(self, setup) -> None:
        client, conn, _ = setup

        response = await client.post(
            "/internal/v1/subscriptions/apply",
            data=json.dumps(apply_body()),
            headers=[("Authorization", "Bearer wrong"), ("Authorization", f"Bearer {SECRET}"), ("Content-Type", "application/json")],
        )

        assert response.status == 401
        assert await nothing_was_written(conn)

    @pytest.mark.parametrize("path", ["/", "/internal/v1/admin", "/internal/v1/guilds", "/../etc/passwd"])
    async def test_an_unauthenticated_caller_cannot_map_the_routes(self, setup, path: str) -> None:
        client, _, _ = setup

        response = await client.get(path)

        assert response.status == 401

    async def test_an_authenticated_unknown_route_is_a_json_404(self, setup) -> None:
        client, _, _ = setup

        response = await client.get("/internal/v1/everything", headers=AUTH)

        assert response.status == 404
        assert await response.json() == {"error": "not_found"}

    async def test_the_wrong_method_is_a_json_405(self, setup) -> None:
        client, _, _ = setup

        response = await client.get("/internal/v1/subscriptions/apply", headers=AUTH)

        assert response.status == 405
        assert await response.json() == {"error": "method_not_allowed"}


class TestBodies:
    async def test_a_body_that_is_not_json_content_is_refused(self, setup) -> None:
        client, conn, _ = setup

        response = await client.post(
            "/internal/v1/subscriptions/apply", data=json.dumps(apply_body()),
            headers={**AUTH, "Content-Type": "text/plain"},
        )

        assert response.status == 415
        assert await nothing_was_written(conn)

    async def test_an_oversized_body_is_refused(self, setup) -> None:
        client, conn, _ = setup
        body = apply_body(padding="x" * (MAX_REQUEST_BYTES + 1))

        response = await client.post("/internal/v1/subscriptions/apply", json=body, headers=AUTH)

        assert response.status == 413
        assert await nothing_was_written(conn)

    @pytest.mark.parametrize(
        "raw",
        [
            b"",
            b"not json",
            b"[]",
            b"null",
            b"\xff\xfe",
            b'{"event_id": NaN}',
            b"[" * 5000 + b"]" * 5000,
        ],
    )
    async def test_malformed_json_is_refused(self, setup, raw: bytes) -> None:
        client, conn, _ = setup

        response = await client.post(
            "/internal/v1/subscriptions/apply", data=raw, headers={**AUTH, "Content-Type": "application/json"}
        )

        assert response.status == 400
        assert await response.json() == {"error": "invalid_request"}
        assert await nothing_was_written(conn)

    async def test_a_duplicated_key_is_refused_rather_than_last_one_wins(self, setup) -> None:
        client, conn, _ = setup
        valid = json.dumps(apply_body())
        raw = '{"expected_version": 5, ' + valid[1:]

        response = await client.post(
            "/internal/v1/subscriptions/apply", data=raw, headers={**AUTH, "Content-Type": "application/json"}
        )

        assert response.status == 400
        assert await nothing_was_written(conn)


class TestApplyValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"unexpected": 1},
            {"expected_version": True},
            {"expected_version": "0"},
            {"expected_version": -1},
            {"expected_version": 1.0},
            {"expected_version": 2**63},
            {"event_id": "evt_1", "event_type": None},
            {"event_id": None, "event_type": "invoice.paid"},
            {"event_id": "evt_1\n"},
            {"event_id": "evt_" + "a" * 201},
            {"event_id": "ch_notanevent"},
            {"event_type": "Invoice.Paid"},
            {"event_type": "invoice"},
            {"event_type": "invoice.paid; DROP TABLE facts"},
        ],
    )
    async def test_a_malformed_envelope_is_refused(self, setup, overrides) -> None:
        client, conn, _ = setup

        response = await client.post("/internal/v1/subscriptions/apply", json=apply_body(**overrides), headers=AUTH)

        assert response.status == 400
        assert await nothing_was_written(conn)

    async def test_a_missing_envelope_key_is_refused_even_when_it_would_be_null(self, setup) -> None:
        client, conn, _ = setup
        body = apply_body()
        del body["event_id"]
        body["event_type"] = None

        response = await client.post("/internal/v1/subscriptions/apply", json=body, headers=AUTH)

        assert response.status == 400
        assert await nothing_was_written(conn)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"guild_id": 100000000000000001},
            {"guild_id": "0100000000000000001"},
            {"guild_id": "0"},
            {"guild_id": "-1"},
            {"guild_id": "9223372036854775808"},
            {"guild_id": "١٢٣"},
            {"guild_id": " 100000000000000001"},
            {"purchaser_user_id": 5000},
            {"subscription_id": "sub_"},
            {"subscription_id": "sub_A1'--"},
            {"subscription_id": "cus_A1"},
            {"customer_id": "cus_A1 "},
            {"status": "free_forever"},
            {"status": "ACTIVE"},
            {"cancel_at_period_end": 0},
            {"cancel_at_period_end": "false"},
            {"collection_paused": None},
            {"livemode": "true"},
            {"latest_invoice_status": "refunded"},
            {"cancel_at": -1},
            {"cancel_at": "1757764800"},
            {"current_period_start": 253_402_300_800},
            {"current_period_end": 1},
            {"current_period_end": 1.5},
        ],
    )
    async def test_a_malformed_snapshot_is_refused(self, setup, overrides) -> None:
        client, conn, _ = setup

        response = await client.post(
            "/internal/v1/subscriptions/apply", json=apply_body(snapshot=snapshot(**overrides)), headers=AUTH
        )

        assert response.status == 400
        assert await response.json() == {"error": "invalid_request"}
        assert await nothing_was_written(conn)


class TestApplyBehaviour:
    async def test_a_valid_snapshot_is_stored_and_moves_the_gate_immediately(self, setup) -> None:
        client, conn, gate = setup
        assert not gate.allows_pro(int(GUILD))

        response = await client.post("/internal/v1/subscriptions/apply", json=apply_body(), headers=AUTH)

        assert response.status == 200
        assert await response.json() == {"outcome": "applied", "version": 1}
        assert gate.allows_pro(int(GUILD))
        assert len(await load_subscription_records(conn)) == 1

    async def test_the_same_event_again_is_a_duplicate_and_changes_nothing(self, setup) -> None:
        client, conn, gate = setup
        await client.post("/internal/v1/subscriptions/apply", json=apply_body(), headers=AUTH)

        again = await client.post(
            "/internal/v1/subscriptions/apply",
            json=apply_body(expected_version=1, snapshot=snapshot(status="canceled")),
            headers=AUTH,
        )

        assert again.status == 200
        assert await again.json() == {"outcome": "duplicate", "version": 1}
        assert gate.allows_pro(int(GUILD))
        assert await count_processed_events(conn) == 1

    async def test_a_stale_version_is_a_409_carrying_the_current_version(self, setup) -> None:
        client, _, gate = setup
        await client.post("/internal/v1/subscriptions/apply", json=apply_body(), headers=AUTH)

        stale = await client.post(
            "/internal/v1/subscriptions/apply",
            json=apply_body(event_id="evt_2", snapshot=snapshot(status="canceled")),
            headers=AUTH,
        )

        assert stale.status == 409
        assert await stale.json() == {"outcome": "version_conflict", "version": 1}
        assert gate.allows_pro(int(GUILD))

    async def test_sync_state_reports_processing_and_version(self, setup) -> None:
        client, _, _ = setup
        await client.post("/internal/v1/subscriptions/apply", json=apply_body(), headers=AUTH)

        response = await client.post(
            "/internal/v1/subscriptions/sync-state", json={"subscription_id": "sub_A1", "event_id": "evt_1"}, headers=AUTH
        )

        assert await response.json() == {"event_processed": True, "version": 1}

    async def test_an_unexpected_failure_is_a_bare_500_code_without_internals(self, setup) -> None:
        client, _, _ = setup

        with patch("aura.billing.internal_api.apply_snapshot", side_effect=RuntimeError("secret internals /path")):
            response = await client.post("/internal/v1/subscriptions/apply", json=apply_body(), headers=AUTH)

        assert response.status == 500
        text = await response.text()
        assert json.loads(text) == {"error": "internal_error"}
        assert "internals" not in text


class TestPlans:
    async def test_plans_are_returned_for_exactly_the_guilds_asked_about(self, setup) -> None:
        client, _, _ = setup
        await client.post("/internal/v1/subscriptions/apply", json=apply_body(), headers=AUTH)

        response = await client.post(
            "/internal/v1/guilds/plans", json={"guild_ids": [GUILD, "200000000000000002"]}, headers=AUTH
        )

        body = await response.json()
        assert set(body["plans"]) == {GUILD, "200000000000000002"}
        paying = body["plans"][GUILD]
        assert paying["tier"] == "pro"
        assert paying["standing"] == "active"
        assert paying["in_force_subscription_count"] == 1
        assert paying["subscriptions"] == [
            {"subscription_id": "sub_A1", "customer_id": "cus_A1", "purchaser_user_id": "5000", "status": "active", "grants_access": True}
        ]
        assert body["plans"]["200000000000000002"]["tier"] == "free"

    @pytest.mark.parametrize(
        "guild_ids",
        [
            [],
            [GUILD, GUILD],
            [100000000000000001],
            [str(100000000000000000 + index) for index in range(MAX_GUILD_IDS_PER_REQUEST + 1)],
            "100000000000000001",
            ["9223372036854775808"],
        ],
    )
    async def test_a_malformed_plan_request_is_refused(self, setup, guild_ids) -> None:
        client, _, _ = setup

        response = await client.post("/internal/v1/guilds/plans", json={"guild_ids": guild_ids}, headers=AUTH)

        assert response.status == 400

    async def test_no_response_ever_contains_the_secret(self, setup) -> None:
        client, _, _ = setup
        requests = [
            ("POST", "/internal/v1/subscriptions/apply", apply_body(), AUTH),
            ("POST", "/internal/v1/subscriptions/apply", apply_body(), {}),
            ("POST", "/internal/v1/guilds/plans", {"guild_ids": [GUILD]}, AUTH),
            ("GET", "/nowhere", None, AUTH),
        ]
        for method, path, body, headers in requests:
            response = await client.request(method, path, json=body, headers=headers)
            raw = await response.read()
            assert SECRET.encode() not in raw
            assert all(SECRET not in value for value in response.headers.values())


class TestListener:
    async def test_starts_on_a_real_socket_and_stops_releasing_it(self) -> None:
        conn = await aiosqlite.connect(":memory:")
        await init_schema(conn)
        server = await start_internal_api(
            conn, PlanGate.unenforced(), secret=SECRET, host="127.0.0.1", port=0
        )
        port = server.bound_port
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{port}/internal/v1/guilds/plans", json={"guild_ids": [GUILD]}, headers=AUTH
                ) as response:
                    assert response.status == 200
        finally:
            await server.stop()
            await conn.close()

        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1).close()

    async def test_a_port_that_cannot_be_bound_fails_loudly(self) -> None:
        conn = await aiosqlite.connect(":memory:")
        await init_schema(conn)
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        try:
            with pytest.raises(OSError):
                await start_internal_api(
                    conn, PlanGate.unenforced(), secret=SECRET, host="127.0.0.1", port=blocker.getsockname()[1]
                )
        finally:
            blocker.close()
            await conn.close()
