"""Phase 4c audit, Attack 5: bypass the gate. Every Pro path, no subscription, nothing paid.

Written by the post-hoc audit of commit 9d0aa23 (reports/phase-4c-audit.md).
tests/test_billing_gating.py checks each Pro trigger with a gate built by hand
and mocks the next function down. This file goes further in three ways:

  * The gate is the production one: `PlanGate.from_settings` over an ENFORCED
    configuration, holding a canceled subscription for this guild and an
    active one for another guild, on the real clock -- so "no subscription"
    here means what it will mean in production, not an empty gate.
  * Everything downstream of the gate is a TRIPWIRE that records any touch,
    litellm's completion call is a spy, and every table's row count is
    compared before and after -- so "nothing paid" is observed, not inferred
    from one mock not being awaited. A recorded touch survives the broad
    ``except Exception`` in the background loops that would otherwise swallow it.
  * Each Free result has a positive control through the same harness on a
    complimentary (Pro) guild, which must touch the tripwires -- so no test
    here can pass because its path was simply unreachable.

It also pins, structurally, the full set of call sites of every paid entry
point, so a new Pro path cannot appear without this file being revisited.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import litellm
import pytest
from fastembed import TextEmbedding

from aura.backfill.worker import advance_due_backfills
from aura.billing import PlanGate, SubscriptionRecord, SubscriptionStatus
from aura.billing.entitlement import InvoiceStatus
from aura.commands.ask import ask_command
from aura.commands.backfill import backfill_start
from aura.commands.config import config_command
from aura.commands.digest import digest_command
from aura.commands.onboarding import onboarding_command
from aura.config import BillingMode, Settings
from aura.db.backfill_runs import BackfillState, get_active_run, start_backfill_run
from aura.db.digest_config import set_digest_config
from aura.db.extraction_channel_config import set_extraction_enabled
from aura.db.extraction_queue import enqueue_message
from aura.db.proactive_channel_config import set_channel_enabled
from aura.db.repository import init_schema
from aura.digest.scheduler import send_due_digests
from aura.extraction.pipeline import flush_due_batches, handle_extraction_message
from aura.facts_service import add_fact
from aura.onboarding.listener import handle_member_join
from aura.proactive.grace import GraceRegistry
from aura.proactive.listener import _still_fresh_enough_for_synthesis, handle_message
from aura.synthesis import SynthesisResult

GUILD: Final = 100000000000000001
OTHER_GUILD: Final = 200000000000000002
CHANNEL: Final = 300000000000000003
SECRET: Final = "audit-internal-api-secret-" + "y" * 20
SRC: Final = Path(__file__).resolve().parent.parent / "src" / "aura"

# The five spend ledgers: a row in any of them is a paid call that was claimed.
LEDGERS: Final = (
    "proactive_escalations",
    "extraction_calls",
    "supersession_calls",
    "variant_calls",
    "backfill_calls",
)


class TouchedTripwireError(Exception):
    """Raised by a tripwire the moment code reaches it."""


@dataclass
class Touches:
    """Everything the tripwires and the LLM spy saw during one run."""

    names: list[str] = field(default_factory=list)


class Tripwire:
    """Stands in for a collaborator the Free path must never reach."""

    def __init__(self, name: str, touches: Touches) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_touches", touches)

    def __getattr__(self, attribute: str) -> Any:
        self._touches.names.append(f"{self._name}.{attribute}")
        raise TouchedTripwireError(f"{self._name}.{attribute}")


@pytest.fixture
async def conn() -> AsyncIterator[aiosqlite.Connection]:
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


@pytest.fixture
def touches() -> Iterator[Touches]:
    """Every LLM completion in the process is replaced by a recording refusal."""
    seen = Touches()

    async def spy(*_args: Any, **_kwargs: Any) -> Any:
        seen.names.append("litellm.acompletion")
        raise TouchedTripwireError("litellm.acompletion")

    with patch.object(litellm, "acompletion", spy):
        yield seen


def settings(**overrides: Any) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="fake-token",  # type: ignore[arg-type]
        llm_api_key="fake-key",  # type: ignore[arg-type]
        synthesis_model="openrouter/fake/synthesis",
        proactive_model="openrouter/fake/proactive",
        extraction_model="openrouter/fake/extraction",
        supersession_model="openrouter/fake/supersession",
        billing_mode=BillingMode.ENFORCED,
        internal_api_secret=SECRET,  # type: ignore[arg-type]
        **overrides,
    )


def production_gate(*, complimentary: bool = False) -> PlanGate:
    """The enforced gate production builds: this guild canceled, another guild active."""
    now = datetime.now(UTC)
    common: dict[str, object] = {
        "customer_id": "cus_Audit",
        "purchaser_user_id": 5000,
        "cancel_at_period_end": False,
        "cancel_at": None,
        "collection_paused": False,
        "latest_invoice_status": InvoiceStatus.PAID,
        "current_period_start": now - timedelta(days=3),
        "current_period_end": now + timedelta(days=27),
        "livemode": False,
        "on_pro_price": True,
        "version": 1,
        "confirmed_at": now,
    }
    canceled_here = SubscriptionRecord.model_validate(
        common
        | {
            "subscription_id": "sub_Here",
            "guild_id": GUILD,
            "status": SubscriptionStatus.CANCELED,
        }
    )
    active_elsewhere = SubscriptionRecord.model_validate(
        common
        | {
            "subscription_id": "sub_Elsewhere",
            "guild_id": OTHER_GUILD,
            "status": SubscriptionStatus.ACTIVE,
        }
    )
    extra = {"billing_complimentary_guild_ids": str(GUILD)} if complimentary else {}
    gate = PlanGate.from_settings(settings(**extra), records=[canceled_here, active_elsewhere])
    assert gate.enforced
    assert gate.allows_pro(GUILD) is complimentary
    return gate


async def row_counts(conn: aiosqlite.Connection) -> dict[str, int]:
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ) as cursor:
        tables = [row[0] for row in await cursor.fetchall()]
    counts: dict[str, int] = {}
    for table in tables:
        async with conn.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
            row = await cursor.fetchone()
        counts[table] = int(row[0]) if row else 0
    return counts


def human_message() -> MagicMock:
    message = MagicMock(spec=discord.Message)
    message.content = "The event starts on Saturday at 18:00, where do I sign up?"
    message.guild = MagicMock()
    message.guild.id = GUILD
    message.channel = MagicMock()
    message.channel.id = CHANNEL
    message.channel.name = "general"
    message.id = 900000000000000009
    message.author = MagicMock()
    message.author.bot = False
    message.author.id = 4242
    message.webhook_id = None
    message.interaction_metadata = None
    message.type = discord.MessageType.default
    message.created_at = datetime.now(UTC)
    return message


def assert_nothing_paid(touches: Touches, before: dict[str, int], after: dict[str, int]) -> None:
    assert touches.names == [], f"the Free path reached {touches.names}"
    for ledger in LEDGERS:
        assert after[ledger] == 0, f"{ledger} gained a row on a Free guild"
    grown = {
        table: (before[table], after[table]) for table in after if after[table] > before[table]
    }
    assert grown == {}, f"tables written on a Free guild: {grown}"


class TestProactiveRelief:
    async def _run(self, conn: aiosqlite.Connection, touches: Touches, gate: PlanGate) -> None:
        await set_channel_enabled(
            conn, guild_id=GUILD, channel_id=CHANNEL, enabled=True, updated_by_id=1
        )
        await handle_message(
            human_message(),
            db=conn,
            detector=Tripwire("detector", touches),  # type: ignore[arg-type]
            model=Tripwire("embedding_model", touches),  # type: ignore[arg-type]
            config=Tripwire("gate_config", touches),  # type: ignore[arg-type]
            settings=settings(),
            grace_registry=GraceRegistry(),
            plan_gate=gate,
        )

    async def test_free_touches_nothing_and_writes_nothing(
        self, conn: aiosqlite.Connection, touches: Touches
    ) -> None:
        await set_channel_enabled(
            conn, guild_id=GUILD, channel_id=CHANNEL, enabled=True, updated_by_id=1
        )
        before = await row_counts(conn)

        await self._run(conn, touches, production_gate())

        assert_nothing_paid(touches, before, await row_counts(conn))

    async def test_positive_control_pro_reaches_the_pipeline(
        self, conn: aiosqlite.Connection, touches: Touches
    ) -> None:
        with suppress(TouchedTripwireError):
            await self._run(conn, touches, production_gate(complimentary=True))

        assert touches.names

    async def test_the_post_grace_recheck_refuses_without_reading_anything(
        self, touches: Touches
    ) -> None:
        fresh = await _still_fresh_enough_for_synthesis(
            Tripwire("db", touches),  # type: ignore[arg-type]
            guild_id=GUILD,
            channel_id=CHANNEL,
            message_id=1,
            plan_gate=production_gate(),
        )

        assert fresh is False
        assert touches.names == []


class TestExtraction:
    async def test_intake_touches_nothing_and_queues_nothing(
        self, conn: aiosqlite.Connection, touches: Touches
    ) -> None:
        await set_extraction_enabled(
            conn, guild_id=GUILD, channel_id=CHANNEL, enabled=True, updated_by_id=1
        )
        before = await row_counts(conn)

        await handle_extraction_message(
            human_message(),
            db=conn,
            detector=Tripwire("detector", touches),  # type: ignore[arg-type]
            settings=settings(),
            plan_gate=production_gate(),
        )

        assert_nothing_paid(touches, before, await row_counts(conn))

    async def _queue_due_batch(self, conn: aiosqlite.Connection) -> None:
        earlier = datetime.now(UTC) - timedelta(hours=2)
        await enqueue_message(
            conn,
            guild_id=GUILD,
            channel_id=CHANNEL,
            message_id=1,
            channel_name="general",
            content="The event starts on Saturday at 18:00.",
            message_created_at=earlier,
            now=earlier,
        )

    async def test_a_batch_queued_under_pro_and_flushed_under_free_spends_nothing(
        self, conn: aiosqlite.Connection, touches: Touches
    ) -> None:
        await self._queue_due_batch(conn)
        before = await row_counts(conn)

        flushed = await flush_due_batches(
            conn,
            Tripwire("embedding_model", touches),  # type: ignore[arg-type]
            settings=settings(),
            now=datetime.now(UTC),
            plan_gate=production_gate(),
        )

        assert flushed == 0
        after = await row_counts(conn)
        assert after["extraction_queue"] == 0  # dropped, as documented, not held
        assert_nothing_paid(touches, before, after)

    async def test_positive_control_pro_reaches_the_distiller(
        self, conn: aiosqlite.Connection, touches: Touches
    ) -> None:
        await self._queue_due_batch(conn)

        # The distiller reads its key through load_settings: the same fake
        # settings, not the developer's .env (V-04).
        with patch("aura.extraction.distiller.load_settings", return_value=settings()):
            await flush_due_batches(
                conn,
                Tripwire("embedding_model", touches),  # type: ignore[arg-type]
                settings=settings(),
                now=datetime.now(UTC),
                plan_gate=production_gate(complimentary=True),
            )

        assert "litellm.acompletion" in touches.names


class TestDigest:
    async def _configure(self, conn: aiosqlite.Connection) -> None:
        await set_digest_config(
            conn,
            guild_id=GUILD,
            channel_id=CHANNEL,
            interval_seconds=86400,
            enabled=True,
            updated_by_id=1,
        )

    async def test_free_touches_nothing_and_claims_no_window(
        self, conn: aiosqlite.Connection, touches: Touches
    ) -> None:
        await self._configure(conn)
        before = await row_counts(conn)

        posted = await send_due_digests(
            conn,
            Tripwire("digest_gateway", touches),  # type: ignore[arg-type]
            now=datetime.now(UTC) + timedelta(days=3),
            plan_gate=production_gate(),
        )

        assert posted == 0
        assert_nothing_paid(touches, before, await row_counts(conn))

    async def test_positive_control_pro_reaches_the_gateway_or_claims_a_window(
        self, conn: aiosqlite.Connection, touches: Touches
    ) -> None:
        await self._configure(conn)
        before = await row_counts(conn)

        await send_due_digests(
            conn,
            Tripwire("digest_gateway", touches),  # type: ignore[arg-type]
            now=datetime.now(UTC) + timedelta(days=3),
            plan_gate=production_gate(complimentary=True),
        )

        assert touches.names or (await row_counts(conn)) != before


class TestOnboarding:
    def _member(self) -> MagicMock:
        member = MagicMock(spec=discord.Member)
        member.bot = False
        member.id = 42
        member.guild = MagicMock()
        member.guild.id = GUILD
        return member

    async def test_free_touches_neither_the_database_nor_discord(self, touches: Touches) -> None:
        await handle_member_join(
            self._member(),
            db=Tripwire("db", touches),  # type: ignore[arg-type]
            gateway=Tripwire("onboarding_gateway", touches),  # type: ignore[arg-type]
            settings=settings(),
            plan_gate=production_gate(),
        )

        assert touches.names == []

    async def test_positive_control_pro_reads_its_configuration(self, touches: Touches) -> None:
        with suppress(TouchedTripwireError):
            await handle_member_join(
                self._member(),
                db=Tripwire("db", touches),  # type: ignore[arg-type]
                gateway=Tripwire("onboarding_gateway", touches),  # type: ignore[arg-type]
                settings=settings(),
                plan_gate=production_gate(complimentary=True),
            )

        assert touches.names


class TestBackfill:
    async def _start(self, conn: aiosqlite.Connection) -> None:
        await start_backfill_run(
            conn,
            guild_id=GUILD,
            channel_id=CHANNEL,
            until_message_id=2**62,
            after_message_id=None,
            requested_by_id=1,
            now=datetime.now(UTC),
        )

    async def _advance(self, conn: aiosqlite.Connection, touches: Touches, gate: PlanGate) -> int:
        return await advance_due_backfills(
            conn,
            Tripwire("embedding_model", touches),  # type: ignore[arg-type]
            Tripwire("backfill_gateway", touches),  # type: ignore[arg-type]
            Tripwire("detector", touches),  # type: ignore[arg-type]
            settings=settings(),
            now=datetime.now(UTC),
            plan_gate=gate,
        )

    async def test_free_reads_no_history_and_the_run_stays_put(
        self, conn: aiosqlite.Connection, touches: Touches
    ) -> None:
        await self._start(conn)
        before = await row_counts(conn)

        advanced = await self._advance(conn, touches, production_gate())

        assert advanced == 0
        assert_nothing_paid(touches, before, await row_counts(conn))
        run = await get_active_run(conn, channel_id=CHANNEL)
        assert run is not None and run.state is BackfillState.RUNNING

    async def test_positive_control_pro_reaches_the_channel(
        self, conn: aiosqlite.Connection, touches: Touches
    ) -> None:
        await self._start(conn)

        await self._advance(conn, touches, production_gate(complimentary=True))

        assert "backfill_gateway.resolve_channel" in touches.names


def interaction(conn: aiosqlite.Connection, gate: Any) -> MagicMock:
    fake = MagicMock(spec=discord.Interaction)
    fake.locale = "de"
    fake.guild_id = GUILD
    fake.user = MagicMock()
    fake.user.id = 4242
    fake.created_at = datetime.now(UTC)
    fake.client = MagicMock()
    fake.client.db = conn
    fake.client.plan_gate = gate
    fake.client.settings = settings()
    fake.response = MagicMock()
    fake.response.send_message = AsyncMock()
    fake.response.defer = AsyncMock()
    fake.response.is_done = MagicMock(return_value=False)
    fake.followup = MagicMock()
    fake.followup.send = AsyncMock()
    return fake


def text_channel() -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = CHANNEL
    channel.mention = f"<#{CHANNEL}>"
    channel.guild = MagicMock()
    channel.guild.id = GUILD
    channel.guild.me = None
    return channel


class TestEveryEnablingCommandRefusesOnFree:
    @pytest.mark.parametrize(
        "invoke",
        [
            pytest.param(
                lambda fake: config_command.callback(fake, text_channel(), True, None),  # type: ignore[arg-type]  # pyright: ignore
                id="config-proactive-on",
            ),
            pytest.param(
                lambda fake: config_command.callback(fake, text_channel(), None, True),  # type: ignore[arg-type]  # pyright: ignore
                id="config-extraction-on",
            ),
            pytest.param(
                lambda fake: config_command.callback(fake, text_channel(), False, True),  # type: ignore[arg-type]  # pyright: ignore
                id="config-mixed-off-and-on",
            ),
            pytest.param(
                lambda fake: digest_command.callback(fake, text_channel(), None, None),  # type: ignore[arg-type]  # pyright: ignore
                id="digest-enable",
            ),
            pytest.param(
                lambda fake: onboarding_command.callback(fake, text_channel(), None),  # type: ignore[arg-type]  # pyright: ignore
                id="onboarding-enable",
            ),
            pytest.param(
                lambda fake: backfill_start.callback(fake, text_channel(), None),  # type: ignore[arg-type]  # pyright: ignore
                id="backfill-start",
            ),
        ],
    )
    async def test_the_command_refuses_in_the_callers_language_and_writes_nothing(
        self, conn: aiosqlite.Connection, touches: Touches, invoke: Any
    ) -> None:
        fake = interaction(conn, production_gate())
        before = await row_counts(conn)

        await invoke(fake)

        fake.response.send_message.assert_awaited_once()
        args, kwargs = fake.response.send_message.call_args
        assert kwargs.get("ephemeral") is True
        # The German refusal, not the English one: the locale reached the text.
        assert "Pro" in args[0] and "This is a Pro feature" not in args[0]
        assert await row_counts(conn) == before
        assert touches.names == []


class TestFreeFeaturesNeverAskThePlan:
    """/aura-ask and manual fact management work with no subscription and a failing status source.

    /aura-ask is the one Free feature that reads the plan, and only to pick which
    daily answer cap applies (aura.db.ask_state); it is therefore checked by its
    own tests below rather than by the marker scan.
    """

    FREE_COMMAND_MODULES: Final = (
        "commands/facts.py",
        "commands/pending.py",
        "commands/supersede.py",
        "commands/links.py",
        "facts_service.py",
        "links_service.py",
    )

    @pytest.mark.parametrize("relative_path", FREE_COMMAND_MODULES)
    def test_no_free_module_references_the_plan_gate(self, relative_path: str) -> None:
        source = (SRC / relative_path).read_text(encoding="utf-8")

        for marker in ("plan_gate", "allows_pro", "aura.billing", "pro_feature_refusal"):
            assert marker not in source, f"{relative_path} consults the plan ({marker})"

    def test_aura_ask_reads_the_plan_only_to_pick_its_cap(self) -> None:
        source = (SRC / "commands/ask.py").read_text(encoding="utf-8")

        assert "pro_feature_refusal" not in source
        assert source.count("allows_pro(") == 1

    async def test_aura_ask_answers_although_the_plan_gate_would_explode(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding, touches: Touches
    ) -> None:
        with patch("aura.facts_service.generate_variants_for_fact", AsyncMock(return_value=[])):
            fact = await add_fact(
                conn,
                embedding_model,
                guild_id=GUILD,
                channel_id=11,
                message_id=101,
                content="The event starts on Saturday at 18:00.",
            )
        fake = interaction(conn, Tripwire("plan_gate", touches))
        fake.client.embedding_model = embedding_model
        fake.client.settings = settings(similarity_threshold=0.0)
        fake.command = MagicMock()
        fake.command.name = "aura-ask"
        answer = SynthesisResult(
            answer="Saturday at 18:00.", used_fact_ids=[fact.id], answers_question=True
        )

        with patch("aura.commands.ask.synthesize_answer", AsyncMock(return_value=answer)):
            await ask_command.callback(fake, "When does the event start?")  # type: ignore[call-arg, arg-type]  # pyright: ignore

        fake.followup.send.assert_awaited_once()
        assert "embed" in fake.followup.send.call_args.kwargs
        # The one touch is the cap lookup, which falls back to the Free caps.
        assert touches.names == ["plan_gate.allows_pro"]


class TestVariantGenerationIsNotGated:
    """Finding F-09 (a scope decision): paid variant generation runs for Free guilds.

    Manual fact entry is Free, and every manually added fact schedules
    `generate_variants_for_fact` -- two LLM calls per fact, charged to the
    `variant_calls` ledger (VARIANT_DAILY_CAP, default 200 per guild per day).
    Nothing on that path knows the guild's plan.
    """

    async def test_adding_a_fact_on_a_free_guild_schedules_paid_variant_generation(
        self, conn: aiosqlite.Connection, embedding_model: TextEmbedding
    ) -> None:
        gate = production_gate()
        assert gate.allows_pro(GUILD) is False
        generate = AsyncMock(return_value=[])

        with patch("aura.facts_service.generate_variants_for_fact", generate):
            await add_fact(
                conn,
                embedding_model,
                guild_id=GUILD,
                channel_id=11,
                message_id=101,
                content="The event starts on Saturday at 18:00.",
            )
            await asyncio.sleep(0)

        generate.assert_awaited_once()


# Every function whose call spends money or runs a Pro trigger, and the exact set
# of modules allowed to call it. A new call site makes the structural test below
# fail, which is the point: it must be classified as Free or gated before it ships.
PAID_ENTRY_POINTS: Final = {
    "synthesize_answer": {"commands/ask.py", "proactive/responder.py"},
    "verify_answer_grounded": {"commands/ask.py", "proactive/responder.py"},
    "distill_facts": {"extraction/pipeline.py", "backfill/worker.py"},
    "stage_distilled_candidates": {"extraction/pipeline.py", "backfill/worker.py"},
    "judge_relationship": {"extraction/pipeline.py"},
    "generate_variants_for_fact": {"facts_service.py"},
    "build_digest": {"digest/scheduler.py"},
    "build_onboarding_content": {"onboarding/listener.py"},
}


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_every_paid_entry_point_is_called_only_from_its_known_sites() -> None:
    observed: dict[str, set[str]] = {name: set() for name in PAID_ENTRY_POINTS}
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(SRC))
        for name in _called_names(path) & set(PAID_ENTRY_POINTS):
            observed[name].add(relative)

    assert observed == PAID_ENTRY_POINTS


def test_every_background_task_the_client_starts_is_given_the_plan_gate() -> None:
    tree = ast.parse((SRC / "main.py").read_text(encoding="utf-8"))
    started: dict[str, bool] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_task"
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
        ):
            inner = node.args[0]
            assert isinstance(inner.func, ast.Name)
            started[inner.func.id] = any(keyword.arg == "plan_gate" for keyword in inner.keywords)

    assert started == {
        "run_extraction_sweeper": True,
        "run_digest_scheduler": True,
        "run_backfill_worker": True,
    }
