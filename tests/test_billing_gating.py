"""Phase 4c's gating: every Pro trigger stays silent on Free, every Free feature keeps working.

Two halves, one per kind of entry point:

  * The five Pro TRIGGERS -- proactive relief, automatic extraction (intake and
    batch flush), the digest, onboarding and backfill -- each run their real
    code against a real database with an enforced gate that says Free, and
    must do NOTHING observable: no embedding, no queue row, no claimed budget
    slot, no claimed digest window, no read of history. The same call with a
    Pro gate must get past the gate, so a test cannot pass merely because the
    path was never reachable.

  * The COMMANDS that switch a Pro trigger on must refuse on Free with a clear,
    translated message and write nothing -- while switching a feature off stays
    possible, and /aura-plan explains the state in every locale.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import pytest

from aura.backfill.history import ChannelUnreadable
from aura.backfill.worker import advance_due_backfills
from aura.billing import (
    GracePolicy,
    PlanBasis,
    PlanGate,
    Standing,
    SubscriptionRecord,
    SubscriptionStatus,
)
from aura.billing.entitlement import InvoiceStatus, decide_plan
from aura.commands.backfill import backfill_start
from aura.commands.config import config_command
from aura.commands.digest import digest_command
from aura.commands.onboarding import onboarding_command
from aura.commands.plan import describe_plan, discord_timestamp, plan_command, pro_feature_refusal
from aura.config import Settings
from aura.db.backfill_runs import BackfillState, get_active_run, start_backfill_run
from aura.db.connection import utc_day
from aura.db.digest_config import get_digest_config, set_digest_config
from aura.db.digest_state import last_covered_until
from aura.db.extraction_channel_config import set_extraction_enabled
from aura.db.extraction_queue import count_queued, enqueue_message
from aura.db.extraction_state import count_extraction_calls_on
from aura.db.onboarding_config import get_onboarding_config
from aura.db.proactive_channel_config import set_channel_enabled
from aura.db.repository import init_schema
from aura.digest.scheduler import send_due_digests
from aura.extraction.pipeline import flush_due_batches, handle_extraction_message
from aura.i18n import SUPPORTED_LOCALES, t
from aura.onboarding.listener import handle_member_join
from aura.proactive.grace import GraceRegistry
from aura.proactive.listener import _still_fresh_enough_for_synthesis, handle_message

NOW = datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)
GUILD = 100000000000000001
CHANNEL = 300000000000000003
DASHBOARD = "https://aura.example/dashboard"
POLICY = GracePolicy(renewal_grace=timedelta(hours=72), payment_failure_grace=timedelta(days=7))


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


def free_gate() -> PlanGate:
    return PlanGate(enforced=True, policy=POLICY, complimentary_guild_ids=frozenset(), records=[])


def pro_gate() -> PlanGate:
    return PlanGate(
        enforced=True, policy=POLICY, complimentary_guild_ids=frozenset({GUILD}), records=[]
    )


def llm_settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="fake-token",
        llm_api_key="fake-key",
        synthesis_model="openrouter/fake/model",
        billing_dashboard_url=DASHBOARD,
        **overrides,
    )


def human_message(content: str = "where are the rules?") -> MagicMock:
    message = MagicMock(spec=discord.Message)
    message.content = content
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
    message.created_at = NOW
    return message


class TestProactiveRelief:
    async def _run(self, conn, gate: PlanGate) -> AsyncMock:
        await set_channel_enabled(
            conn, guild_id=GUILD, channel_id=CHANNEL, enabled=True, updated_by_id=1
        )
        decision = MagicMock(would_escalate=False)
        with (
            patch(
                "aura.proactive.listener.evaluate_message", AsyncMock(return_value=decision)
            ) as evaluate,
            patch("aura.proactive.listener.record_signal", AsyncMock()),
        ):
            await handle_message(
                human_message(),
                db=conn,
                detector=MagicMock(),
                model=MagicMock(),
                config=MagicMock(),
                settings=llm_settings(),
                grace_registry=GraceRegistry(),
                plan_gate=gate,
            )
        return evaluate

    async def test_a_free_guild_is_never_evaluated(self, conn) -> None:
        evaluate = await self._run(conn, free_gate())

        evaluate.assert_not_awaited()

    async def test_a_pro_guild_is(self, conn) -> None:
        evaluate = await self._run(conn, pro_gate())

        evaluate.assert_awaited_once()

    async def test_a_plan_that_ends_during_the_grace_period_stands_the_answer_down(self) -> None:
        stood_down = await _still_fresh_enough_for_synthesis(
            MagicMock(), guild_id=GUILD, channel_id=CHANNEL, message_id=1, plan_gate=free_gate()
        )

        assert stood_down is False


class TestExtraction:
    async def test_intake_on_a_free_guild_queues_nothing_and_embeds_nothing(self, conn) -> None:
        await set_extraction_enabled(
            conn, guild_id=GUILD, channel_id=CHANNEL, enabled=True, updated_by_id=1
        )
        detector = MagicMock()
        detector.question_likeness = AsyncMock(return_value=0.9)

        await handle_extraction_message(
            human_message("The event is on Saturday."),
            db=conn,
            detector=detector,
            settings=llm_settings(),
            plan_gate=free_gate(),
        )

        detector.question_likeness.assert_not_awaited()
        assert await count_queued(conn) == 0

    async def test_intake_on_a_pro_guild_queues(self, conn) -> None:
        await set_extraction_enabled(
            conn, guild_id=GUILD, channel_id=CHANNEL, enabled=True, updated_by_id=1
        )
        detector = MagicMock()
        detector.question_likeness = AsyncMock(return_value=0.9)

        await handle_extraction_message(
            human_message("The event is on Saturday."),
            db=conn,
            detector=detector,
            settings=llm_settings(),
            plan_gate=pro_gate(),
        )

        assert await count_queued(conn) == 1

    async def _queue_a_due_batch(self, conn) -> None:
        await enqueue_message(
            conn,
            guild_id=GUILD,
            channel_id=CHANNEL,
            message_id=1,
            channel_name="general",
            content="The event is on Saturday.",
            message_created_at=NOW - timedelta(hours=1),
            now=NOW - timedelta(hours=1),
        )

    async def test_a_batch_queued_on_pro_and_flushed_on_free_is_dropped_without_spending(
        self, conn
    ) -> None:
        await self._queue_a_due_batch(conn)

        with patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=[])) as distill:
            flushed = await flush_due_batches(
                conn, MagicMock(), settings=llm_settings(), now=NOW, plan_gate=free_gate()
            )

        assert flushed == 0
        distill.assert_not_awaited()
        assert await count_queued(conn) == 0
        assert await count_extraction_calls_on(conn, guild_id=GUILD, day=utc_day(NOW)) == 0

    async def test_the_same_batch_on_pro_is_distilled(self, conn) -> None:
        await self._queue_a_due_batch(conn)

        with patch("aura.extraction.pipeline.distill_facts", AsyncMock(return_value=[])) as distill:
            await flush_due_batches(
                conn, MagicMock(), settings=llm_settings(), now=NOW, plan_gate=pro_gate()
            )

        distill.assert_awaited_once()
        assert await count_extraction_calls_on(conn, guild_id=GUILD, day=utc_day(NOW)) == 1


class TestDigest:
    async def _configure(self, conn) -> None:
        await set_digest_config(
            conn,
            guild_id=GUILD,
            channel_id=CHANNEL,
            interval_seconds=86400,
            enabled=True,
            updated_by_id=1,
        )

    async def test_a_free_guild_gets_no_digest_and_keeps_its_window_for_later(self, conn) -> None:
        await self._configure(conn)
        gateway = MagicMock()
        gateway.resolve_channel = AsyncMock()

        posted = await send_due_digests(
            conn, gateway, now=datetime.now(UTC) + timedelta(days=3), plan_gate=free_gate()
        )

        assert posted == 0
        gateway.resolve_channel.assert_not_awaited()
        assert await last_covered_until(conn, guild_id=GUILD) is None
        assert (await get_digest_config(conn, guild_id=GUILD)) is not None

    async def test_a_pro_guild_is_evaluated(self, conn) -> None:
        await self._configure(conn)

        await send_due_digests(
            conn,
            MagicMock(),
            now=datetime.now(UTC) + timedelta(days=3),
            plan_gate=pro_gate(),
        )

        # No facts, so the window is claimed as empty -- which only happens past the gate.
        assert await last_covered_until(conn, guild_id=GUILD) is not None


class TestOnboarding:
    async def _join(self, gate: PlanGate) -> AsyncMock:
        member = MagicMock(spec=discord.Member)
        member.bot = False
        member.id = 42
        member.guild = MagicMock()
        member.guild.id = GUILD
        with patch(
            "aura.onboarding.listener.get_onboarding_config", AsyncMock(return_value=None)
        ) as reader:
            await handle_member_join(
                member, db=MagicMock(), gateway=MagicMock(), settings=llm_settings(), plan_gate=gate
            )
        return reader

    async def test_a_free_guild_does_not_even_read_its_configuration(self) -> None:
        (await self._join(free_gate())).assert_not_awaited()

    async def test_a_pro_guild_does(self) -> None:
        (await self._join(pro_gate())).assert_awaited_once()


class TestBackfill:
    async def _start(self, conn) -> None:
        await start_backfill_run(
            conn,
            guild_id=GUILD,
            channel_id=CHANNEL,
            until_message_id=2**62,
            after_message_id=None,
            requested_by_id=1,
            now=NOW,
        )

    async def test_a_free_guilds_run_reads_nothing_and_stays_exactly_where_it_was(
        self, conn
    ) -> None:
        await self._start(conn)
        gateway = MagicMock()
        gateway.resolve_channel = AsyncMock()

        advanced = await advance_due_backfills(
            conn,
            MagicMock(),
            gateway,
            MagicMock(),
            settings=llm_settings(),
            now=NOW,
            plan_gate=free_gate(),
        )

        assert advanced == 0
        gateway.resolve_channel.assert_not_awaited()
        run = await get_active_run(conn, channel_id=CHANNEL)
        assert run is not None and run.state is BackfillState.RUNNING

    async def test_a_pro_guilds_run_reaches_the_channel(self, conn) -> None:
        await self._start(conn)
        gateway = MagicMock()
        gateway.resolve_channel = AsyncMock(side_effect=ChannelUnreadable("gone"))

        await advance_due_backfills(
            conn,
            MagicMock(),
            gateway,
            MagicMock(),
            settings=llm_settings(),
            now=NOW,
            plan_gate=pro_gate(),
        )

        gateway.resolve_channel.assert_awaited_once()


def interaction(
    conn, gate: PlanGate, *, locale: str = "en-US", dashboard: str | None = DASHBOARD
) -> MagicMock:
    fake = MagicMock(spec=discord.Interaction)
    fake.locale = locale
    fake.guild_id = GUILD
    fake.user = MagicMock()
    fake.user.id = 4242
    fake.client = MagicMock()
    fake.client.db = conn
    fake.client.plan_gate = gate
    fake.client.settings = (
        llm_settings() if dashboard else Settings(_env_file=None, discord_token="t")
    )  # type: ignore[call-arg]
    fake.response = MagicMock()
    fake.response.send_message = AsyncMock()
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


def reply(fake: MagicMock) -> str:
    fake.response.send_message.assert_awaited_once()
    args, kwargs = fake.response.send_message.call_args
    assert kwargs.get("ephemeral") is True
    return args[0]


async def row_count(conn, table: str) -> int:
    async with conn.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
        return (await cursor.fetchone())[0]


class TestCommandsRefuseOnFree:
    async def test_enabling_proactive_relief_is_refused_and_nothing_is_written(self, conn) -> None:
        fake = interaction(conn, free_gate())

        await config_command.callback(fake, text_channel(), True, None)  # pyright: ignore

        assert reply(fake) == t("plan_pro_required", "en-US") + "\n" + t(
            "plan_upgrade_link", "en-US", url=DASHBOARD
        )
        assert await row_count(conn, "proactive_channel_config") == 0

    async def test_a_call_mixing_off_and_on_changes_nothing_at_all(self, conn) -> None:
        fake = interaction(conn, free_gate())

        await config_command.callback(fake, text_channel(), False, True)  # pyright: ignore

        assert t("plan_pro_required", "en-US") in reply(fake)
        assert await row_count(conn, "proactive_channel_config") == 0
        assert await row_count(conn, "extraction_channel_config") == 0

    async def test_switching_features_off_never_needs_pro(self, conn) -> None:
        fake = interaction(conn, free_gate())

        await config_command.callback(fake, text_channel(), False, False)  # pyright: ignore

        assert t("plan_pro_required", "en-US") not in reply(fake)
        assert await row_count(conn, "proactive_channel_config") == 1

    async def test_enabling_the_digest_is_refused(self, conn) -> None:
        fake = interaction(conn, free_gate())

        await digest_command.callback(fake, text_channel(), None, None)  # pyright: ignore

        assert t("plan_pro_required", "en-US") in reply(fake)
        assert await get_digest_config(conn, guild_id=GUILD) is None

    async def test_turning_an_existing_digest_off_on_free_is_allowed(self, conn) -> None:
        await set_digest_config(
            conn,
            guild_id=GUILD,
            channel_id=CHANNEL,
            interval_seconds=86400,
            enabled=True,
            updated_by_id=1,
        )
        fake = interaction(conn, free_gate())

        await digest_command.callback(fake, None, None, False)  # pyright: ignore

        assert reply(fake) == t("digest_disabled", "en-US")
        config = await get_digest_config(conn, guild_id=GUILD)
        assert config is not None and config.digest_enabled is False

    async def test_enabling_onboarding_is_refused(self, conn) -> None:
        fake = interaction(conn, free_gate())

        await onboarding_command.callback(fake, text_channel(), None)  # pyright: ignore

        assert t("plan_pro_required", "en-US") in reply(fake)
        assert await get_onboarding_config(conn, guild_id=GUILD) is None

    async def test_starting_a_backfill_is_refused_before_anything_else_is_checked(
        self, conn
    ) -> None:
        fake = interaction(conn, free_gate())

        await backfill_start.callback(fake, text_channel(), None)  # pyright: ignore

        assert t("plan_pro_required", "en-US") in reply(fake)
        assert await get_active_run(conn, channel_id=CHANNEL) is None

    async def test_on_pro_the_same_command_proceeds(self, conn) -> None:
        fake = interaction(conn, pro_gate())

        await config_command.callback(fake, text_channel(), True, None)  # pyright: ignore

        assert t("plan_pro_required", "en-US") not in reply(fake)
        assert await row_count(conn, "proactive_channel_config") == 1

    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    async def test_the_refusal_is_translated_in_every_locale(self, conn, locale: str) -> None:
        refusal = pro_feature_refusal(interaction(conn, free_gate(), locale=locale))

        assert refusal is not None
        assert "[" not in refusal.split("\n")[0]
        assert refusal.split("\n")[0] == t("plan_pro_required", locale)
        assert DASHBOARD in refusal

    async def test_without_a_dashboard_url_the_refusal_still_says_what_happened(self, conn) -> None:
        refusal = pro_feature_refusal(interaction(conn, free_gate(), dashboard=None))

        assert refusal == t("plan_pro_required", "en-US")


def record(**overrides: object) -> SubscriptionRecord:
    values: dict[str, object] = {
        "subscription_id": "sub_A",
        "guild_id": GUILD,
        "customer_id": "cus_A",
        "purchaser_user_id": 5000,
        "status": SubscriptionStatus.ACTIVE,
        "cancel_at_period_end": False,
        "cancel_at": None,
        "collection_paused": False,
        "latest_invoice_status": InvoiceStatus.PAID,
        "current_period_start": NOW - timedelta(days=1),
        "current_period_end": NOW + timedelta(days=29),
        "livemode": False,
        "version": 1,
        "confirmed_at": NOW,
    }
    values.update(overrides)
    return SubscriptionRecord(**values)  # type: ignore[arg-type]


def plan(records: list[SubscriptionRecord], *, enforced: bool = True, complimentary: bool = False):
    return decide_plan(
        guild_id=GUILD,
        records=records,
        now=NOW,
        policy=POLICY,
        enforced=enforced,
        complimentary=complimentary,
    )


class TestPlanDescription:
    @pytest.mark.parametrize("locale", sorted(SUPPORTED_LOCALES))
    @pytest.mark.parametrize(
        "records, expected_key",
        [
            ([], "plan_state_no_subscription"),
            ([record(status=SubscriptionStatus.CANCELED)], "plan_state_ended"),
            ([record()], "plan_state_active"),
            (
                [
                    record(
                        current_period_end=NOW - timedelta(hours=1),
                        current_period_start=NOW - timedelta(days=30),
                    )
                ],
                "plan_state_renewal_pending",
            ),
            ([record(cancel_at_period_end=True)], "plan_state_canceling"),
            (
                [
                    record(
                        status=SubscriptionStatus.PAST_DUE,
                        current_period_start=NOW - timedelta(days=2),
                    )
                ],
                "plan_state_payment_grace",
            ),
        ],
    )
    def test_every_standing_is_described_in_every_locale(
        self, records, expected_key: str, locale: str
    ) -> None:
        text = describe_plan(plan(records), locale=locale, dashboard_url=DASHBOARD)

        # Every literal fragment of the expected template, around its {date},
        # must be in the reply -- which holds wherever a locale puts the date.
        template = t(expected_key, locale, date="\x00").split("\x00")
        assert all(fragment in text for fragment in template if fragment)
        assert "[plan_" not in text
        assert "{date}" not in text and "{url}" not in text and "{count}" not in text
        assert DASHBOARD in text

    def test_the_active_standing_shows_the_paid_through_date_as_a_discord_timestamp(self) -> None:
        text = describe_plan(plan([record()]), locale="en-US", dashboard_url=None)

        assert discord_timestamp(NOW + timedelta(days=29)) in text

    def test_payment_grace_shows_the_date_pro_ends(self) -> None:
        failing = record(
            status=SubscriptionStatus.PAST_DUE, current_period_start=NOW - timedelta(days=2)
        )

        text = describe_plan(plan([failing]), locale="en-US", dashboard_url=None)

        assert discord_timestamp(NOW - timedelta(days=2) + timedelta(days=7)) in text
        assert "⚠️" in text

    def test_free_lists_what_free_still_includes(self) -> None:
        text = describe_plan(plan([]), locale="de", dashboard_url=None)

        assert t("plan_free_includes", "de") in text

    def test_two_paying_subscriptions_are_pointed_out(self) -> None:
        text = describe_plan(
            plan([record(subscription_id="sub_1"), record(subscription_id="sub_2")]),
            locale="en-US",
            dashboard_url=None,
        )

        assert t("plan_multiple_subscriptions", "en-US", count=2) in text

    def test_billing_not_enforced_says_so_and_nothing_about_free(self) -> None:
        text = describe_plan(plan([], enforced=False), locale="en-US", dashboard_url=None)

        assert text == t("plan_state_not_enforced", "en-US")

    def test_complimentary_says_so(self) -> None:
        described = plan([], complimentary=True)

        assert described.basis is PlanBasis.COMPLIMENTARY
        assert describe_plan(described, locale="en-US", dashboard_url=None) == t(
            "plan_state_complimentary", "en-US"
        )


class TestPlanCommand:
    async def test_replies_ephemerally_with_the_description(self, conn) -> None:
        fake = interaction(conn, free_gate())

        await plan_command.callback(fake)  # pyright: ignore

        assert t("plan_state_no_subscription", "en-US") in reply(fake)

    def test_is_moderator_only_and_guild_only(self) -> None:
        from discord import app_commands

        with pytest.raises(app_commands.MissingPermissions):
            for check in plan_command.checks:
                check(MagicMock(permissions=discord.Permissions(manage_guild=False)))
        assert plan_command.guild_only is True

    def test_the_standing_enum_is_fully_covered_by_translations(self) -> None:
        from aura.commands.plan import _STANDING_KEYS

        assert set(_STANDING_KEYS) == set(Standing)
