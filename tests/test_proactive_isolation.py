"""Where the proactive path may and may not reach an LLM, and that it never writes a fact.

Phase 2a-3 is the moment the earlier phases' promise -- "this path reaches no
LLM at all" -- is deliberately broken, in exactly ONE place: the responder. So
this file no longer claims the whole path is LLM-free; it claims the break is
confined to where it belongs and that the read-only guarantee is total. Three
claims, asserted separately:

1. **The free gate path stays LLM-free.** The scorer, the gate, the budget, the
   diagnostics and the debug command reach no LLM, no HTTP client, no socket.
   The listener routes an eligible message onward but does not itself import a
   synthesis function -- it delegates.
2. **The break is confined to the responder.** The responder is the single
   sanctioned place that reaches synthesis, and even it writes no fact.
3. **The knowledge model is read-only across the whole path.** Stage 2 and the
   responder rank/read a guild's facts; creating, updating, superseding,
   linking or deleting one is not allowed anywhere, and no statement the
   pipeline issues may do so -- proven even when the responder runs in full.

Several independent angles, because any one alone leaves a gap:

* Static: the modules' import statements are audited, so no forbidden name is
  even in scope to be called -- and the responder's sanctioned exception is
  asserted positively, so the audit cannot pass by simply excluding it.
* Per-module: the scorer specifically is held to the stricter Phase 2a-1 bar
  (no facts at all), since it is the function the rescoring changed.
* Runtime: with no LLM configured the real pipeline still reaches no litellm
  and opens no socket, and with one configured it writes no fact.
* Data: every SQL statement the pipeline issues is captured and inspected, so
  the claim rests on what the database actually saw.
"""
from __future__ import annotations

import ast
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import discord
import numpy as np
import pytest
from fastembed import TextEmbedding

import aura.embeddings
import aura.facts_service
import aura.synthesis
from aura.config import Settings
from aura.db.proactive_channel_config import set_channel_enabled
from aura.db.proactive_signals import get_recent_signals
from aura.db.repository import get_active_facts, init_schema
from aura.facts_service import add_fact
from aura.proactive.gate import ProactiveGateConfig, evaluate_message
from aura.proactive.grace import GraceRegistry
from aura.proactive.listener import handle_message
from aura.proactive.question_detector import QuestionDetector

GUILD_A = 100000000000000001
CHANNEL = 555

CONFIG = ProactiveGateConfig(
    question_threshold=-2.0 + 1e-9,  # everything passes Stage 1, so the path runs in full
    similarity_threshold=-1.0,  # and Stage 2 too, so the budget is reached as well
    minimum_confidence_gap=0.0,
    cooldown_seconds=900.0,
    daily_cap=20,
)


def _unconfigured_settings() -> Settings:
    """No LLM: the responder short-circuits, so the whole path stays LLM-free."""
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        discord_token="fake-token",
        llm_api_key=None,
        synthesis_model=None,
        proactive_model=None,
        proactive_grace_period_seconds=0.0,
    )


def _configured_settings() -> Settings:
    """A proactive-capable LLM, for the runtime test that lets the responder run in full."""
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        discord_token="fake-token",
        llm_api_key="fake-key",
        synthesis_model="openrouter/fake/model",
        proactive_model=None,
        proactive_grace_period_seconds=0.0,
    )

_SOURCE_ROOT = Path(aura.embeddings.__file__).parent

# Word boundaries, not a substring search: the fact columns include
# "created_at" and "superseded_at", so a naive `"CREATE" in statement` reads
# every ordinary SELECT as a schema change and the check quietly inverts.
_MUTATING_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE)\b", re.IGNORECASE
)

# Every module this sub-phase adds or routes a message through.
_PHASE_MODULES = (
    "proactive/__init__.py",
    "proactive/question_detector.py",
    "proactive/gate.py",
    "proactive/listener.py",
    "proactive/grace.py",
    "db/proactive_signals.py",
    "db/proactive_state.py",
    "commands/proactive.py",
)

# Reaching any of these means this phase can cost money or answer a user.
_FORBIDDEN_MODULES = frozenset(
    {"litellm", "aura.synthesis", "aura.facts_service", "openai", "anthropic", "httpx", "requests"}
)

# Reaching any of these means this phase can *change* the knowledge model.
# Reads are absent from this list on purpose: Stage 2 ranks a guild's active
# facts, which is the whole point of the second gate.
_FORBIDDEN_WRITE_NAMES = frozenset(
    {
        "create_fact",
        "supersede_fact",
        "link_facts",
        "add_fact",
        "synthesize_answer",
        "acompletion",
    }
)

# The knowledge-model WRITERS specifically -- a subset of _FORBIDDEN_WRITE_NAMES
# that excludes the LLM names, so the responder (which legitimately reaches
# synthesis) can still be held to "writes no fact".
_FORBIDDEN_FACT_WRITERS = frozenset(
    {"create_fact", "supersede_fact", "link_facts", "add_fact"}
)

# The stricter bar Phase 2a-1 applied to everything, still applied to the
# scorer alone: it answers one question about one string and has no business
# knowing that facts exist.
_FORBIDDEN_READ_NAMES = frozenset(
    {"find_similar_facts", "get_active_facts", "get_linked_facts"}
)

# The only module allowed to read the knowledge model, so that permission
# cannot spread quietly to the scorer, the listener, or the diagnostics.
_FACT_READING_MODULE = "proactive/gate.py"


@pytest.fixture
async def conn():
    connection = await aiosqlite.connect(":memory:")
    await init_schema(connection)
    yield connection
    await connection.close()


@pytest.fixture(scope="session")
async def detector(embedding_model: TextEmbedding) -> QuestionDetector:
    return await QuestionDetector.create(embedding_model)


class _MatchingModel:
    """Every fact matches every query, so the pipeline always runs to the end."""

    def embed(self, documents: list[str], **_kwargs: object):
        for _ in documents:
            yield np.ones(4, dtype=np.float32)


async def _seed_fact(conn: aiosqlite.Connection) -> None:
    await add_fact(
        conn,
        _MatchingModel(),  # type: ignore[arg-type]
        guild_id=GUILD_A,
        channel_id=1,
        message_id=1,
        content="The server rules are in the welcome channel.",
    )


def _make_message(*, content: str = "where can I find the server rules?") -> MagicMock:
    message = MagicMock(spec=discord.Message)
    message.content = content
    message.guild = MagicMock()
    message.guild.id = GUILD_A
    message.channel = MagicMock()
    message.channel.id = 555
    message.id = 777
    message.author = MagicMock()
    message.author.bot = False
    message.webhook_id = None
    message.interaction_metadata = None
    message.type = discord.MessageType.default
    return message


async def _run_pipeline(
    conn: aiosqlite.Connection,
    detector: QuestionDetector,
    *,
    message: MagicMock | None = None,
    settings: Settings | None = None,
) -> None:
    msg = message if message is not None else _make_message()
    # The channel-enabled gate is first now, so opt the channel in before
    # running -- otherwise the pipeline short-circuits and every assertion below
    # would pass vacuously.
    if msg.guild is not None:
        await set_channel_enabled(
            conn, guild_id=msg.guild.id, channel_id=msg.channel.id, enabled=True, updated_by_id=1
        )
    await handle_message(
        msg,
        db=conn,
        detector=detector,
        model=_MatchingModel(),  # type: ignore[arg-type]
        config=CONFIG,
        settings=settings if settings is not None else _unconfigured_settings(),
        grace_registry=GraceRegistry(),
    )


def _imports_of(relative_path: str) -> list[tuple[str, str]]:
    """Return every (module, imported_name) pair in a source file's import statements."""
    tree = ast.parse((_SOURCE_ROOT / relative_path).read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pairs.extend((alias.name, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            pairs.extend((module, alias.name) for alias in node.names)
    return pairs


class TestStaticImportSurface:
    """Nothing forbidden is even in scope, so nothing forbidden can be called."""

    @pytest.mark.parametrize("relative_path", _PHASE_MODULES)
    def test_no_llm_or_http_module_is_imported(self, relative_path: str) -> None:
        offending = [
            module
            for module, _ in _imports_of(relative_path)
            if module.split(".")[0] in _FORBIDDEN_MODULES or module in _FORBIDDEN_MODULES
        ]
        assert offending == [], f"{relative_path} imports {offending}"

    @pytest.mark.parametrize("relative_path", _PHASE_MODULES)
    def test_no_knowledge_model_writer_is_imported(self, relative_path: str) -> None:
        offending = [
            name for _, name in _imports_of(relative_path) if name in _FORBIDDEN_WRITE_NAMES
        ]
        assert offending == [], f"{relative_path} imports {offending}"

    @pytest.mark.parametrize(
        "relative_path", [path for path in _PHASE_MODULES if path != _FACT_READING_MODULE]
    )
    def test_only_the_gate_may_even_read_the_knowledge_model(self, relative_path: str) -> None:
        # Stage 2 has to read facts; nothing else on this path does. Keeping
        # that permission to one module is what makes "the scorer cannot see
        # the knowledge model" a structural fact rather than a convention.
        offending = [
            name for _, name in _imports_of(relative_path) if name in _FORBIDDEN_READ_NAMES
        ]
        assert offending == [], f"{relative_path} imports {offending}"

    def test_the_scorer_still_reaches_nothing_at_all(self) -> None:
        # Re-derived rather than assumed after the contrastive rescoring: the
        # change was to how question_likeness scores, and it would have been
        # easy to reach for a fact or a second model along the way.
        imports = _imports_of("proactive/question_detector.py")
        modules = {module for module, _ in imports}
        names = {name for _, name in imports}

        assert not names & (_FORBIDDEN_READ_NAMES | _FORBIDDEN_WRITE_NAMES)
        assert not any(module.split(".")[0] in _FORBIDDEN_MODULES for module in modules)
        # CLAUDE.md's testing principle, enforced structurally rather than by
        # convention: the scoring logic must be verifiable without a Discord
        # connection existing anywhere near it.
        assert not any(module.split(".")[0] == "discord" for module in modules)
        # And no database at all -- it scores a string, it does not persist.
        assert not any(module.split(".")[0] in {"aiosqlite", "sqlite3"} for module in modules)

    def test_the_audit_would_actually_catch_a_violation(self) -> None:
        # A test that can only pass is worth nothing. aura.commands.ask is a
        # module that genuinely does reach synthesis, so it must fail the
        # audits above -- proving they discriminate.
        ask_imports = _imports_of("commands/ask.py")
        assert any(name in _FORBIDDEN_WRITE_NAMES for _, name in ask_imports)
        assert any(name in _FORBIDDEN_READ_NAMES for _, name in ask_imports)

    def test_the_responder_is_the_one_sanctioned_place_that_reaches_synthesis(self) -> None:
        # Phase 2a-3's deliberate break, asserted positively so the LLM-free
        # audit over the gate path above cannot be passing merely because the
        # responder happens to be excluded from _PHASE_MODULES.
        names = {name for _, name in _imports_of("proactive/responder.py")}
        assert "synthesize_answer" in names  # it DOES reach synthesis, on purpose
        # ...and even so it reaches no knowledge-model writer.
        assert not (names & _FORBIDDEN_FACT_WRITERS), names & _FORBIDDEN_FACT_WRITERS

    def test_the_listener_delegates_synthesis_rather_than_importing_it(self) -> None:
        # The listener routes an eligible message onward; keeping the break in
        # exactly one module means the listener must not pull synthesis into its
        # own namespace.
        names = {name for _, name in _imports_of("proactive/listener.py")}
        assert "synthesize_answer" not in names
        assert "acompletion" not in names


class TestRuntimeCallSurface:
    async def test_the_full_pipeline_reaches_no_llm_without_a_configured_model(
        self, conn: aiosqlite.Connection, detector: QuestionDetector
    ) -> None:
        # With no LLM configured, even a fully eligible message reaches no
        # synthesis and no litellm: proactive relief is provably inert until an
        # operator configures a model. The tripwire on synthesize_answer patches
        # the responder's own reference (the one actually on the path), not just
        # aura.synthesis's.
        await _seed_fact(conn)
        tripwires = {
            "litellm.acompletion": patch("litellm.acompletion", side_effect=AssertionError),
            "add_fact": patch.object(aura.facts_service, "add_fact", side_effect=AssertionError),
            "synthesize_answer": patch(
                "aura.proactive.responder.synthesize_answer", side_effect=AssertionError
            ),
        }

        started = {name: cm.start() for name, cm in tripwires.items()}
        try:
            await _run_pipeline(conn, detector, settings=_unconfigured_settings())
        finally:
            for cm in tripwires.values():
                cm.stop()

        for name, mock in started.items():
            assert mock.call_count == 0, f"{name} was reached"

    async def test_the_pipeline_ran_to_the_end_so_the_tripwires_mean_something(
        self, conn: aiosqlite.Connection, detector: QuestionDetector
    ) -> None:
        # Without this, every assertion above would also pass on a pipeline
        # that silently did nothing at all.
        await _seed_fact(conn)

        await _run_pipeline(conn, detector)

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)
        assert signal.would_escalate is True  # reached the very last gate
        async with conn.execute("SELECT COUNT(*) FROM proactive_escalations") as cursor:
            assert await cursor.fetchone() == (1,)

    async def test_evaluation_opens_no_network_socket(
        self, conn: aiosqlite.Connection, detector: QuestionDetector
    ) -> None:
        # The detector is already built (session fixture), so this covers the
        # per-message path exactly as it runs in production. Any outbound
        # connection attempt becomes an immediate, unmissable failure.
        await _seed_fact(conn)

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("evaluation attempted to open a socket")

        with patch.object(socket, "socket", refuse):
            await _run_pipeline(conn, detector)

        # Proves the message really was processed rather than skipped by a
        # filter, which would have made the assertion above vacuous.
        assert len(await get_recent_signals(conn, guild_id=GUILD_A, limit=10)) == 1

    async def test_the_gate_itself_opens_no_socket_either(
        self, conn: aiosqlite.Connection, detector: QuestionDetector
    ) -> None:
        # Called directly, so a future listener-level guard cannot be what is
        # providing this guarantee.
        await _seed_fact(conn)

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("the gate attempted to open a socket")

        with patch.object(socket, "socket", refuse):
            decision = await evaluate_message(
                conn,
                _MatchingModel(),  # type: ignore[arg-type]
                detector,
                guild_id=GUILD_A,
                channel_id=1,
                message_id=1,
                content="where are the rules?",
                config=CONFIG,
                now=datetime.now(timezone.utc),
            )

        assert decision.would_escalate is True

    async def test_a_failure_inside_the_listener_does_not_fall_back_to_an_llm(
        self, conn: aiosqlite.Connection
    ) -> None:
        # The error path is the easiest place for an "answer it anyway"
        # fallback to be added later without anyone noticing.
        exploding_detector = MagicMock(spec=QuestionDetector)
        exploding_detector.question_likeness.side_effect = RuntimeError("inference failed")

        with patch("litellm.acompletion", side_effect=AssertionError) as acompletion:
            await _run_pipeline(conn, exploding_detector)

        assert acompletion.call_count == 0

    async def test_an_eligible_verdict_sends_nothing_without_a_configured_llm(
        self, conn: aiosqlite.Connection, detector: QuestionDetector
    ) -> None:
        # Eligibility is reached and, with no LLM configured, deliberately goes
        # nowhere -- the responder short-circuits before it could post. The
        # message mock would record any send.
        await _seed_fact(conn)
        message = _make_message()

        await _run_pipeline(conn, detector, message=message, settings=_unconfigured_settings())

        [signal] = await get_recent_signals(conn, guild_id=GUILD_A, limit=10)
        assert signal.would_escalate is True
        message.channel.send.assert_not_called()
        message.reply.assert_not_called()


class TestDataSurface:
    async def test_no_statement_the_pipeline_issues_modifies_the_knowledge_model(
        self, conn: aiosqlite.Connection, detector: QuestionDetector
    ) -> None:
        await _seed_fact(conn)
        statements: list[str] = []

        await conn.set_trace_callback(statements.append)
        try:
            await _run_pipeline(conn, detector)
        finally:
            # sqlite3 documents None as "disable tracing"; the stub types the
            # parameter as a plain callable and cannot express that.
            await conn.set_trace_callback(None)  # pyright: ignore[reportArgumentType]

        assert statements, "no SQL was captured; the trace callback is not working"
        for statement in statements:
            if not _MUTATING_SQL.search(statement):
                continue  # a read; Stage 2 is allowed those
            lowered = statement.lower()
            assert "facts" not in lowered or "proactive_" in lowered, statement
            assert "fact_links" not in lowered, statement

    async def test_the_pipeline_does_read_facts_which_is_what_stage_two_is_for(
        self, conn: aiosqlite.Connection, detector: QuestionDetector
    ) -> None:
        # Asserted positively so the read-only claim above is a boundary and
        # not an accident of the pipeline never getting that far.
        await _seed_fact(conn)
        statements: list[str] = []

        await conn.set_trace_callback(statements.append)
        try:
            await _run_pipeline(conn, detector)
        finally:
            await conn.set_trace_callback(None)  # pyright: ignore[reportArgumentType]

        assert any(
            "select" in statement.lower() and "from facts" in statement.lower()
            for statement in statements
        )

    async def test_the_responder_running_in_full_still_writes_no_fact(
        self, conn: aiosqlite.Connection, detector: QuestionDetector
    ) -> None:
        # The strongest form of claim 3: with an LLM configured and synthesis
        # mocked to a confident answer, the responder runs its entire path --
        # ranking facts, then posting -- and STILL issues no statement that
        # creates, updates, supersedes, links or deletes a fact.
        await _seed_fact(conn)
        message = _make_message()
        message.channel.send = AsyncMock()
        message.guild.preferred_locale = "en-US"
        statements: list[str] = []
        result = aura.synthesis.SynthesisResult(
            answer="ans", used_fact_ids=[1], answers_question=True
        )

        await conn.set_trace_callback(statements.append)
        try:
            with patch(
                "aura.proactive.responder.synthesize_answer", AsyncMock(return_value=result)
            ):
                await _run_pipeline(
                    conn, detector, message=message, settings=_configured_settings()
                )
        finally:
            await conn.set_trace_callback(None)  # pyright: ignore[reportArgumentType]

        assert message.channel.send.await_count == 1  # it really did run to a post
        for statement in statements:
            if not _MUTATING_SQL.search(statement):
                continue
            lowered = statement.lower()
            assert "facts" not in lowered or "proactive_" in lowered, statement
            assert "fact_links" not in lowered, statement

    async def test_the_knowledge_model_is_byte_for_byte_unchanged_after_a_burst(
        self, conn: aiosqlite.Connection, detector: QuestionDetector
    ) -> None:
        await _seed_fact(conn)
        async with conn.execute("SELECT * FROM facts ORDER BY id") as cursor:
            before = await cursor.fetchall()

        for message_id in range(10):
            message = _make_message(content=f"how do I do thing {message_id}?")
            message.id = message_id
            await _run_pipeline(conn, detector, message=message)

        async with conn.execute("SELECT * FROM facts ORDER BY id") as cursor:
            assert await cursor.fetchall() == before
        assert len(await get_active_facts(conn, GUILD_A)) == 1
        async with conn.execute("SELECT COUNT(*) FROM fact_links") as cursor:
            assert await cursor.fetchone() == (0,)
        assert len(await get_recent_signals(conn, guild_id=GUILD_A, limit=100)) == 10
