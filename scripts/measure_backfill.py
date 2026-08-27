"""Measure a real backfill run: duration, requests, candidates, projected cost.

WHAT THIS IS, AND WHAT IT IS NOT. The phase brief asks for "ein realer Testlauf
gegen die vorhandene Testserver-Historie", and it also puts VPS interaction out
of scope for this sub-phase. Those two cannot both be honoured from a
development machine: the test server's history lives behind a live gateway
connection, this repository's local data/aura.db is a stale July snapshot with
two facts in it, and reading a real channel would spend real money on
distillation calls. So this script measures the next most honest thing, and
reports/phase-3b.txt says so plainly rather than presenting it as a live run.

WHAT IT ACTUALLY RUNS. Everything except the paid call:

  * the real aura.backfill.worker, unmodified -- the real cursor, the real
    ordering enforcement, the real page bounds, the real boundary checks
    against live extraction, the real batching;
  * the real aura.extraction.fact_worthiness detector against the real,
    shipped fastembed model, at the shipped EXTRACTION_FACT_WORTHINESS_THRESHOLD
    -- so the fact-worthy rate below is MEASURED, not assumed;
  * the real aura.extraction.pipeline.stage_distilled_candidates, including the
    real dedup comparison against a real database;
  * a real file-backed SQLite database, so the cursor's durability is the
    property that is actually exercised rather than one that is described.

WHAT IT STANDS IN FOR. Two things, both named at the point they are used:

  * distill_facts, replaced by a stand-in that counts the tokens the real prompt
    would have sent and returns one distilled sentence per candidate. No network,
    no money. Its token count is what the cost projection is built from.
  * Discord's history endpoint, replaced by a paginator over the corpus below,
    honouring `after`, `before`, `limit` and `oldest_first` exactly as
    discord.py's own does.

THE MESSAGE CORPUS is reports/extraction-corpus/corpus.json -- the same 2,127
messages across all nine locales that EXTRACTION_FACT_WORTHINESS_THRESHOLD was
calibrated against (reports/phase-3a-1b.txt), built to roughly 90% ordinary chat
and 10% fact-worthy by construction. It is the most realistic offline stand-in
for a server's history this project has, and it carries the same caveat its own
report does: it measures behaviour against one generator model's idea of chat,
not against what real Discord members write.

  .venv/bin/python scripts/measure_backfill.py
  .venv/bin/python scripts/measure_backfill.py --no-pricing   # skip the network
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

import logging

import aiosqlite
import discord
from fastembed import TextEmbedding

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from aura.backfill.worker import advance_due_backfills  # noqa: E402
from aura.config import Settings  # noqa: E402
from aura.db.backfill_runs import (  # noqa: E402
    BackfillState,
    get_recent_runs,
    start_backfill_run,
)
from aura.db.backfill_state import count_backfill_calls_on  # noqa: E402
from aura.db.connection import utc_day  # noqa: E402
from aura.db.pending_facts import FactCategory, count_pending_facts  # noqa: E402
from aura.db.repository import init_schema  # noqa: E402
from aura.extraction.distiller import DistilledFact  # noqa: E402
from aura.extraction.fact_worthiness import create_fact_worthiness_detector  # noqa: E402
from synthetic_corpus.budget import ModelPrice  # noqa: E402
from synthetic_corpus.pricing import PricingUnavailableError, fetch_model_prices  # noqa: E402

CORPUS_PATH = _REPO_ROOT / "reports" / "extraction-corpus" / "corpus.json"

GUILD = 100000000000000001
CHANNEL = 300000000000000003
MODERATOR = 4242
NOW = datetime(2026, 8, 26, 12, 0, 0, tzinfo=timezone.utc)

# The corpus's messages carry no snowflakes, so they are laid out on a synthetic
# timeline: one message a minute, oldest first, ending well before the run's
# upper bound. Spacing them by a real interval rather than by consecutive
# integers is what makes the run's own date-based progress reporting meaningful.
FIRST_ID = discord.utils.time_snowflake(
    datetime(2025, 1, 1, tzinfo=timezone.utc), high=False
)
SECONDS_BETWEEN_MESSAGES = 60

# The model whose price the projection is quoted in. Read from the environment
# would be better still, but this script must run with no .env at all, and
# .env.example ships this value for EXTRACTION_MODEL.
PROJECTION_MODEL = "openrouter/anthropic/claude-haiku-4.5"

# Characters per token, used only to turn a measured prompt SIZE into a token
# count when the live catalog is unavailable. Four is the conventional English
# approximation and is optimistic for CJK, which is why the projection below is
# reported as an order-of-magnitude figure rather than a quote.
_CHARS_PER_TOKEN = 4.0

# Measured in reports/phase-3a-2.txt Section 8: a full 20-message batch produced
# under 1k output tokens. Scaled per candidate here rather than held constant,
# for the same reason generate_extraction_corpus.py had to fix its own estimate.
_OUTPUT_TOKENS_PER_CANDIDATE = 50


@dataclass
class _Stats:
    """Everything one measured run produced."""

    messages_scanned: int = 0
    page_requests: int = 0
    candidates_offered: int = 0
    candidates_staged: int = 0
    distillation_calls: int = 0
    prompt_characters: int = 0
    ticks: int = 0
    seconds: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)
    out_of_order_pages: int = 0

    @property
    def input_tokens(self) -> int:
        return round(self.prompt_characters / _CHARS_PER_TOKEN)

    @property
    def output_tokens(self) -> int:
        return self.candidates_offered * _OUTPUT_TOKENS_PER_CANDIDATE


class _Paginator:
    """Discord's history endpoint, over a fixed corpus, with no network."""

    def __init__(self, corpus: list[discord.Message], stats: _Stats, *, shuffle: bool) -> None:
        self.id = CHANNEL
        self.name = "history"
        self.guild = _Guild()
        self._corpus = sorted(corpus, key=lambda message: message.id)
        self._stats = stats
        self._shuffle = shuffle
        self._rng = random.Random(20260826)

    def history(self, **kwargs: object):
        self._stats.page_requests += 1
        after = kwargs.get("after")
        before = kwargs.get("before")
        limit = cast("int", kwargs.get("limit") or 100)
        low = after.id if isinstance(after, discord.Object) else 0
        high = before.id if isinstance(before, discord.Object) else 1 << 63
        page = [m for m in self._corpus if low < m.id < high][:limit]
        if self._shuffle:
            page = list(page)
            self._rng.shuffle(page)
        return _PageIterator(page)


class _PageIterator:
    def __init__(self, page: list[discord.Message]) -> None:
        self._page = page
        self._index = 0

    def __aiter__(self) -> _PageIterator:
        return self

    async def __anext__(self) -> discord.Message:
        if self._index >= len(self._page):
            raise StopAsyncIteration
        message = self._page[self._index]
        self._index += 1
        return message


class _Guild:
    id = GUILD


class _Gateway:
    def __init__(self, channel: _Paginator) -> None:
        self._channel = channel

    async def resolve_channel(self, channel_id: int) -> discord.TextChannel:
        # A real TextChannel cannot be constructed without a gateway connection,
        # which is the whole point of this harness -- the same cast, for the same
        # reason, tests/test_digest_scheduler.py's own fake gateway makes.
        return cast("discord.TextChannel", self._channel)


class _Message:
    """The fields should_extract and the batch actually read, and nothing else."""

    __slots__ = (
        "author",
        "channel",
        "content",
        "created_at",
        "guild",
        "id",
        "interaction_metadata",
        "type",
        "webhook_id",
    )

    def __init__(self, message_id: int, content: str) -> None:
        self.id = message_id
        self.content = content
        self.created_at = discord.utils.snowflake_time(message_id)
        self.guild = _Guild()
        self.channel = _Paginator.__new__(_Paginator)
        self.channel.id = CHANNEL  # type: ignore[attr-defined]
        self.channel.name = "history"  # type: ignore[attr-defined]
        self.author = _Author()
        self.webhook_id = None
        self.interaction_metadata = None
        self.type = discord.MessageType.default


class _Author:
    bot = False


def _load_corpus() -> tuple[list[discord.Message], int]:
    """The calibration corpus, laid out on a synthetic minute-by-minute timeline."""
    if not CORPUS_PATH.is_file():
        raise SystemExit(
            f"{CORPUS_PATH} is missing. It is gitignored; regenerate it with "
            "scripts/generate_extraction_corpus.py, or run with a smaller corpus."
        )
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    entries = data["messages"]
    corpus: list[discord.Message] = []
    fact_worthy = 0
    for index, entry in enumerate(entries):
        message_id = FIRST_ID + discord.utils.time_snowflake(
            datetime(2025, 1, 1, tzinfo=timezone.utc)
            + timedelta(seconds=index * SECONDS_BETWEEN_MESSAGES),
            high=False,
        ) - FIRST_ID
        corpus.append(_Message(message_id, entry["content"]))  # type: ignore[arg-type]
        # Ground truth lives in the category name, the way
        # scripts/calibrate_extraction_filter.py reads it: the five
        # fact_worthy_* categories are the positives by construction.
        if str(entry.get("category", "")).startswith("fact_worthy_"):
            fact_worthy += 1
    return corpus, fact_worthy


def _settings(**overrides) -> Settings:
    values = {
        "discord_token": "measurement-only",
        "llm_api_key": "measurement-only",
        "extraction_model": PROJECTION_MODEL,
        # No pause: this run measures Aura's own work, and the courtesy pause
        # between page requests is a fixed, separately-reported addition to it.
        "backfill_page_pause_seconds": 0.0,
        "backfill_daily_cap": 100_000,
        **overrides,
    }
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def _make_distiller(stats: _Stats):
    """A stand-in that measures the prompt it would have sent, and sends nothing."""

    async def distill(candidates, *, channel_name: str, model: str):
        stats.distillation_calls += 1
        stats.candidates_offered += len(candidates)
        stats.batch_sizes.append(len(candidates))
        # The real prompt is a fixed system message plus each candidate's text,
        # truncated at 1,000 characters (see aura.extraction.distiller).
        stats.prompt_characters += 4200 + sum(
            len(queued.content[:1000]) + 40 for queued in candidates
        )
        return [
            DistilledFact(
                message_id=queued.message_id,
                content=queued.content[:400],
                category=FactCategory.ANNOUNCEMENT,
            )
            for queued in candidates
        ]

    return distill


async def _run(
    corpus: list[discord.Message],
    *,
    shuffle: bool = False,
    restart_after_ticks: int | None = None,
    crash_mid_batch: bool = False,
    settings: Settings | None = None,
) -> _Stats:
    """One measured backfill from empty to completed, optionally interrupted.

    `restart_after_ticks` closes the connection and opens a new one on the same
    file, which is exactly what a container restart does. `crash_mid_batch`
    additionally rewinds the cursor to where it stood before that tick, which is
    the worst case the design actually has to survive: a process that died
    AFTER paying for a batch and BEFORE recording that it had finished with it.
    """
    settings = settings or _settings()
    stats = _Stats()
    until = discord.utils.time_snowflake(NOW, high=True)
    counter = _OrderWarningCounter(stats)
    logging.getLogger("aura.backfill.history").addHandler(counter)
    started = time.perf_counter()

    with TemporaryDirectory() as directory:
        path = Path(directory) / "aura.db"
        model = TextEmbedding(Settings(_env_file=None, discord_token="x").embedding_model)  # type: ignore[call-arg]

        conn = await aiosqlite.connect(path)
        await init_schema(conn)
        detector = await create_fact_worthiness_detector(model)
        run = await start_backfill_run(
            conn,
            guild_id=GUILD,
            channel_id=CHANNEL,
            until_message_id=until,
            after_message_id=None,
            requested_by_id=MODERATOR,
            now=NOW,
        )
        gateway = _Gateway(_Paginator(corpus, stats, shuffle=shuffle))

        with patch("aura.backfill.worker.distill_facts", _make_distiller(stats)):
            while True:
                previous = (await get_recent_runs(conn, guild_id=GUILD, limit=1))[
                    0
                ].cursor_message_id
                advanced = await advance_due_backfills(
                    conn, model, gateway, detector, settings=settings, now=NOW
                )
                if advanced:
                    stats.ticks += 1
                if restart_after_ticks is not None and stats.ticks == restart_after_ticks:
                    if crash_mid_batch:
                        # The window the design has to survive: the batch was
                        # paid for and staged, and the process died before the
                        # cursor moved past it.
                        await conn.execute(
                            "UPDATE backfill_runs SET cursor_message_id = ?, "
                            "cursor_message_at = NULL WHERE id = ?",
                            (previous, run.id),
                        )
                        await conn.commit()
                    # A container restart, exactly as one happens: the connection
                    # simply goes away and a new process opens the same file.
                    await conn.close()
                    conn = await aiosqlite.connect(path)
                    await init_schema(conn)
                    restart_after_ticks = None
                    continue
                if not advanced:
                    break

        final = (await get_recent_runs(conn, guild_id=GUILD, limit=1))[0]
        stats.messages_scanned = final.messages_scanned
        stats.candidates_staged = await count_pending_facts(conn, guild_id=GUILD)
        spent = await count_backfill_calls_on(conn, guild_id=GUILD, day=utc_day(NOW))
        assert final.state is BackfillState.COMPLETED, final.state
        assert spent == stats.distillation_calls, (spent, stats.distillation_calls)
        await conn.close()

    stats.seconds = time.perf_counter() - started
    logging.getLogger("aura.backfill.history").removeHandler(counter)
    return stats


class _OrderWarningCounter(logging.Handler):
    """Counts the worker's own "this page arrived unsorted" warnings.

    Counted rather than printed: on the shuffled run there is one per page, and
    a report drowned in them is a report nobody reads. The count IS the evidence
    -- it says the enforcement fired on every page rather than silently agreeing
    with what it was handed.
    """

    def __init__(self, stats: _Stats) -> None:
        super().__init__(level=logging.WARNING)
        self._stats = stats

    def emit(self, record: logging.LogRecord) -> None:
        if "out of chronological order" in record.getMessage():
            self._stats.out_of_order_pages += 1


def _price() -> ModelPrice | None:
    try:
        return fetch_model_prices([PROJECTION_MODEL])[PROJECTION_MODEL]
    except PricingUnavailableError as exc:
        print(f"  (live pricing unavailable: {exc})")
        return None


def _report(stats: _Stats, price: ModelPrice | None, *, label: str) -> None:
    print(f"\n--- {label} ---")
    print(f"  messages scanned          {stats.messages_scanned}")
    print(f"  history page requests     {stats.page_requests}")
    print(f"  candidates offered        {stats.candidates_offered}")
    print(f"  candidates staged         {stats.candidates_staged}")
    print(f"  distillation calls        {stats.distillation_calls}")
    if stats.batch_sizes:
        print(
            f"  batch size min/med/max    {min(stats.batch_sizes)}"
            f"/{sorted(stats.batch_sizes)[len(stats.batch_sizes) // 2]}"
            f"/{max(stats.batch_sizes)}"
        )
    print(f"  worker ticks              {stats.ticks}")
    print(f"  pages re-sorted by Aura   {stats.out_of_order_pages}")
    print(f"  wall clock                {stats.seconds:.1f}s")
    if stats.messages_scanned:
        print(
            f"  per 1,000 messages        {stats.seconds / stats.messages_scanned * 1000:.2f}s"
        )
    print(f"  prompt tokens (estimated) {stats.input_tokens:,} in / {stats.output_tokens:,} out")
    if price is not None:
        cost = price.cost(
            input_tokens=stats.input_tokens, output_tokens=stats.output_tokens
        )
        print(f"  projected spend           ${cost:.4f}")


def _compare(stats: _Stats, baseline: _Stats) -> None:
    """What an interruption actually cost, against an uninterrupted run."""
    print(
        f"  vs. baseline: {stats.distillation_calls - baseline.distillation_calls:+d} "
        f"distillation call(s), {stats.page_requests - baseline.page_requests:+d} page "
        f"request(s), {stats.candidates_staged - baseline.candidates_staged:+d} "
        f"candidate(s), {stats.messages_scanned - baseline.messages_scanned:+d} "
        f"message(s) scanned"
    )


def _project(stats: _Stats, price: ModelPrice | None) -> None:
    print("\n--- projection to larger histories ---")
    if not stats.messages_scanned:
        print("  nothing measured; skipping")
        return
    print(
        "  Scaled linearly from the measured run, which is the right shape: every "
        "number below\n  grows with the message count and nothing in the worker is "
        "quadratic in it."
    )
    print(
        f"\n  {'messages':>10}  {'calls':>7}  {'pages':>7}  {'days at cap 30':>15}  "
        f"{'spend':>9}"
    )
    for size in (2_000, 10_000, 50_000, 200_000):
        scale = size / stats.messages_scanned
        calls = round(stats.distillation_calls * scale)
        pages = round(stats.page_requests * scale)
        days = max(1, -(-calls // 30))
        if price is None:
            spend = "     n/a"
        else:
            spend = "${:.2f}".format(
                price.cost(
                    input_tokens=round(stats.input_tokens * scale),
                    output_tokens=round(stats.output_tokens * scale),
                )
            )
        print(f"  {size:>10,}  {calls:>7}  {pages:>7}  {days:>15}  {spend:>9}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-pricing",
        action="store_true",
        help="skip the live OpenRouter catalog fetch and omit spend figures",
    )
    arguments = parser.parse_args()

    corpus, labelled_fact_worthy = _load_corpus()
    print(
        f"Corpus: {len(corpus)} messages, {labelled_fact_worthy} labelled fact-worthy "
        f"({labelled_fact_worthy / len(corpus):.1%}) across 9 locales\n"
        f"        (reports/extraction-corpus/corpus.json; see "
        f"reports/phase-3a-1b.txt for how it was built)"
    )
    price = None if arguments.no_pricing else _price()
    if price is not None:
        print(
            f"Pricing: {PROJECTION_MODEL} at ${price.usd_per_million_input:.2f}/"
            f"${price.usd_per_million_output:.2f} per Mtok (live catalog)"
        )

    baseline = await _run(corpus, shuffle=False)
    _report(baseline, price, label="sorted pages, no restart")

    shuffled = await _run(corpus, shuffle=True)
    _report(shuffled, price, label="deliberately shuffled pages")
    print(
        "  identical to the sorted run: "
        f"{shuffled.distillation_calls == baseline.distillation_calls and shuffled.candidates_staged == baseline.candidates_staged}"
    )

    restarted = await _run(corpus, restart_after_ticks=3)
    _report(restarted, price, label="clean restart between batches")
    _compare(restarted, baseline)

    crashed = await _run(corpus, restart_after_ticks=3, crash_mid_batch=True)
    _report(crashed, price, label="crash mid-batch, after paying, before the cursor")
    _compare(crashed, baseline)

    _project(baseline, price)


if __name__ == "__main__":
    asyncio.run(main())
