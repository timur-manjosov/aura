"""Loading the corpus into the scratch database, and reading it back out.

The facts go in through the *production* write path -- the real schema, the
real `create_fact`, the real embedding dtype -- because Stage 2 retrieval reads
them back through the real `get_active_facts`. Anything hand-rolled here would
mean the simulation measured a slightly different system than the one that
ships, which is the whole failure this phase exists to avoid.

The one deliberate divergence from `aura.facts_service.add_fact` is batching:
this is a backfill of a few hundred facts at once, and CLAUDE.md's Performance
section asks for batch operations rather than a loop of single calls. So the
embeddings are computed in one batched inference pass and then written, instead
of one inference call per fact. The bytes written are identical either way --
same dtype, same `tobytes()`, same column.
"""
from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
from fastembed import TextEmbedding

from aura.db.connection import connection_lock
from aura.db.repository import create_fact
from aura.embeddings import EMBEDDING_DTYPE, embed_texts
from synthetic_corpus.corpus_model import SyntheticCorpus, synthetic_message_id
from synthetic_corpus.scenarios import SCENARIOS

# All synthetic facts for one guild land in a single synthetic channel. Aura's
# retrieval is guild-scoped, so the channel only matters for the permalink a
# real answer would render -- one channel keeps the IDs readable without
# changing anything the pipeline measures.
_FACT_CHANNEL_OFFSET = 1


class CorpusLoadError(RuntimeError):
    """Raised when a corpus file is unreadable or internally inconsistent."""


def read_corpus(path: Path) -> SyntheticCorpus:
    """Read and validate a corpus JSON file.

    Referential integrity is checked here rather than trusted: a message
    pointing at a fact that does not exist would otherwise become a silently
    unscoreable case, shrinking the evidence base without anyone noticing.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusLoadError(f"could not read corpus at {path}: {exc}") from exc

    corpus = SyntheticCorpus.model_validate(raw)
    problems = corpus.check_referential_integrity()
    if problems:
        raise CorpusLoadError(
            f"corpus at {path} has {len(problems)} broken reference(s):\n  "
            + "\n  ".join(problems[:20])
        )
    return corpus


def write_corpus(corpus: SyntheticCorpus, path: Path) -> None:
    """Write a corpus to JSON, creating its directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        corpus.model_dump_json(indent=2, exclude_none=False), encoding="utf-8"
    )


async def store_corpus(
    conn: aiosqlite.Connection, model: TextEmbedding, corpus: SyntheticCorpus
) -> dict[tuple[str, str], int]:
    """Write every guild's facts and messages into the scratch database.

    Returns the mapping from (guild_key, fact_key) to the database fact ID, so
    a Stage 2 result can be scored against the fact it was generated for. The
    same mapping is persisted in `synthetic_fact_keys`, so a later run that
    only reads the database does not need the JSON corpus to interpret it.
    """
    key_to_id: dict[tuple[str, str], int] = {}

    for guild in corpus.guilds:
        if not guild.facts:
            continue

        embeddings = await embed_texts(model, [fact.content for fact in guild.facts])
        for offset, (fact, embedding) in enumerate(zip(guild.facts, embeddings, strict=True)):
            created = await create_fact(
                conn,
                guild_id=guild.guild_id,
                channel_id=guild.guild_id + _FACT_CHANNEL_OFFSET,
                message_id=synthetic_message_id(guild.index, offset),
                content=fact.content,
                embedding=embedding.astype(EMBEDDING_DTYPE, copy=False).tobytes(),
            )
            key_to_id[(guild.key, fact.key)] = created.id

        async with connection_lock(conn):
            await conn.executemany(
                "INSERT OR REPLACE INTO synthetic_fact_keys "
                "(fact_id, guild_key, fact_key, contradicts_key) VALUES (?, ?, ?, ?)",
                [
                    (key_to_id[(guild.key, fact.key)], guild.key, fact.key, fact.contradicts_key)
                    for fact in guild.facts
                ],
            )
            await conn.commit()

    async with connection_lock(conn):
        await conn.executemany(
            "INSERT OR REPLACE INTO synthetic_messages "
            "(key, guild_key, category, locale, content, target_fact_keys, "
            "adversarial_kind, rationale) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    message.key,
                    message.guild_key,
                    message.category.value,
                    message.locale,
                    message.content,
                    json.dumps(message.target_fact_keys),
                    message.adversarial_kind.value if message.adversarial_kind else None,
                    message.rationale,
                )
                for message in corpus.messages
            ],
        )
        await conn.commit()

    return key_to_id


async def read_fact_key_map(conn: aiosqlite.Connection) -> dict[int, tuple[str, str]]:
    """Return database fact ID -> (guild_key, fact_key) for every stored fact."""
    async with connection_lock(conn):
        async with conn.execute(
            "SELECT fact_id, guild_key, fact_key FROM synthetic_fact_keys"
        ) as cursor:
            rows = await cursor.fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def assert_corpus_matches_database(
    corpus: SyntheticCorpus, fact_key_by_id: dict[int, tuple[str, str]]
) -> None:
    """Refuse to simulate a corpus against a database built from a different one.

    Silently the most dangerous mismatch in this tooling: pointing `--corpus` at
    one run and `--db` at another produces a complete, plausible-looking report
    in which every ground-truth fact simply fails to be found, because the keys
    do not line up. Recall would read as catastrophic and the cause would be
    invisible. Checking the key sets agree turns that into an immediate, loud
    failure instead.
    """
    in_database = set(fact_key_by_id.values())
    in_corpus = {
        (guild.key, fact.key) for guild in corpus.guilds for fact in guild.facts
    }
    missing = in_corpus - in_database
    if missing:
        example = sorted(missing)[:5]
        raise CorpusLoadError(
            f"the scratch database is missing {len(missing)} of the corpus's "
            f"{len(in_corpus)} facts (e.g. {example}). The --corpus file and the "
            "--db file come from different generation runs; re-run "
            "generate_synthetic_corpus.py or point both at the same run."
        )


def assert_corpus_matches_scenario_grid(corpus: SyntheticCorpus) -> None:
    """Refuse to simulate a corpus that covers fewer guilds than the full grid.

    `generate_synthetic_corpus.py --limit N` and `write_corpus` both write to
    the same fixed default path a full run uses, with no distinct name and no
    check for an existing, larger file already there -- a small run (a
    developer smoke-testing the generator, say) silently overwrites the real
    corpus with one that is completely well-formed and self-consistent, just
    scoped to fewer guilds. Neither `read_corpus`'s schema/referential checks
    nor `assert_corpus_matches_database` catch this: a smaller corpus is still
    internally valid and can still pair correctly with a scratch database built
    from that same smaller run. This is the one check that actually compares
    the loaded corpus against what a full run is supposed to contain.
    """
    expected = {scenario.key for scenario in SCENARIOS}
    actual = {guild.key for guild in corpus.guilds}
    missing = expected - actual
    if missing:
        raise CorpusLoadError(
            f"the corpus is missing {len(missing)} of the {len(expected)} guilds "
            f"in the full scenario grid (e.g. {sorted(missing)[:5]}). This looks "
            "like a partial run (e.g. generate_synthetic_corpus.py --limit N) "
            "that overwrote a full corpus at the default path. Re-run "
            "generate_synthetic_corpus.py for the full grid, or pass --corpus "
            "to point at the intended file explicitly."
        )


def message_discord_id(guild_index: int, position: int) -> int:
    """A synthetic Discord message ID for one corpus message.

    Offset well past the block used by facts so a message ID and a fact's
    origin ID can never collide inside the same guild -- which matters because
    the gate's escalation ledger is keyed on (channel_id, message_id) and a
    collision there would silently drop a case as a duplicate delivery.
    """
    return synthetic_message_id(guild_index, 500_000 + position)
