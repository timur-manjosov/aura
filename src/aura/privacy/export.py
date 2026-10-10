"""The knowledge-base export (P7a, R5): every fact of a server as CSV and as Markdown.

Pure functions over facts and links: no database, no Discord, no model. The
command (`aura.commands.data_admin`) reads the rows and attaches the two files
to an ephemeral reply only the requesting admin sees.

What is in it: every fact, active and superseded, with its number, status,
sentence, when it was recorded, when and by which fact it was replaced, the
link to its source message (or the server, when the source was removed on
request) and the facts it is linked to. Not in it: candidates (they are not
facts yet), author IDs, anything about members.

Hostile text, because a fact's sentence comes from a member's message:

- **CSV:** a cell that a spreadsheet would read as a formula (it starts with
  `=`, `+`, `-`, `@`, a tab or a carriage return, or their full-width forms) is
  prefixed with an apostrophe, the OWASP recommendation; NUL characters are
  replaced. Quoting is the csv module's, so commas, quotes and line breaks
  cannot break a row.
- **Markdown:** each sentence is collapsed to one line, invisible format
  characters (bidirectional overrides included) are dropped, and every
  markdown character is escaped -- no link, image, HTML tag, heading or list
  can come out of a fact. The only links are the ones built here from IDs.

Imports `aura.db.models`, `aura.i18n` and `aura.rendering`.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from aura.db.models import Fact, FactStatus
from aura.i18n import t
from aura.rendering import collapse_display_text, escape_display_markdown, source_link

# What a spreadsheet may evaluate at the start of a cell (ASCII and full-width).
_FORMULA_PREFIXES: Final = frozenset("=+-@\t\r＝＋－＠")

# The CSV header: machine-readable, so English in every locale.
CSV_COLUMNS: Final[tuple[str, ...]] = (
    "fact_id",
    "status",
    "content",
    "created_at",
    "superseded_at",
    "superseded_by",
    "source",
    "linked_fact_ids",
)

# Discord's upload limit for a bot without boosts is 10 MiB; stay below it.
MAX_EXPORT_FILE_BYTES: Final = 8 * 1024 * 1024


class ExportTooLargeError(Exception):
    """An export file would exceed what Discord accepts as an attachment."""


@dataclass(frozen=True)
class ExportFile:
    """One file of an export.

    Attributes
    ----------
    filename
        The attachment's name.
    data
        Its bytes (UTF-8).
    """

    filename: str
    data: bytes


def csv_safe_cell(value: str) -> str:
    """Return a cell value no spreadsheet will evaluate.

    Parameters
    ----------
    value
        Any text.

    Returns
    -------
    str
        The text, NUL characters replaced, and prefixed with an apostrophe when
        it would otherwise start with a formula character.
    """
    cleaned = value.replace("\x00", "�")
    if cleaned[:1] in _FORMULA_PREFIXES:
        return "'" + cleaned
    return cleaned


def _date(moment: datetime | None) -> str:
    return "" if moment is None else moment.strftime("%Y-%m-%d %H:%M UTC")


def _links_by_fact(links: Iterable[tuple[int, int]]) -> dict[int, list[int]]:
    linked: dict[int, list[int]] = defaultdict(list)
    for first, second in links:
        linked[first].append(second)
        linked[second].append(first)
    return {fact_id: sorted(others) for fact_id, others in linked.items()}


def render_csv(facts: Sequence[Fact], links: Iterable[tuple[int, int]]) -> bytes:
    """Return the CSV export: one row per fact, UTF-8 with a byte-order mark.

    Parameters
    ----------
    facts
        Every fact of the server, in the order to write them.
    links
        The server's linked pairs.

    Returns
    -------
    bytes
        The file. The byte-order mark makes spreadsheet programs read the
        text as UTF-8 (every supported language, including Japanese and Korean).
    """
    linked = _links_by_fact(links)
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for fact in facts:
        writer.writerow(
            [
                str(fact.id),
                fact.status.value,
                csv_safe_cell(fact.content),
                _date(fact.created_at),
                _date(fact.superseded_at),
                "" if fact.superseded_by_id is None else str(fact.superseded_by_id),
                source_link(fact),
                " ".join(str(other) for other in linked.get(fact.id, [])),
            ]
        )
    return ("﻿" + buffer.getvalue()).encode("utf-8")


def _markdown_text(text: str) -> str:
    collapsed = collapse_display_text(text)
    return escape_display_markdown(collapsed) if collapsed else "…"


def render_markdown(
    facts: Sequence[Fact],
    links: Iterable[tuple[int, int]],
    *,
    locale: str,
    exported_at: datetime,
) -> bytes:
    """Return the Markdown export: the current facts, then the replacement history.

    Parameters
    ----------
    facts
        Every fact of the server.
    links
        The server's linked pairs.
    locale
        The requesting admin's locale, for the headings.
    exported_at
        When the export was made.

    Returns
    -------
    bytes
        The file, UTF-8.
    """
    linked = _links_by_fact(links)
    by_id = {fact.id: fact for fact in facts}
    lines = [
        f"# {t('export_md_title', locale)}",
        "",
        t("export_md_intro", locale, date=_date(exported_at), count=len(facts)),
        "",
        f"## {t('export_md_active', locale)}",
        "",
    ]
    active = [fact for fact in facts if fact.status is FactStatus.ACTIVE]
    if not active:
        lines.append(t("export_md_none", locale))
    for fact in active:
        line = (
            f"- **#{fact.id}** {_markdown_text(fact.content)} "
            f"({t('export_md_recorded', locale, date=_date(fact.created_at))}, "
            f"[{t('export_md_source', locale)}]({source_link(fact)}))"
        )
        if fact.id in linked:
            others = ", ".join(f"#{other}" for other in linked[fact.id])
            line += f" — {t('export_md_linked', locale, facts=others)}"
        lines.append(line)
    lines.extend(["", f"## {t('export_md_history', locale)}", ""])
    superseded = [fact for fact in facts if fact.status is FactStatus.SUPERSEDED]
    if not superseded:
        lines.append(t("export_md_none", locale))
    for fact in superseded:
        successor = by_id.get(fact.superseded_by_id) if fact.superseded_by_id else None
        replaced = (
            t("export_md_replaced_by", locale, fact_id=successor.id)
            if successor is not None
            else t("export_md_replaced_removed", locale)
        )
        lines.append(
            f"- **#{fact.id}** {_markdown_text(fact.content)} "
            f"({t('export_md_recorded', locale, date=_date(fact.created_at))}; "
            f"{t('export_md_superseded', locale, date=_date(fact.superseded_at))}; {replaced}, "
            f"[{t('export_md_source', locale)}]({source_link(fact)}))"
        )
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def build_export(
    facts: Sequence[Fact],
    links: Iterable[tuple[int, int]],
    *,
    locale: str,
    exported_at: datetime,
) -> tuple[ExportFile, ExportFile]:
    """Return the CSV and the Markdown export of one server's facts.

    Parameters
    ----------
    facts
        Every fact of the server.
    links
        Its linked pairs.
    locale
        The requesting admin's locale.
    exported_at
        When the export was made (also in the file names).

    Returns
    -------
    tuple[ExportFile, ExportFile]
        (CSV, Markdown).

    Raises
    ------
    ExportTooLargeError
        If either file would exceed `MAX_EXPORT_FILE_BYTES`.
    """
    ordered = sorted(facts, key=lambda fact: fact.id)
    pairs = list(links)
    stamp = exported_at.strftime("%Y-%m-%d")
    files = (
        ExportFile(filename=f"aura-facts-{stamp}.csv", data=render_csv(ordered, pairs)),
        ExportFile(
            filename=f"aura-facts-{stamp}.md",
            data=render_markdown(ordered, pairs, locale=locale, exported_at=exported_at),
        ),
    )
    if any(len(file.data) > MAX_EXPORT_FILE_BYTES for file in files):
        raise ExportTooLargeError("an export file exceeds the attachment limit")
    return files
