"""The purge job: re-applies the deletion ledger, purges servers Aura left, applies retention.

One cycle (`run_purge_cycle`), run at start-up's end and then every
DATA_PURGE_CHECK_INTERVAL_SECONDS by `run_purge_sweeper`:

1. **Ledger.** Every recorded deletion is re-applied (see
   `aura.privacy.requests.reapply_ledger`). Always on, whatever the mode:
   requested deletions are not the purge job's to postpone.
2. **Servers Aura left.** For every departure whose period has ended: if Aura
   is back in that server, the mark is cleared; otherwise, in DELETE mode the
   server's data is purged (and recorded in the ledger), in REPORT mode the
   purge is run as a dry run and its counts are logged. Nothing happens while
   the gateway is not ready, because "not in the server" can only be judged
   from a live, complete guild list.
3. **Retention.** The retention rules run for real in DELETE mode and as a
   dry run in REPORT mode.

Every log line is counts and guild IDs, never content. A failing cycle is
logged and the next one runs on schedule; the job never stops on its own.

Imports `aura.db.deletion`, `aura.db.guild_departures` and `aura.privacy`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import aiosqlite

from aura.config import DataPurgeMode, Settings
from aura.db.connection import utc_now
from aura.db.deletion import DeletionCounts, RetentionPolicy, apply_retention, purge_guild
from aura.db.guild_departures import clear_departure, due_departures
from aura.privacy.ledger import DeletionLedger, DeletionReason
from aura.privacy.requests import execute_guild_purge, reapply_ledger

logger = logging.getLogger(__name__)


class GuildPresence(Protocol):
    """What the purge job needs to know from the Discord gateway."""

    def is_ready(self) -> bool:
        """Report whether the gateway connection is live and its guild list complete."""
        ...

    def is_member_of(self, guild_id: int) -> bool:
        """Report whether Aura is in this server right now (an unavailable server counts)."""
        ...


@dataclass
class PurgeCycleReport:
    """What one cycle did or, in report mode, would have done.

    Attributes
    ----------
    reapplied
        Rows removed again by the ledger.
    purged
        Per server purged in this cycle (DELETE mode).
    would_purge
        Per server due but only reported (REPORT mode).
    returned
        Servers whose departure was cleared because Aura is back.
    retention
        The retention rules' counts (real or dry, by mode).
    skipped_not_ready
        True when the departure step was skipped because the gateway was not ready.
    """

    reapplied: DeletionCounts = field(default_factory=DeletionCounts)
    purged: dict[int, DeletionCounts] = field(default_factory=dict)
    would_purge: dict[int, DeletionCounts] = field(default_factory=dict)
    returned: list[int] = field(default_factory=list)
    retention: DeletionCounts = field(default_factory=DeletionCounts)
    skipped_not_ready: bool = False


def retention_policy(settings: Settings) -> RetentionPolicy:
    """Return the retention periods configured in the settings.

    Parameters
    ----------
    settings
        Loaded configuration.

    Returns
    -------
    RetentionPolicy
        The three periods.
    """
    return RetentionPolicy(
        proactive_signal_days=settings.proactive_signal_retention_days,
        ask_member_id_days=settings.ask_member_id_retention_days,
        onboarding_send_days=settings.onboarding_send_retention_days,
    )


async def run_purge_cycle(
    db: aiosqlite.Connection,
    ledger: DeletionLedger,
    presence: GuildPresence,
    *,
    settings: Settings,
    now: datetime,
) -> PurgeCycleReport:
    """Run one cycle of the purge job.

    Parameters
    ----------
    db
        The main database.
    ledger
        The deletion ledger.
    presence
        The gateway's view of which servers Aura is in.
    settings
        Loaded configuration (mode and periods).
    now
        The cycle's moment; also the bound of any purge it executes.

    Returns
    -------
    PurgeCycleReport
        What happened.
    """
    report = PurgeCycleReport()
    report.reapplied = await reapply_ledger(db, ledger)
    deleting = settings.data_purge_mode is DataPurgeMode.DELETE

    if not presence.is_ready():
        report.skipped_not_ready = True
    else:
        for departure in await due_departures(db, now=now):
            # The last check before anything is destroyed: Discord's own
            # current answer, not the mark written when Aura left.
            if presence.is_member_of(departure.guild_id):
                await clear_departure(db, guild_id=departure.guild_id)
                report.returned.append(departure.guild_id)
                continue
            if deleting:
                report.purged[departure.guild_id] = await execute_guild_purge(
                    db,
                    ledger,
                    guild_id=departure.guild_id,
                    reason=DeletionReason.LEFT_SERVER,
                    now=now,
                )
            else:
                report.would_purge[departure.guild_id] = await purge_guild(
                    db, guild_id=departure.guild_id, before=now, dry_run=True
                )

    report.retention = await apply_retention(
        db, now=now, policy=retention_policy(settings), dry_run=not deleting
    )

    for guild_id, counts in report.would_purge.items():
        logger.info(
            "Purge job (report only): server %s left Aura more than %d day(s) ago; a purge "
            "would delete %s",
            guild_id,
            settings.guild_purge_grace_days,
            counts.summary(),
        )
    for guild_id in report.returned:
        logger.info("Purge job: Aura is back in server %s; its data is kept", guild_id)
    if report.retention.total:
        logger.info(
            "Purge job retention (%s): %s",
            "deleted" if deleting else "report only, nothing deleted",
            report.retention.summary(),
        )
    if report.skipped_not_ready:
        logger.info("Purge job: gateway not ready; departed servers are checked next cycle")
    return report


async def run_purge_sweeper(
    db: aiosqlite.Connection,
    ledger: DeletionLedger,
    presence: GuildPresence,
    *,
    settings: Settings,
) -> None:
    """Run the purge job forever, one cycle per interval.

    Parameters
    ----------
    db
        The main database.
    ledger
        The deletion ledger.
    presence
        The gateway's view of which servers Aura is in.
    settings
        Loaded configuration.

    Returns
    -------
    None
        Only ever ends by cancellation.
    """
    interval = settings.data_purge_check_interval_seconds
    logger.info(
        "Purge job started: every %.0fs, mode=%s, %d day(s) after Aura leaves a server",
        interval,
        settings.data_purge_mode.value,
        settings.guild_purge_grace_days,
    )
    while True:
        try:
            await run_purge_cycle(db, ledger, presence, settings=settings, now=utc_now())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Purge job cycle failed; the next cycle runs on schedule")
        await asyncio.sleep(interval)
