"""Phase 4c-F01: repeatable round trips of the billing path against a Stripe SANDBOX.

Reruns every automatable part of reports/phase-4c-stripe-verification.md --
everything except the two browser steps (paying on the hosted Checkout page,
looking at the customer portal) -- against real Stripe, and writes a
transcript in the format of reports/phase-4b-verification.txt.

WHAT RUNS WHERE. The application under test is the real code: the web backend
(aura_web.app.create_app) with every Stripe value from web/.env, behind the
real Next.js dev server, and the bot's real internal billing API and plan gate
(aura.billing) on a throwaway SQLite file with BILLING_MODE=enforced. Only
Discord is replaced, by the repository's own stand-in (verify_oauth_fixtures),
so nothing here can reach Discord or start the bot. The app talks to Stripe
with the restricted key from web/.env and nothing else. The Stripe CLI, with
its own full sandbox permissions, is the harness: it creates customers, test
clocks, subscriptions and prices, and resends events.

    stripe listen --> 127.0.0.1:3000 capture hop --> Next.js :3001 --> backend :8080
                                                                      +--> bot API :18081

The capture hop records each delivery's exact bytes before Next.js sees them,
which is what the tampering checks replay, and what proves Next.js passes a
signed body through unchanged.

PREREQUISITES
  * `stripe listen --forward-to localhost:3000/api/stripe/webhook --events <the
    13 handled types>` running in another terminal (web/.env.example lists them);
    its signing secret must equal AURA_WEB_STRIPE_WEBHOOK_SECRET.
  * web/.env: a sandbox restricted key, the webhook secret, the Pro price.
  * the root .env: INTERNAL_API_SECRET equal to AURA_WEB_BOT_INTERNAL_API_SECRET.
  * Ports 3000, 3001, 8080, 8091, 8092, 18081, 18090, 18099 free on 127.0.0.1.

MANAGED PAYMENTS (V-01). The app must create its checkout whatever the
account's "Managed Payments by default" setting is. Stripe exposes that setting
through no API field, so step A3 detects it by behaviour: one card-only
Checkout Session request WITHOUT managed_payments[enabled], sent with the
application's key. Default on: Stripe refuses it (nothing is created) and the
refusal's type/code/param are recorded. Default off: the session is created
and expired at once. The app's own checkout is then required to succeed either
way. To test both states, the operator switches the setting in the Dashboard
(Settings -> Payments -> Managed Payments) between two runs, and may pass
--managed-payments-default on|off so the run fails if the account is not in
the state the operator meant to test.

REFUSES TO RUN unless: the key is rk_test_/sk_test_, no live key value exists
in either .env, the key's own account and the CLI's account are the expected
sandbox, the key cannot create payouts, the CLI reports that sandbox on every
call, every object it returns has livemode=false, and the listener's secret
hashes equal to the configured one.

NEVER PRINTS A SECRET. Every line of output passes through `redact`, which
replaces the exact values of every credential on this machine, Stripe's
masked key form, portal-session URLs and Checkout fragments. The last step
scans every file the run produced for those exact values, with a planted
canary as the positive control.

Usage:  python scripts/verify_stripe_sandbox.py [--out PATH] [--keep-objects]
                                                [--managed-payments-default on|off]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import signal
import socket
import sqlite3
import string
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
BACKEND_DIR: Final[Path] = REPO_ROOT / "web" / "backend"
SCRIPTS_DIR: Final[Path] = REPO_ROOT / "scripts"
FRONTEND_DIR: Final[Path] = REPO_ROOT / "web" / "frontend"
for _extra_path in (str(REPO_ROOT / "src"), str(BACKEND_DIR), str(SCRIPTS_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import httpx  # noqa: E402  -- after the sys.path edit, like every harness here

EXPECTED_ACCOUNT: Final[str] = "acct_1UF7PdPVSNHvCZiz"
STRIPE_API: Final[str] = "https://api.stripe.com"
STRIPE_VERSION: Final[str] = "2026-08-26.dahlia"

CAPTURE_PORT: Final[int] = 3000
FRONTEND_PORT: Final[int] = 3001
BACKEND_PORT: Final[int] = 8080
WRONG_KEY_PORT: Final[int] = 8091
LIVE_KEY_PORT: Final[int] = 8092
BOT_PORT: Final[int] = 18081
DISCORD_PORT: Final[int] = 18090
RECORDER_PORT: Final[int] = 18099

FIXTURE_USER: Final[str] = "5000"
FIXTURE_GUILD: Final[str] = "1000"
RENEWAL_GRACE: Final[timedelta] = timedelta(hours=72)
PAYMENT_GRACE: Final[timedelta] = timedelta(days=7)
# Stripe finalizes a renewal draft about an hour after creating it; two hours
# past a period boundary is safely past the first charge attempt.
PAST_BOUNDARY_SECONDS: Final[int] = 2 * 3600
# ...but not always: in the Phase 4c-F01 fixes acceptance run a renewal was
# still a draft at +2 h, the next step's failing card then paid for it, and the
# whole lifecycle shifted by a period. A step that lands on a draft therefore
# advances an hour at a time until Stripe has finalized it, never further than
# Stripe's own limit (automatically_finalizes_at: created + 72 h).
DRAFT_SETTLE_STEP_SECONDS: Final[int] = 3600
DRAFT_SETTLE_LIMIT_SECONDS: Final[int] = 72 * 3600
DAY: Final[int] = 86400

HANDLED_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "checkout.session.completed",
        "checkout.session.async_payment_succeeded",
        "checkout.session.async_payment_failed",
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
        "customer.subscription.paused",
        "customer.subscription.resumed",
        "invoice.paid",
        "invoice.payment_failed",
        "invoice.payment_action_required",
        "invoice.voided",
        "invoice.marked_uncollectible",
    }
)

_ALPHANUMERIC: Final[str] = string.ascii_letters + string.digits
_CREDENTIAL_SHAPE: Final[re.Pattern[str]] = re.compile(
    r"((?:sk|rk|pk|oak|rak)_(?:test|live)_|whsec_)[A-Za-z0-9_]{6,}"
)
_MASKED_KEY_SHAPE: Final[re.Pattern[str]] = re.compile(
    r"((?:sk|rk|pk)_(?:test|live)_)\.\.\.[A-Za-z0-9]+"
)
_BEARER_URLS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"(billing\.stripe\.com/p/session\?secret=)[A-Za-z0-9_\-]+"), r"\1<REDACTED>"),
    (re.compile(r"(billing\.stripe\.com/p/session/)[A-Za-z0-9_\-]+"), r"\1<REDACTED>"),
    (
        re.compile(r"(checkout\.stripe\.com/c/pay/cs_(?:test|live)_[A-Za-z0-9]+)#[A-Za-z0-9%_\-]+"),
        r"\1#<REDACTED>",
    ),
)
_LIVE_VALUE: Final[re.Pattern[str]] = re.compile(r"(?:sk|rk)_live_[A-Za-z0-9]")


class SandboxRefusal(SystemExit):
    """A safety precondition failed; the run stops before touching anything."""


# --------------------------------------------------------------------------- secrets


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines of a .env file into memory.

    Parameters
    ----------
    path
        The file to read.

    Returns
    -------
    dict[str, str]
        Every assignment, comments and blanks skipped, surrounding quotes removed.
    """
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass
class Secrets:
    """Every credential on this machine, in memory only, and the redaction built from them."""

    web: dict[str, str]
    root: dict[str, str]
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Secrets:
        """Read web/.env, the root .env and the Stripe CLI's credential store.

        Returns
        -------
        Secrets
            The loaded values. Nothing is printed.
        """
        extra: dict[str, str] = {}
        cli_store = Path.home() / ".config" / "stripe" / "credentials.json"
        with suppress(OSError, ValueError):
            for key, value in json.loads(cli_store.read_text(encoding="utf-8")).items():
                if isinstance(value, str):
                    extra[f"Stripe CLI {key}"] = value
        return cls(
            read_env_file(REPO_ROOT / "web" / ".env"), read_env_file(REPO_ROOT / ".env"), extra
        )

    @property
    def restricted_key(self) -> str:
        """The application's Stripe key (web/.env)."""
        return self.web["AURA_WEB_STRIPE_SECRET_KEY"]

    @property
    def webhook_secret(self) -> str:
        """The webhook endpoint secret (web/.env)."""
        return self.web["AURA_WEB_STRIPE_WEBHOOK_SECRET"]

    @property
    def internal_secret(self) -> str:
        """The bot's internal API secret (root .env)."""
        return self.root["INTERNAL_API_SECRET"]

    @property
    def price_id(self) -> str:
        """The configured Pro price."""
        return self.web["AURA_WEB_STRIPE_PRICE_ID"]

    def labelled(self) -> dict[str, str]:
        """Every secret value by label, high-entropy values only (a date is not a secret).

        Returns
        -------
        dict[str, str]
            Label to value.
        """
        values = {
            "restricted key (AURA_WEB_STRIPE_SECRET_KEY)": self.web.get(
                "AURA_WEB_STRIPE_SECRET_KEY", ""
            ),
            "webhook secret (AURA_WEB_STRIPE_WEBHOOK_SECRET)": self.web.get(
                "AURA_WEB_STRIPE_WEBHOOK_SECRET", ""
            ),
            "web internal API secret": self.web.get("AURA_WEB_BOT_INTERNAL_API_SECRET", ""),
            "bot internal API secret (INTERNAL_API_SECRET)": self.root.get(
                "INTERNAL_API_SECRET", ""
            ),
            "web Discord client secret": self.web.get("AURA_WEB_DISCORD_CLIENT_SECRET", ""),
            "web Discord bot token": self.web.get("AURA_WEB_DISCORD_BOT_TOKEN", ""),
            "bot DISCORD_TOKEN": self.root.get("DISCORD_TOKEN", ""),
            "bot LLM_API_KEY": self.root.get("LLM_API_KEY", ""),
            **self.extra,
        }
        return {label: value for label, value in values.items() if len(value) >= 16}

    def redact(self, text: str) -> str:
        """Remove every secret value and every credential-shaped string from text.

        Parameters
        ----------
        text
            Anything about to be printed or written.

        Returns
        -------
        str
            The text with exact values, Stripe's masked key form, bearer URLs and
            anything shaped like a Stripe credential replaced.
        """
        for label, value in sorted(self.labelled().items(), key=lambda item: -len(item[1])):
            text = text.replace(value, f"<REDACTED:{label}>")
        for pattern, replacement in _BEARER_URLS:
            text = pattern.sub(replacement, text)
        text = _MASKED_KEY_SHAPE.sub(lambda match: f"{match.group(1)}...<REDACTED>", text)
        return _CREDENTIAL_SHAPE.sub(lambda match: f"{match.group(1)}<REDACTED>", text)


def fingerprint(value: str) -> str:
    """Return the first 12 hex digits of a value's SHA-256, for comparing secrets without showing them.

    Parameters
    ----------
    value
        The value to fingerprint.

    Returns
    -------
    str
        A 12-character hex prefix.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- report


@dataclass
class Report:
    """The transcript and the pass/fail tally, every line redacted on the way in."""

    secrets: Secrets
    lines: list[str] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    unverifiable: list[str] = field(default_factory=list)

    def section(self, title: str) -> None:
        """Start a titled section."""
        self.lines += ["", "=" * 78, title, "=" * 78]
        print(f"\n== {title}", flush=True)

    def note(self, text: str = "") -> None:
        """Add a line of prose or evidence."""
        clean = self.secrets.redact(text)
        self.lines.append(clean)
        print(clean, flush=True)

    def check(self, description: str, condition: bool, detail: str = "") -> bool:
        """Record one check.

        Parameters
        ----------
        description
            What is asserted.
        condition
            Whether it holds.
        detail
            Evidence shown on failure.

        Returns
        -------
        bool
            The condition, so a caller can branch on it.
        """
        if condition:
            self.passed += 1
            self.note(f"  [PASS] {description}")
        else:
            self.failed += 1
            self.note(f"  [FAIL] {description}" + (f" -- {detail}" if detail else ""))
        return condition

    def not_verifiable(self, description: str, reason: str) -> None:
        """Record a check that cannot be made here, with why."""
        self.unverifiable.append(description)
        self.note(f"  [NOT VERIFIABLE] {description} -- {reason}")


# --------------------------------------------------------------------------- Stripe CLI (harness)


class StripeCli:
    """The Stripe CLI as the harness: full sandbox permissions, never the application's key.

    Every call checks the sandbox banner and refuses a livemode object. Every
    object created with `record` is remembered for cleanup.
    """

    def __init__(self, report: Report) -> None:
        self.report = report
        self.created: list[tuple[str, str]] = []

    def _run(self, arguments: list[str]) -> Any:
        completed = subprocess.run(
            ["stripe", *arguments], capture_output=True, text=True, timeout=180
        )
        if f"({EXPECTED_ACCOUNT})" not in completed.stderr:
            raise SandboxRefusal(
                "the Stripe CLI did not report the expected sandbox account; stopping"
            )
        start = completed.stdout.find("{")
        if start < 0:
            raise RuntimeError(f"the Stripe CLI returned no JSON (exit {completed.returncode})")
        data = json.loads(completed.stdout[start:])
        if isinstance(data, dict) and data.get("livemode") is True:
            raise SandboxRefusal("a livemode object came back from the CLI; stopping")
        return data

    def call(
        self,
        method: str,
        path: str,
        params: dict[str, str] | list[tuple[str, str]] | None = None,
        *,
        record: str | None = None,
    ) -> Any:
        """Call the API through the CLI.

        Parameters
        ----------
        method
            get, post or delete.
        path
            The API path.
        params
            Form parameters (not secrets; the CLI supplies its own credential).
        record
            When given, the created object's ID is remembered for cleanup.

        Returns
        -------
        Any
            The decoded JSON body.

        Raises
        ------
        RuntimeError
            For a Stripe error body.
        """
        arguments = [method, path] + (["--confirm"] if method == "delete" else [])
        items = params.items() if isinstance(params, dict) else (params or [])
        for key, value in items:
            arguments += ["-d", f"{key}={value}"]
        data = self._run(arguments)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(
                f"Stripe error on {method.upper()} {path}: {json.dumps(data['error'])[:500]}"
            )
        if record and isinstance(data, dict) and "id" in data:
            self.created.append((str(data["id"]), record))
        return data

    def get(self, path: str, params: dict[str, str] | list[tuple[str, str]] | None = None) -> Any:
        """GET through the CLI."""
        return self.call("get", path, params)

    def post(
        self,
        path: str,
        params: dict[str, str] | list[tuple[str, str]] | None = None,
        *,
        record: str | None = None,
    ) -> Any:
        """POST through the CLI."""
        return self.call("post", path, params, record=record)

    def resend(self, event_id: str) -> None:
        """Ask Stripe to redeliver one event to the listening CLI."""
        self._run(["events", "resend", event_id, "--confirm"])

    def advance(self, clock_id: str, frozen_time: int) -> int:
        """Advance a test clock and wait until it is ready.

        Parameters
        ----------
        clock_id
            The clock.
        frozen_time
            The target time (Unix seconds).

        Returns
        -------
        int
            The clock's frozen time once ready.
        """
        self.post(
            f"/v1/test_helpers/test_clocks/{clock_id}/advance", {"frozen_time": str(frozen_time)}
        )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            clock = self.get(f"/v1/test_helpers/test_clocks/{clock_id}")
            if clock["status"] == "ready":
                return int(clock["frozen_time"])
            if clock["status"] == "internal_failure":
                raise RuntimeError(f"test clock {clock_id} failed")
            time.sleep(2)
        raise TimeoutError(f"test clock {clock_id} is still advancing")

    def events_since(self, created_gte: int) -> list[dict[str, Any]]:
        """Every event created since a moment, oldest first (one page of 100)."""
        page = self.get("/v1/events", {"created[gte]": str(created_gte), "limit": "100"})
        return list(reversed(page["data"]))


# --------------------------------------------------------------------------- processes


def _listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.3)
        return probe.connect_ex(("127.0.0.1", port)) == 0


@dataclass
class Stack:
    """The local processes of the run, each in its own session, all on 127.0.0.1."""

    work: Path
    processes: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)
    next_files_before: set[str] = field(default_factory=set)

    def _environment(self) -> dict[str, str]:
        environment = {
            k: v for k, v in os.environ.items() if not k.startswith(("AURA_WEB_", "STRIPE_"))
        }
        environment.pop("INTERNAL_API_SECRET", None)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(BACKEND_DIR), str(SCRIPTS_DIR), str(REPO_ROOT / "src")]
        )
        environment["SBX_WORK_DIR"] = str(self.work)
        return environment

    def commands(self) -> dict[str, tuple[list[str], Path, dict[str, str], int]]:
        """The command, directory, environment and port of each process."""
        python = sys.executable
        environment = self._environment()
        script = str(Path(__file__).resolve())
        backend_env = {
            **environment,
            "SBX_DISCORD_API_BASE": f"http://127.0.0.1:{DISCORD_PORT}/api/v10",
            "SBX_BOT_API_URL": f"http://127.0.0.1:{BOT_PORT}",
        }
        return {
            "bot": ([python, script, "--serve", "bot"], REPO_ROOT, environment, BOT_PORT),
            "discord": (
                [
                    python,
                    "-m",
                    "uvicorn",
                    "verify_oauth_fixtures:build_fake",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(DISCORD_PORT),
                    "--log-level",
                    "warning",
                ],
                REPO_ROOT,
                environment,
                DISCORD_PORT,
            ),
            "backend": (
                [
                    python,
                    "-m",
                    "uvicorn",
                    "verify_stripe_sandbox:build_backend",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(BACKEND_PORT),
                    "--log-level",
                    "debug",
                ],
                REPO_ROOT,
                backend_env,
                BACKEND_PORT,
            ),
            "frontend": (
                ["npx", "next", "dev", "-p", str(FRONTEND_PORT), "-H", "127.0.0.1"],
                FRONTEND_DIR,
                {
                    **environment,
                    "AURA_WEB_BACKEND_ORIGIN": f"http://127.0.0.1:{BACKEND_PORT}",
                    "NEXT_TELEMETRY_DISABLED": "1",
                },
                FRONTEND_PORT,
            ),
            "capture": (
                [python, script, "--serve", "capture"],
                REPO_ROOT,
                environment,
                CAPTURE_PORT,
            ),
        }

    def start(self, *names: str) -> None:
        """Start the named processes and wait until each listens."""
        self.next_files_before = self.next_files_before or {p.name for p in FRONTEND_DIR.iterdir()}
        for name in names:
            command, directory, environment, port = self.commands()[name]
            if _listening(port):
                raise SandboxRefusal(
                    f"port {port} ({name}) is already in use; stop whatever holds it"
                )
            log = (self.work / "logs" / f"{name}.log").open("ab")
            process = subprocess.Popen(
                command,
                cwd=directory,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.processes[name] = process
            deadline = time.monotonic() + (180 if name == "frontend" else 45)
            while not _listening(port):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"{name} did not start; see {self.work / 'logs' / (name + '.log')}"
                    )
                time.sleep(0.2)

    def stop(self, *names: str) -> None:
        """Stop the named processes (all when none are named), children included."""
        for name in names or tuple(reversed(list(self.processes))):
            process = self.processes.pop(name, None)
            if process is None:
                continue
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        if not self.processes:
            # next dev writes AGENTS.md and CLAUDE.md into the frontend on first
            # start; a verification run must leave the working tree as it found it.
            for created in (
                {p.name for p in FRONTEND_DIR.iterdir()} - self.next_files_before - {".next"}
            ):
                if created in {"AGENTS.md", "CLAUDE.md"}:
                    (FRONTEND_DIR / created).unlink()


def build_backend() -> Any:
    """uvicorn factory: the real web backend, web/.env's Stripe values, the Discord stand-in.

    Returns
    -------
    Any
        The FastAPI application.

    Notes
    -----
    The Discord client secret and bot token are replaced by the stand-in's
    fixtures, so this process cannot reach Discord. Everything Stripe-related
    comes from web/.env (or from the environment, which pydantic-settings
    prefers -- the wrong-key probe relies on exactly that).
    """
    import logging

    from pydantic import SecretStr

    from aura_web.app import create_app
    from aura_web.config import WebSettings
    from verify_oauth_fixtures import BOT_TOKEN, CLIENT_ID, CLIENT_SECRET

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    settings = WebSettings(  # type: ignore[call-arg]
        discord_client_id=CLIENT_ID,
        discord_client_secret=SecretStr(CLIENT_SECRET),
        discord_bot_token=SecretStr(BOT_TOKEN),
        discord_api_base=os.environ["SBX_DISCORD_API_BASE"],
        bot_internal_api_url=os.environ["SBX_BOT_API_URL"],
        log_level="DEBUG",
    )
    return create_app(settings)


def serve_bot_harness(work: Path) -> None:
    """The bot's side: the real internal API and plan gate, enforced, on a throwaway database.

    Parameters
    ----------
    work
        The run's work directory: database and clock file live there.

    Notes
    -----
    If clock.txt holds a Unix time, that is "now" for the gate and every apply
    -- the bot side follows a Stripe test clock that way.
    """
    import logging

    import aiosqlite
    from pydantic import SecretStr

    from aura.billing import PlanGate
    from aura.billing.internal_api import start_internal_api
    from aura.config import BillingMode, Settings
    from aura.db import init_schema
    from aura.db.subscriptions import load_subscription_records, verify_subscriptions_schema

    def clock() -> datetime:
        with suppress(OSError):
            raw = (work / "clock.txt").read_text().strip()
            if raw.isdigit():
                return datetime.fromtimestamp(int(raw), tz=UTC)
        return datetime.now(UTC)

    async def run() -> None:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            stream=sys.stdout,
        )
        logging.getLogger("aiosqlite").setLevel(logging.INFO)
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            discord_token=SecretStr("sandbox-harness-never-connects-to-discord"),
            billing_mode=BillingMode.ENFORCED,
            internal_api_secret=SecretStr(read_env_file(REPO_ROOT / ".env")["INTERNAL_API_SECRET"]),
            internal_api_host="127.0.0.1",
            internal_api_port=BOT_PORT,
            database_path=str(work / "bot.db"),
        )
        connection = await aiosqlite.connect(work / "bot.db")
        await init_schema(connection)
        await verify_subscriptions_schema(connection)
        gate = PlanGate.from_settings(
            settings, records=await load_subscription_records(connection), clock=clock
        )
        assert settings.internal_api_secret is not None
        server = await start_internal_api(
            connection,
            gate,
            secret=settings.internal_api_secret,
            host="127.0.0.1",
            port=BOT_PORT,
            clock=clock,
        )
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, stop.set)
        await stop.wait()
        await server.stop()
        await connection.close()

    asyncio.run(run())


def serve_capture_hop(work: Path) -> None:
    """127.0.0.1:3000 -> Next.js on 3001, recording each webhook delivery's exact bytes first.

    Parameters
    ----------
    work
        The run's work directory; deliveries go to captures/ with mode 600.
    """
    import aiohttp
    from aiohttp import web

    captures = work / "captures"
    captures.mkdir(exist_ok=True)
    hop = {"host", "content-length", "connection", "transfer-encoding", "keep-alive"}

    async def forward(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        stamp = f"{time.time():.6f}"
        is_webhook = request.method == "POST" and request.path == "/api/stripe/webhook"
        if is_webhook:
            (captures / f"{stamp}.body").write_bytes(body)
            (captures / f"{stamp}.headers.json").write_text(
                json.dumps(list(request.headers.items()))
            )
            for suffix in (".body", ".headers.json"):
                os.chmod(captures / f"{stamp}{suffix}", 0o600)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in hop}
        async with (
            aiohttp.ClientSession(auto_decompress=False) as session,
            session.request(
                request.method,
                f"http://127.0.0.1:{FRONTEND_PORT}{request.path_qs}",
                headers=headers,
                data=body,
                allow_redirects=False,
            ) as upstream,
        ):
            payload = await upstream.read()
            response = web.Response(status=upstream.status, body=payload)
            for key, value in upstream.headers.items():
                if key.lower() not in hop and key.lower() != "content-encoding":
                    response.headers.add(key, value)
        if is_webhook:
            (captures / f"{stamp}.response.json").write_text(
                json.dumps({"status": upstream.status, "body": payload.decode("utf-8", "replace")})
            )
        return response

    app = web.Application(client_max_size=4 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", forward)
    web.run_app(app, host="127.0.0.1", port=CAPTURE_PORT, access_log=None, print=None)


# --------------------------------------------------------------------------- the run


@dataclass
class Run:
    """State shared by the steps of one verification run."""

    secrets: Secrets
    report: Report
    cli: StripeCli
    stack: Stack
    work: Path
    session_cookie: str = ""
    # What the operator says the account's Managed Payments default is set to,
    # if they said; step A3 fails when the detected default disagrees.
    expected_managed_payments_default: bool | None = None
    # Every run uses its own guild IDs: the backend's reconciler syncs every Aura
    # subscription in the account into the fresh database, and an earlier run's
    # subscription on the same guild would otherwise decide this run's plan.
    guild_base: int = field(default_factory=lambda: 10**15 + secrets.randbelow(10**12) * 1000)

    # ---- small helpers ----------------------------------------------------

    def guild(self, suffix: int) -> str:
        """This run's guild ID for a scenario number."""
        return str(self.guild_base + suffix)

    def key_headers(self, key: str | None = None) -> dict[str, str]:
        """Authorization and version headers for a direct call with the application's key."""
        return {
            "Authorization": f"Bearer {key or self.secrets.restricted_key}",
            "Stripe-Version": STRIPE_VERSION,
        }

    def set_bot_clock(self, seconds: int | None) -> None:
        """Move the bot side's clock (None: real time)."""
        (self.work / "clock.txt").write_text("" if seconds is None else str(seconds))

    def db(self) -> sqlite3.Connection:
        """A read-only connection to the bot harness's database."""
        connection = sqlite3.connect(f"file:{self.work / 'bot.db'}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def stored(self, subscription_id: str) -> dict[str, Any] | None:
        """The bot's stored row for one subscription."""
        with self.db() as connection:
            row = connection.execute(
                "SELECT * FROM guild_subscriptions WHERE subscription_id = ?", (subscription_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def ledger(self, subscription_id: str | None = None) -> list[str]:
        """Event IDs in the bot's processed-event ledger."""
        with self.db() as connection:
            if subscription_id is None:
                rows = connection.execute("SELECT event_id FROM stripe_processed_events").fetchall()
            else:
                rows = connection.execute(
                    "SELECT event_id FROM stripe_processed_events WHERE subscription_id = ?",
                    (subscription_id,),
                ).fetchall()
        return [row[0] for row in rows]

    def db_fingerprint(self) -> str:
        """A hash over every stored subscription and ledger row."""
        with self.db() as connection:
            rows = [
                tuple(r)
                for r in connection.execute(
                    "SELECT * FROM guild_subscriptions ORDER BY subscription_id"
                )
            ]
            rows += [
                tuple(r)
                for r in connection.execute(
                    "SELECT * FROM stripe_processed_events ORDER BY event_id"
                )
            ]
        return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()[:16]

    def settle(self, *, timeout: float = 60.0, quiet: float = 6.0) -> None:
        """Wait until the ledger has stopped growing for `quiet` seconds."""
        deadline = time.monotonic() + timeout
        last, last_change = len(self.ledger()), time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(1.0)
            now = len(self.ledger())
            if now != last:
                last, last_change = now, time.monotonic()
            elif time.monotonic() - last_change >= quiet:
                return

    def plan(self, guild_id: str) -> dict[str, Any]:
        """The live gate's answer for one guild, through the internal API the web backend uses."""
        response = httpx.post(
            f"http://127.0.0.1:{BOT_PORT}/internal/v1/guilds/plans",
            headers={"Authorization": f"Bearer {self.secrets.internal_secret}"},
            json={"guild_ids": [guild_id]},
            timeout=10,
        )
        return dict(response.json()["plans"][guild_id])

    def plan_text(self, guild_id: int, at: datetime) -> str:
        """The German /aura-plan text the bot would send for a guild at a moment."""
        from aura.billing import PlanGate
        from aura.billing.entitlement import GracePolicy
        from aura.commands.plan import describe_plan
        from aura.db.subscriptions import load_subscription_records

        async def records() -> list[Any]:
            import aiosqlite

            async with aiosqlite.connect(
                f"file:{self.work / 'bot.db'}?mode=ro", uri=True
            ) as connection:
                return list(await load_subscription_records(connection))

        gate = PlanGate(
            enforced=True,
            policy=GracePolicy(renewal_grace=RENEWAL_GRACE, payment_failure_grace=PAYMENT_GRACE),
            complimentary_guild_ids=frozenset(),
            records=asyncio.run(records()),
            clock=lambda: at,
        )
        return describe_plan(gate.plan_for(guild_id), locale="de", dashboard_url=None).replace(
            "\n", " / "
        )

    def parse(self, payload: dict[str, Any]) -> Any:
        """Feed a real subscription object through the REAL parser."""
        from aura_web.stripe_api import parse_subscription

        return parse_subscription(payload, pro_price_id=self.secrets.price_id)

    def subscription(self, subscription_id: str) -> dict[str, Any]:
        """Fetch a subscription with its latest invoice expanded, and save it (redacted)."""
        payload: dict[str, Any] = self.cli.get(
            f"/v1/subscriptions/{subscription_id}", [("expand[]", "latest_invoice")]
        )
        return payload

    def browser(self, method: str, path: str, body: dict[str, Any] | None = None) -> httpx.Response:
        """A same-origin browser request through localhost:3000 with the stand-in session."""
        headers = {"Cookie": f"aura_session={self.session_cookie}"}
        if body is not None:
            headers.update(
                {
                    "Content-Type": "application/json",
                    "Origin": "http://localhost:3000",
                    "Sec-Fetch-Site": "same-origin",
                }
            )
        return httpx.request(
            method,
            f"http://localhost:{CAPTURE_PORT}{path}",
            headers=headers,
            content=None if body is None else json.dumps(body),
            timeout=60,
        )

    def login(self) -> None:
        """Log the fixture moderator in through the Discord stand-in."""
        with httpx.Client(timeout=30, follow_redirects=False) as http:
            start = http.get(f"http://localhost:{CAPTURE_PORT}/api/auth/login")
            state_match = re.search(r"[?&]state=([^&]+)", start.headers["location"])
            assert state_match is not None
            state = state_match.group(1)
            code = http.post(
                f"http://127.0.0.1:{DISCORD_PORT}/__fake__/issue-code",
                json={"user_id": FIXTURE_USER},
            ).json()["code"]
            callback = http.get(
                f"http://localhost:{CAPTURE_PORT}/api/auth/callback",
                params={"code": code, "state": state},
                headers={"Cookie": f"aura_oauth_state={state}"},
            )
        self.session_cookie = next(
            value.split(";", 1)[0].split("=", 1)[1]
            for value in callback.headers.get_list("set-cookie")
            if value.startswith("aura_session=")
        )

    def _let_a_renewal_draft_finalize(
        self, subscription_id: str, clock: str, frozen_time: int
    ) -> None:
        """Advance the clock an hour at a time while the latest invoice is still a draft."""
        target = frozen_time
        while target - frozen_time < DRAFT_SETTLE_LIMIT_SECONDS:
            latest = self.subscription(subscription_id).get("latest_invoice") or {}
            if latest.get("status") != "draft":
                return
            plan_while_draft = self.plan_for_subscription(subscription_id)
            self.report.note(
                "  the renewal invoice is still a draft (Stripe has not finalized it yet); gate "
                f"meanwhile: {plan_while_draft}; advancing the clock one more hour"
            )
            target += DRAFT_SETTLE_STEP_SECONDS
            now = self.cli.advance(clock, target)
            self.set_bot_clock(now)
            self.settle()
        self.report.note("  the renewal draft was still not finalized at Stripe's own limit")

    def plan_for_subscription(self, subscription_id: str) -> str:
        """The gate's tier and standing for the guild a stored subscription belongs to."""
        row = self.stored(subscription_id) or {}
        if "guild_id" not in row:
            return "(not stored yet)"
        plan = self.plan(str(row["guild_id"]))
        return f"tier={plan['tier']} standing={plan['standing']}"

    def lifecycle_step(
        self,
        label: str,
        *,
        subscription_id: str,
        guild: str,
        clock: str | None = None,
        frozen_time: int | None = None,
        action: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Run one step: act, advance a clock, follow it on the bot side, and record the evidence.

        Parameters
        ----------
        label
            Printed step name.
        subscription_id, guild
            What to observe.
        clock, frozen_time
            A test clock to advance, and where to.
        action
            Something to do before advancing.

        Returns
        -------
        dict[str, Any]
            The subscription as Stripe returns it afterwards.
        """
        self.report.note(f"--- {label}")
        began = int(time.time()) - 2
        if action is not None:
            action()
        if clock is not None and frozen_time is not None:
            now = self.cli.advance(clock, frozen_time)
            self.set_bot_clock(now)
            self.report.note(f"  test clock at {_utc(now)}")
        self.settle()
        if clock is not None and frozen_time is not None:
            self._let_a_renewal_draft_finalize(subscription_id, clock, frozen_time)
        for event in self.cli.events_since(began):
            if event["type"] in HANDLED_EVENT_TYPES and subscription_id in json.dumps(
                event["data"]["object"]
            ):
                self.report.note(
                    f"  event {event['id']} {event['type']} (in ledger: {event['id'] in self.ledger()})"
                )
        payload = self.subscription(subscription_id)
        (self.work / "raw" / f"subscription_{label}.json").write_text(
            self.secrets.redact(json.dumps(payload, indent=1))
        )
        snapshot = self.parse(payload)
        row = self.stored(subscription_id) or {}
        latest = payload.get("latest_invoice") or {}
        self.report.note(
            f"  Stripe: status={payload['status']} latest_invoice={latest.get('status')} "
            f"({latest.get('billing_reason')}) attempts={latest.get('attempt_count')}"
        )
        self.report.note(
            f"  parser: status={snapshot.status} latest_invoice_status={snapshot.latest_invoice_status} "
            f"on_pro_price={snapshot.on_pro_price} period {_utc(snapshot.current_period_start)} -> "
            f"{_utc(snapshot.current_period_end)}"
        )
        plan = self.plan(guild)
        self.report.note(
            f"  stored: status={row.get('status')} latest_invoice_status={row.get('latest_invoice_status')} "
            f"past_due_since={_utc(row.get('past_due_since'))} version={row.get('version')}"
        )
        self.report.note(
            f"  gate:   tier={plan['tier']} standing={plan['standing']} "
            f"access_until={_utc(plan['access_until'])} paid_through={_utc(plan['paid_through'])}"
        )
        self.report.check(
            "stored status equals Stripe's",
            row.get("status") == payload["status"],
            f"{row.get('status')} vs {payload['status']}",
        )
        return payload


def _utc(seconds: object) -> str:
    if not isinstance(seconds, int):
        return "None"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(seconds))


def _fabricated(prefix: str) -> str:
    return prefix + "".join(secrets.choice(_ALPHANUMERIC) for _ in range(99))


# --------------------------------------------------------------------------- steps


def step_safety(run: Run) -> None:
    """Refuse to go on unless every sandbox precondition holds."""
    report, bag = run.report, run.secrets
    report.section("STEP 0 -- sandbox-only preconditions")
    key = bag.restricted_key
    report.note(f"  AURA_WEB_STRIPE_SECRET_KEY: length {len(key)}, prefix {key[:8]}")
    if not report.check(
        "the application's key is a test-mode key (rk_test_/sk_test_)",
        key.startswith(("rk_test_", "sk_test_")),
    ):
        raise SandboxRefusal("not a test-mode key")
    live = [
        name
        for name, env in (("web/.env", bag.web), (".env", bag.root))
        for value in env.values()
        if _LIVE_VALUE.search(value)
    ]
    if not report.check("no live key VALUE in web/.env or .env", not live, ", ".join(live)):
        raise SandboxRefusal("a live key value is present")
    probe = httpx.post(
        f"{STRIPE_API}/v1/payouts",
        headers=run.key_headers(),
        data={"aura_verification_probe": "1"},
        timeout=30,
    )
    message = str((probe.json().get("error") or {}).get("message", ""))
    accounts = set(re.findall(r"acct_[A-Za-z0-9]+", message))
    if not report.check(
        "the key cannot create payouts, and Stripe names its account as the sandbox",
        probe.status_code == 403 and accounts == {EXPECTED_ACCOUNT},
        f"HTTP {probe.status_code}, accounts {sorted(accounts)}",
    ):
        raise SandboxRefusal("the key's account or its breadth is not what this run requires")
    account = run.cli.get("/v1/account")
    report.check(
        "the Stripe CLI is logged in to the same sandbox", account.get("id") == EXPECTED_ACCOUNT
    )
    listeners = subprocess.run(
        ["pgrep", "-f", "stripe listen"], capture_output=True, text=True
    ).stdout.split()
    if not report.check("a `stripe listen` process is running", bool(listeners)):
        raise SandboxRefusal("start `stripe listen` first (see this script's docstring)")
    printed = subprocess.run(
        ["stripe", "listen", "--print-secret"], capture_output=True, text=True, timeout=60
    ).stdout.strip()
    report.note(
        f"  listener secret fingerprint {fingerprint(printed)}, configured {fingerprint(bag.webhook_secret)}"
    )
    if not report.check(
        "the listener's signing secret equals AURA_WEB_STRIPE_WEBHOOK_SECRET (by hash)",
        hmac.compare_digest(printed, bag.webhook_secret),
    ):
        raise SandboxRefusal("the listener signs with a different secret than web/.env holds")
    report.check(
        "INTERNAL_API_SECRET equals AURA_WEB_BOT_INTERNAL_API_SECRET (by hash)",
        hmac.compare_digest(
            bag.internal_secret, bag.web.get("AURA_WEB_BOT_INTERNAL_API_SECRET", "")
        ),
    )


def step_permissions(run: Run) -> None:
    """A1: what the application's key may do, probed without creating anything."""
    report = run.report
    report.section("STEP A1 -- the restricted key's permissions (probes create nothing)")
    report.note(
        "  A POST with an unknown parameter is refused by validation (400) when permitted and by the"
    )
    report.note("  permission check (403) when not; Stripe checks the permission first.")
    expectations = [
        ("POST", "/v1/checkout/sessions", True),
        ("POST", "/v1/billing_portal/sessions", True),
        ("GET", "/v1/subscriptions", True),
        ("GET", "/v1/invoices", True),
        ("POST", "/v1/subscriptions", False),
        ("GET", "/v1/customers", False),
        ("GET", "/v1/refunds", False),
        ("POST", "/v1/webhook_endpoints", False),
        ("GET", "/v1/balance", False),
        ("GET", "/v1/events", False),
    ]
    for method, path, allowed in expectations:
        if method == "POST":
            response = httpx.post(
                f"{STRIPE_API}{path}",
                headers=run.key_headers(),
                data={"aura_verification_probe": "1"},
                timeout=30,
            )
        else:
            response = httpx.get(
                f"{STRIPE_API}{path}", headers=run.key_headers(), params={"limit": "1"}, timeout=30
            )
        granted = response.status_code in (200, 400)
        report.check(
            f"{method} {path}: {'allowed' if allowed else 'denied'}",
            granted == allowed,
            f"HTTP {response.status_code}",
        )
    listing = httpx.get(
        f"{STRIPE_API}/v1/subscriptions",
        headers=run.key_headers(),
        params=(("limit", "1"), ("status", "all"), ("expand[]", "data.latest_invoice")),
        timeout=30,
    )
    report.check(
        "expanding latest_invoice is permitted (it needs Invoices: Read)",
        listing.status_code == 200,
        f"HTTP {listing.status_code}",
    )
    report.check(
        "the backend's startup self-check found the Invoices (read) permission (V-02)",
        _wait_for_log_line(
            run.work / "logs" / "backend.log", "Stripe key self-check passed", timeout=30
        ),
    )


def _wait_for_log_line(path: Path, needle: str, *, timeout: float) -> bool:
    """Whether `needle` appears in the log at `path` within `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and needle in path.read_text(errors="replace"):
            return True
        time.sleep(0.5)
    return False


def detect_managed_payments_default(run: Run) -> bool | None:
    """Whether the account's Managed Payments default is on, found by Stripe's own answer.

    Returns
    -------
    bool or None
        True when Stripe refuses a card-only checkout that does not switch
        Managed Payments off, False when it accepts one (the session is then
        expired at once), None for any other answer.
    """
    report = run.report
    form = {
        "mode": "subscription",
        "line_items[0][price]": run.secrets.price_id,
        "line_items[0][quantity]": "1",
        "success_url": "http://localhost:3000/?checkout=success",
        "payment_method_types[0]": "card",
    }
    response = httpx.post(
        f"{STRIPE_API}/v1/checkout/sessions", headers=run.key_headers(), data=form, timeout=30
    )
    if response.status_code == 200:
        session_id = response.json()["id"]
        run.cli.post(f"/v1/checkout/sessions/{session_id}/expire")
        report.note("  probe without managed_payments[enabled]: 200 (session expired at once)")
        return False
    error = (response.json() or {}).get("error") or {}
    fields = {key: error.get(key) for key in ("type", "code", "param", "decline_code")}
    report.note(
        f"  probe without managed_payments[enabled]: HTTP {response.status_code}, "
        f"error fields present {sorted(error)}, values (message aside) {fields}"
    )
    report.note(f"  Stripe's message: {str(error.get('message', ''))[:400]}")
    if response.status_code == 400 and "Managed Payments" in str(error.get("message", "")):
        return True
    return None


def step_checkout(run: Run) -> None:
    """A3: a Checkout Session through the real web service; its real object read back."""
    report = run.report
    report.section("STEP A3 -- POST /api/billing/checkout through Next.js, as a browser does")
    default_on = detect_managed_payments_default(run)
    report.note(
        "  the account's Managed Payments default (detected): "
        + {True: "ON", False: "OFF", None: "UNKNOWN"}[default_on]
    )
    if run.expected_managed_payments_default is not None:
        report.check(
            "the detected Managed Payments default is the one the operator set",
            default_on is run.expected_managed_payments_default,
            f"expected {'ON' if run.expected_managed_payments_default else 'OFF'}",
        )
    run.login()
    plans = run.browser("GET", "/api/billing/guilds")
    report.note(f"  GET /api/billing/guilds -> {plans.status_code} {plans.text[:160]}")
    response = run.browser("POST", "/api/billing/checkout", {"guild_id": FIXTURE_GUILD})
    report.note(f"  POST /api/billing/checkout -> {response.status_code} {response.text[:120]}")
    if response.status_code == 409:
        report.not_verifiable(
            "a fresh Checkout Session for the fixture guild",
            "the guild already holds a subscription the bot knows (already_subscribed)",
        )
        return
    state = {True: "ON", False: "OFF", None: "UNKNOWN"}[default_on]
    if not report.check(
        f"the app created a session (200) with the Managed Payments default {state}",
        response.status_code == 200,
        response.text[:200],
    ):
        refusals = [
            line
            for line in (run.work / "logs" / "backend.log").read_text(errors="replace").splitlines()
            if "Stripe refused a checkout" in line
        ]
        report.note(f"  backend's log of the refusal: {refusals[-1] if refusals else '(none)'}")
        form = {
            "mode": "subscription",
            "line_items[0][price]": run.secrets.price_id,
            "line_items[0][quantity]": "1",
            "success_url": "http://localhost:3000/?checkout=success",
            "payment_method_types[0]": "card",
            "integration_identifier": "aura_guild_pro_checkout_qxmvrtlk",
        }
        reproduction = httpx.post(
            f"{STRIPE_API}/v1/checkout/sessions", headers=run.key_headers(), data=form, timeout=30
        )
        report.note(
            f"  Stripe's reason (reproduced, nothing created): {reproduction.json().get('error', {}).get('message', '')[:400]}"
        )
        return
    session_id = re.search(r"cs_test_[A-Za-z0-9]+", response.text)
    assert session_id is not None
    session = run.cli.get(f"/v1/checkout/sessions/{session_id.group(0)}")
    report.check(
        "payment_method_types echoes exactly ['card']", session["payment_method_types"] == ["card"]
    )
    report.check(
        "integration_identifier is echoed",
        session.get("integration_identifier") == "aura_guild_pro_checkout_qxmvrtlk",
    )
    report.check(
        "session metadata carries the guild and the purchaser",
        session["metadata"]
        == {"aura_guild_id": FIXTURE_GUILD, "aura_discord_user_id": FIXTURE_USER},
    )
    report.check(
        "the session was created with Managed Payments off (V-01)",
        session.get("managed_payments") == {"enabled": False},
        str(session.get("managed_payments")),
    )
    run.cli.post(f"/v1/checkout/sessions/{session['id']}/expire")
    report.note("  the session was expired again (nobody pays in an automated run)")


@dataclass
class Clock:
    """One test clock with one customer holding a working and an always-failing card."""

    clock_id: str
    customer: str
    visa: str
    failing: str


def new_clock(run: Run, name: str) -> Clock:
    """Create a test clock, a customer on it, and both test cards attached (visa default)."""
    now = int(time.time())
    clock = run.cli.post(
        "/v1/test_helpers/test_clocks",
        {"frozen_time": str(now), "name": name},
        record=f"clock {name}",
    )
    customer = run.cli.post("/v1/customers", {"test_clock": clock["id"], "name": name})
    visa = run.cli.post("/v1/payment_methods/pm_card_visa/attach", {"customer": customer["id"]})
    failing = run.cli.post(
        "/v1/payment_methods/pm_card_chargeCustomerFail/attach", {"customer": customer["id"]}
    )
    run.cli.post(
        f"/v1/customers/{customer['id']}", {"invoice_settings[default_payment_method]": visa["id"]}
    )
    run.set_bot_clock(now)
    return Clock(clock["id"], customer["id"], visa["id"], failing["id"])


def aura_subscription(
    run: Run,
    customer: str,
    guild: str,
    *,
    record: str,
    extra: dict[str, str] | None = None,
    items: list[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """A subscription carrying exactly the metadata create_checkout_session's subscription_data writes."""
    params: dict[str, str] = {
        "customer": customer,
        "metadata[aura_guild_id]": guild,
        "metadata[aura_discord_user_id]": FIXTURE_USER,
        **(extra or {}),
    }
    for index, (price, quantity) in enumerate(items or [(run.secrets.price_id, 1)]):
        params[f"items[{index}][price]"] = price
        params[f"items[{index}][quantity]"] = str(quantity)
    subscription: dict[str, Any] = run.cli.post("/v1/subscriptions", params, record=record)
    return subscription


def step_first_subscription_and_tamper(run: Run) -> tuple[Clock, dict[str, Any]]:
    """A2/B5 without a browser, A4 on its captured deliveries, B7 by resending one."""
    report = run.report
    report.section(
        "STEP B5/A2 -- a subscription with the app's metadata reaches the bot through real webhooks"
    )
    report.note("  Created by the CLI instead of by paying on the hosted page: it differs from a")
    report.note(
        "  Checkout-created subscription only in how it was created (metadata, price and shape are the same)."
    )
    clock = new_clock(run, "aura-verify-lifecycle")
    subscription = aura_subscription(
        run, clock.customer, run.guild(1100), record="lifecycle subscription (guild +1100)"
    )
    run.settle()
    payload = run.lifecycle_step(
        "B5_created", subscription_id=subscription["id"], guild=run.guild(1100)
    )
    item = payload["items"]["data"][0]
    report.check(
        "every item carries price as an object with an id, and a quantity",
        isinstance(item.get("price"), dict)
        and "id" in item["price"]
        and isinstance(item.get("quantity"), int),
    )
    report.check(
        "the billing period sits on the item, not on the subscription",
        "current_period_end" in item and "current_period_end" not in payload,
    )
    report.check(
        "the invoice names its subscription under parent.subscription_details",
        ((payload["latest_invoice"].get("parent") or {}).get("subscription_details") or {}).get(
            "subscription"
        )
        == subscription["id"],
    )
    report.check(
        "the gate says Pro with basis subscription",
        run.plan(run.guild(1100))["tier"] == "pro"
        and run.plan(run.guild(1100))["basis"] == "subscription",
    )
    report.note(f"  /aura-plan: {run.plan_text(int(run.guild(1100)), datetime.now(UTC))}")

    report.section(
        "STEP A4 -- tampering with a real, correctly signed delivery (full chain via Next.js)"
    )
    body, header = _newest_accepted_delivery(run)
    parts = dict(part.split("=", 1) for part in header.split(",") if "=" in part)
    stamp = int(parts["t"])
    expected = hmac.new(
        run.secrets.webhook_secret.encode(), f"{stamp}.".encode() + body, hashlib.sha256
    ).hexdigest()
    report.check(
        "the captured bytes are exactly what Stripe signed (HMAC equals v1)",
        hmac.compare_digest(expected, parts["v1"]),
    )
    before = run.db_fingerprint()
    index = body.index(b'"livemode"')
    variants = [
        (
            "one byte altered",
            body[: index + 1] + bytes([body[index + 1] ^ 0x20]) + body[index + 2 :],
            header,
        ),
        ("trailing newline appended", body + b"\n", header),
        ("timestamp +1 s", body, header.replace(f"t={stamp}", f"t={stamp + 1}")),
        (
            "v1 digit changed",
            body,
            header.replace(
                parts["v1"], parts["v1"][:-1] + ("0" if parts["v1"][-1] != "0" else "1")
            ),
        ),
        ("no Stripe-Signature header", body, None),
    ]
    for label, variant_body, variant_header in variants:
        headers = {"Content-Type": "application/json"}
        if variant_header is not None:
            headers["Stripe-Signature"] = variant_header
        response = httpx.post(
            f"http://localhost:{CAPTURE_PORT}/api/stripe/webhook",
            content=variant_body,
            headers=headers,
            timeout=30,
        )
        report.check(
            f"{label}: refused with 400",
            response.status_code == 400,
            f"HTTP {response.status_code}",
        )
    report.check("no state changed after the forgeries", run.db_fingerprint() == before)
    genuine = httpx.post(
        f"http://localhost:{CAPTURE_PORT}/api/stripe/webhook",
        content=body,
        headers={"Content-Type": "application/json", "Stripe-Signature": header},
        timeout=30,
    )
    report.check(
        "contrast: the unmodified body is accepted (as a duplicate)",
        genuine.status_code == 200 and "duplicate" in genuine.text,
        genuine.text,
    )
    report.check("and it changed nothing either", run.db_fingerprint() == before)

    report.section("STEP B7 -- a real redelivery (stripe events resend) has exactly-once effect")
    created = next(
        e
        for e in run.cli.events_since(int(time.time()) - 900)
        if e["type"] == "customer.subscription.created"
        and e["data"]["object"]["id"] == subscription["id"]
    )
    before = run.db_fingerprint()
    run.cli.resend(created["id"])
    run.settle(quiet=5)
    report.check("the resent event changed nothing", run.db_fingerprint() == before)
    return clock, subscription


def _newest_accepted_delivery(run: Run) -> tuple[bytes, str]:
    captures = run.work / "captures"
    for response_file in sorted(captures.glob("*.response.json"), reverse=True):
        stamp = response_file.name[: -len(".response.json")]
        if json.loads(response_file.read_text())["status"] != 200:
            continue
        headers = dict(json.loads((captures / f"{stamp}.headers.json").read_text()))
        signature = next(v for k, v in headers.items() if k.lower() == "stripe-signature")
        return (captures / f"{stamp}.body").read_bytes(), signature
    raise RuntimeError("no accepted delivery was captured")


def step_lifecycle(run: Run, clock: Clock, subscription: dict[str, Any]) -> None:
    """C8-C12 on one test clock: renewal, failure and grace, retries, rollover, write-off, recovery."""
    report = run.report
    sub_id = subscription["id"]
    period_end = int(subscription["items"]["data"][0]["current_period_end"])

    report.section("STEP C8 -- a successful renewal")
    payload = run.lifecycle_step(
        "C08_renewal",
        subscription_id=sub_id,
        guild=run.guild(1100),
        clock=clock.clock_id,
        frozen_time=period_end + PAST_BOUNDARY_SECONDS,
    )
    report.check(
        "renewed: active, paid, next period stored",
        payload["status"] == "active"
        and payload["latest_invoice"]["status"] == "paid"
        and run.plan(run.guild(1100))["tier"] == "pro",
    )
    period_end = int(payload["items"]["data"][0]["current_period_end"])

    report.section("STEP C9 -- a failed renewal: past_due, the anchor, the grace boundary")
    payload = run.lifecycle_step(
        "C09_failed",
        subscription_id=sub_id,
        guild=run.guild(1100),
        clock=clock.clock_id,
        frozen_time=period_end + PAST_BOUNDARY_SECONDS,
        action=lambda: run.cli.post(
            f"/v1/customers/{clock.customer}",
            {"invoice_settings[default_payment_method]": clock.failing},
        ),
    )
    failed_period_end = int(payload["items"]["data"][0]["current_period_end"])
    row = run.stored(sub_id) or {}
    report.check(
        "past_due, anchored at the unpaid period's start",
        payload["status"] == "past_due" and row.get("past_due_since") == period_end,
    )
    grace_end = period_end + int(PAYMENT_GRACE.total_seconds())
    for offset, tier in ((-1, "pro"), (0, "free"), (1, "free")):
        run.set_bot_clock(grace_end + offset)
        report.check(
            f"gate at grace end {offset:+d} s: {tier}", run.plan(run.guild(1100))["tier"] == tier
        )
    report.note(
        f"  /aura-plan inside the grace: {run.plan_text(int(run.guild(1100)), datetime.fromtimestamp(grace_end - 60, tz=UTC))}"
    )
    payload = run.lifecycle_step(
        "C09_retries_exhausted",
        subscription_id=sub_id,
        guild=run.guild(1100),
        clock=clock.clock_id,
        frozen_time=period_end + 15 * DAY,
    )
    row = run.stored(sub_id) or {}
    report.check("retries never moved the anchor", row.get("past_due_since") == period_end)
    report.note(f"  Stripe's final action after the last retry: status={payload['status']}")
    if payload["status"] != "past_due":
        report.not_verifiable(
            "C10-C12 (rollover while past_due, write-off, recovery)",
            f"the account's retry setting ends in {payload['status']!r}; set 'If all retries for a "
            "payment fail' to 'Leave the subscription past-due' to run them",
        )
        return

    report.section("STEP C10 -- the period rolls over while past_due: no new grace")
    payload = run.lifecycle_step(
        "C10_rollover",
        subscription_id=sub_id,
        guild=run.guild(1100),
        clock=clock.clock_id,
        frozen_time=failed_period_end + PAST_BOUNDARY_SECONDS,
    )
    row = run.stored(sub_id) or {}
    new_start = int(payload["items"]["data"][0]["current_period_start"])
    report.check(
        "a new period began while past_due",
        new_start > period_end and payload["status"] == "past_due",
    )
    report.check("the anchor did not move", row.get("past_due_since") == period_end)
    report.check("no new grace: Free", run.plan(run.guild(1100))["tier"] == "free")

    report.section(
        "STEP C11 -- write-off: Stripe's status after it, and the anchor at the next failure"
    )
    latest = payload["latest_invoice"]["id"]
    payload = run.lifecycle_step(
        "C11_write_off",
        subscription_id=sub_id,
        guild=run.guild(1100),
        action=lambda: run.cli.post(f"/v1/invoices/{latest}/mark_uncollectible"),
    )
    report.note(
        f"  observed: marking the latest invoice uncollectible leaves the subscription {payload['status']!r}"
    )
    report.check(
        "written off: the gate grants nothing, the anchor survives",
        run.plan(run.guild(1100))["tier"] == "free"
        and (run.stored(sub_id) or {}).get("past_due_since") == period_end,
    )
    next_end = int(payload["items"]["data"][0]["current_period_end"])
    payload = run.lifecycle_step(
        "C11_next_failure",
        subscription_id=sub_id,
        guild=run.guild(1100),
        clock=clock.clock_id,
        frozen_time=next_end + PAST_BOUNDARY_SECONDS,
    )
    report.check(
        "the next failure kept the old anchor (no fresh grace)",
        (run.stored(sub_id) or {}).get("past_due_since") == period_end
        and run.plan(run.guild(1100))["tier"] == "free",
    )

    report.section("STEP C12 -- recovery clears the anchor; a later failure earns its own grace")
    latest = payload["latest_invoice"]["id"]
    payload = run.lifecycle_step(
        "C12_paid",
        subscription_id=sub_id,
        guild=run.guild(1100),
        action=lambda: run.cli.post(f"/v1/invoices/{latest}/pay", {"payment_method": clock.visa}),
    )
    report.check(
        "paid: anchor cleared, Pro",
        (run.stored(sub_id) or {}).get("past_due_since") is None
        and run.plan(run.guild(1100))["tier"] == "pro",
    )
    later_end = int(payload["items"]["data"][0]["current_period_end"])
    run.lifecycle_step(
        "C12_later_failure",
        subscription_id=sub_id,
        guild=run.guild(1100),
        clock=clock.clock_id,
        frozen_time=later_end + PAST_BOUNDARY_SECONDS,
    )
    report.check(
        "the later failure is anchored at its own period start",
        (run.stored(sub_id) or {}).get("past_due_since") == later_end
        and run.plan(run.guild(1100))["standing"] == "payment_grace",
    )


def step_cancel_at_period_end(run: Run) -> None:
    """C13: cancel at period end ends Pro exactly at the period end; then the deletion event."""
    report = run.report
    report.section(
        "STEP C13 -- cancel_at_period_end, the period end, customer.subscription.deleted"
    )
    clock = new_clock(run, "aura-verify-cancel")
    subscription = aura_subscription(
        run, clock.customer, run.guild(1300), record="cancel subscription (guild +1300)"
    )
    run.settle()
    payload = run.lifecycle_step(
        "C13_cancel_requested",
        subscription_id=subscription["id"],
        guild=run.guild(1300),
        action=lambda: run.cli.post(
            f"/v1/subscriptions/{subscription['id']}", {"cancel_at_period_end": "true"}
        ),
    )
    end = int(payload["items"]["data"][0]["current_period_end"])
    report.note(
        f"  real shape: cancel_at={_utc(payload.get('cancel_at'))} alongside cancel_at_period_end=true"
    )
    report.check(
        "standing canceling, access until exactly the period end",
        run.plan(run.guild(1300))["standing"] == "canceling"
        and run.plan(run.guild(1300))["access_until"] == end,
    )
    for offset, tier in ((-1, "pro"), (0, "free")):
        run.set_bot_clock(end + offset)
        report.check(
            f"gate at period end {offset:+d} s: {tier}", run.plan(run.guild(1300))["tier"] == tier
        )
    payload = run.lifecycle_step(
        "C13_after_end",
        subscription_id=subscription["id"],
        guild=run.guild(1300),
        clock=clock.clock_id,
        frozen_time=end + 3600,
    )
    report.check(
        "Stripe deleted it; stored canceled; Free",
        payload["status"] == "canceled" and run.plan(run.guild(1300))["tier"] == "free",
    )


def step_wrong_price_retry_race(run: Run) -> None:
    """C14 wrong price, C15 retry behaviour, C16 / Attack 4 races -- no clock needed."""
    report = run.report
    report.section("STEP C14 -- subscriptions carrying Aura metadata but not on the Pro price")
    product = run.cli.post(
        "/v1/products", {"name": "Aura Verify Other Plan (sandbox test)"}, record="second product"
    )
    price = run.cli.post(
        "/v1/prices",
        {
            "product": product["id"],
            "unit_amount": "199",
            "currency": "eur",
            "recurring[interval]": "month",
        },
        record="second price",
    )
    customer = run.cli.post(
        "/v1/customers", {"name": "aura-verify-no-clock"}, record="customer without clock"
    )
    visa = run.cli.post("/v1/payment_methods/pm_card_visa/attach", {"customer": customer["id"]})
    run.cli.post(
        f"/v1/customers/{customer['id']}", {"invoice_settings[default_payment_method]": visa["id"]}
    )
    run.set_bot_clock(None)
    other = aura_subscription(
        run, customer["id"], run.guild(1400), record="other price", items=[(price["id"], 1)]
    )
    mixed = aura_subscription(
        run,
        customer["id"],
        run.guild(1401),
        record="Pro + other",
        items=[(run.secrets.price_id, 1), (price["id"], 1)],
    )
    switched = aura_subscription(run, customer["id"], run.guild(1402), record="Pro, then switched")
    run.settle()
    item_id = run.cli.get(f"/v1/subscriptions/{switched['id']}")["items"]["data"][0]["id"]
    run.cli.post(
        f"/v1/subscriptions/{switched['id']}",
        {"items[0][id]": item_id, "items[0][price]": price["id"], "proration_behavior": "none"},
    )
    run.settle()
    for guild, sub in (
        (run.guild(1400), other),
        (run.guild(1401), mixed),
        (run.guild(1402), switched),
    ):
        snapshot = run.parse(run.subscription(sub["id"]))
        report.check(
            f"guild {guild}: real parser on_pro_price=False, gate Free",
            snapshot.on_pro_price is False and run.plan(guild)["tier"] == "free",
        )

    report.section("STEP C15 -- the bot's API is down: 503, then recovery by resend")
    run.stack.stop("bot")
    began = time.time()
    run.cli.post(
        f"/v1/subscriptions/{switched['id']}",
        {
            "items[0][id]": item_id,
            "items[0][price]": run.secrets.price_id,
            "items[0][quantity]": "0",
            "proration_behavior": "none",
        },
    )
    time.sleep(12)
    answered = [
        json.loads(f.read_text())
        for f in sorted((run.work / "captures").glob("*.response.json"))
        if float(f.name.split(".response")[0]) >= began
    ]
    report.check(
        "the delivery was answered 503",
        any(a["status"] == 503 for a in answered),
        json.dumps(answered)[:200],
    )
    report.not_verifiable(
        "Stripe's retry schedule and endpoint-disabling threshold",
        "a CLI listener is not a registered endpoint (pending_webhooks=0); Stripe does not retry to it",
    )
    event = next(
        e
        for e in run.cli.events_since(int(began) - 2)
        if e["type"] == "customer.subscription.updated"
        and e["data"]["object"]["id"] == switched["id"]
    )
    run.stack.start("bot")
    run.cli.resend(event["id"])
    run.settle(quiet=5)
    report.check(
        "after restart and resend the event is applied", event["id"] in run.ledger(switched["id"])
    )
    report.check(
        "Pro price at quantity 0 grants nothing",
        run.parse(run.subscription(switched["id"])).on_pro_price is False,
    )

    report.section("STEP C16 / ATTACK 4 -- racing real state changes; read-after-write")
    racer = aura_subscription(
        run, customer["id"], run.guild(1500), record="race subscription (guild +1500)"
    )
    run.settle()
    began_race = int(time.time()) - 2
    path = f"/v1/subscriptions/{racer['id']}"
    run.cli.post(path, {"cancel_at_period_end": "true"})
    run.cli.post(path, {"pause_collection[behavior]": "mark_uncollectible"})
    run.cli.post(path, {"pause_collection": ""})
    run.cli.post(path, {"cancel_at_period_end": "false"})
    run.cli.call("delete", path)
    run.settle(timeout=120, quiet=10)
    final = run.subscription(racer["id"])
    row = run.stored(racer["id"]) or {}
    report.check(
        "stored end state equals Stripe's",
        row.get("status") == final["status"]
        and bool(row.get("cancel_at_period_end")) == final["cancel_at_period_end"]
        and bool(row.get("collection_paused")) == (final["pause_collection"] is not None),
    )
    emitted = {
        e["id"]
        for e in run.cli.events_since(began_race - 60)
        if e["type"] in HANDLED_EVENT_TYPES and racer["id"] in json.dumps(e["data"]["object"])
    }
    ledger = run.ledger(racer["id"])
    report.check(
        "every handled event in the ledger exactly once",
        emitted <= set(ledger) and len(ledger) == len(set(ledger)),
        f"missing {sorted(emitted - set(ledger))}",
    )
    reader = aura_subscription(
        run, customer["id"], run.guild(1501), record="read-after-write subscription (guild +1501)"
    )
    stale = 0
    for number in range(1, 26):
        run.cli.post(f"/v1/subscriptions/{reader['id']}", {"metadata[aura_seq]": str(number)})
        read = httpx.get(
            f"{STRIPE_API}/v1/subscriptions/{reader['id']}",
            headers=run.key_headers(),
            params=(("expand[]", "latest_invoice"),),
            timeout=30,
        ).json()
        stale += read["metadata"].get("aura_seq") != str(number)
    report.check(
        "25 writes each read back immediately with the app's key: no stale read",
        stale == 0,
        f"{stale} stale",
    )
    report.note("  (absence of a stale read in 25 tries is evidence, not proof)")


def step_statuses(run: Run) -> None:
    """A2: trialing, paused, incomplete and incomplete_expired through the real parser."""
    report = run.report
    report.section("STEP A2 -- statuses not produced by the lifecycle")
    now = int(time.time())
    clock = run.cli.post(
        "/v1/test_helpers/test_clocks",
        {"frozen_time": str(now), "name": "aura-verify-statuses"},
        record="clock aura-verify-statuses",
    )
    trial_customer = run.cli.post("/v1/customers", {"test_clock": clock["id"], "name": "trial"})
    incomplete_customer = run.cli.post(
        "/v1/customers", {"test_clock": clock["id"], "name": "incomplete"}
    )
    run.set_bot_clock(now)
    trial = aura_subscription(
        run,
        trial_customer["id"],
        run.guild(1600),
        record="trial",
        extra={
            "trial_period_days": "1",
            "trial_settings[end_behavior][missing_payment_method]": "pause",
        },
    )
    incomplete = aura_subscription(
        run,
        incomplete_customer["id"],
        run.guild(1601),
        record="incomplete",
        extra={"payment_behavior": "default_incomplete"},
    )
    run.settle()
    for sub, expected in ((trial, "trialing"), (incomplete, "incomplete")):
        snapshot = run.parse(run.subscription(sub["id"]))
        report.check(f"{expected}: parsed", snapshot.status == expected)
    run.set_bot_clock(run.cli.advance(clock["id"], now + 2 * DAY))
    run.settle()
    for sub, expected, guild in (
        (trial, "paused", run.guild(1600)),
        (incomplete, "incomplete_expired", run.guild(1601)),
    ):
        snapshot = run.parse(run.subscription(sub["id"]))
        report.check(
            f"{expected}: parsed, stored, gate Free",
            snapshot.status == expected
            and (run.stored(sub["id"]) or {}).get("status") == expected
            and run.plan(guild)["tier"] == "free",
        )


def step_wrong_credentials(run: Run) -> list[str]:
    """Attack 3: a wrong restricted key (503), a live-looking key (refused, no connection)."""
    report = run.report
    report.section("ATTACK 3 -- a wrong restricted key, and a live-looking key")
    wrong_key, live_key = _fabricated("rk_test_"), _fabricated("sk_live_")
    environment = {**run.stack.commands()["backend"][2], "AURA_WEB_STRIPE_SECRET_KEY": wrong_key}
    log_path = run.work / "logs" / "backend_wrong_key.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "verify_stripe_sandbox:build_backend",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(WRONG_KEY_PORT),
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while not _listening(WRONG_KEY_PORT) and time.monotonic() < deadline:
                time.sleep(0.2)
            some_subscription = next(s for s, label in run.cli.created if s.startswith("sub_"))
            event = {
                "id": "evt_auraVerifyWrongKey" + secrets.token_hex(6),
                "object": "event",
                "livemode": False,
                "type": "customer.subscription.updated",
                "created": int(time.time()),
                "api_version": STRIPE_VERSION,
                "data": {"object": {"id": some_subscription, "object": "subscription"}},
            }
            body = json.dumps(event).encode()
            stamp = int(time.time())
            digest = hmac.new(
                run.secrets.webhook_secret.encode(), f"{stamp}.".encode() + body, hashlib.sha256
            ).hexdigest()
            before = run.db_fingerprint()
            response = httpx.post(
                f"http://127.0.0.1:{WRONG_KEY_PORT}/api/stripe/webhook",
                content=body,
                timeout=30,
                headers={
                    "Content-Type": "application/json",
                    "Stripe-Signature": f"t={stamp},v1={digest}",
                },
            )
            report.check(
                "wrong key: a correctly signed delivery is answered 503",
                response.status_code == 503,
                response.text,
            )
            report.check("wrong key: nothing changed", run.db_fingerprint() == before)
        finally:
            process.terminate()
            process.wait(timeout=15)
    report.check(
        "wrong key: its startup self-check named a rejected key (HTTP 401), without the key",
        "Stripe rejected AURA_WEB_STRIPE_SECRET_KEY itself (HTTP 401)" in log_path.read_text(),
    )
    report.check(
        "wrong key: not in its log, not even its last 8 characters",
        wrong_key[-8:] not in log_path.read_text(),
    )

    connections: list[str] = []
    recorder = socket.socket()
    recorder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    recorder.bind(("127.0.0.1", RECORDER_PORT))
    recorder.listen()
    recorder.settimeout(0.5)
    done = threading.Event()

    def record() -> None:
        while not done.is_set():
            with suppress(OSError):
                connection, address = recorder.accept()
                connections.append(str(address))
                connection.close()

    thread = threading.Thread(target=record, daemon=True)
    thread.start()
    live_env = {
        **run.stack.commands()["backend"][2],
        "AURA_WEB_STRIPE_SECRET_KEY": live_key,
        "AURA_WEB_STRIPE_API_BASE": f"http://127.0.0.1:{RECORDER_PORT}",
        "AURA_WEB_DISCORD_API_BASE": f"http://127.0.0.1:{RECORDER_PORT}",
        "AURA_WEB_BOT_INTERNAL_API_URL": f"http://127.0.0.1:{RECORDER_PORT}",
    }
    live_log = run.work / "logs" / "backend_live_key.log"
    with live_log.open("wb") as log:
        code = subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "aura_web.app:build_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(LIVE_KEY_PORT),
            ],
            cwd=REPO_ROOT,
            env=live_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=60,
        ).returncode
    time.sleep(1)
    done.set()
    thread.join()
    recorder.close()
    text = live_log.read_text()
    report.check(
        "live-looking key: startup refused (non-zero exit, 'Startup aborted')",
        code != 0 and "Startup aborted" in text,
    )
    report.check(
        "live-looking key: zero connections to anything",
        not connections,
        f"{len(connections)} connection(s)",
    )
    report.check(
        "live-looking key: not in the log, not even its last 8 characters",
        live_key[-8:] not in text,
    )
    return [wrong_key, live_key]


def step_leak_scan(run: Run, fabricated: list[str], extra_files: list[Path]) -> None:
    """Attack 2: every produced file and the transcript, scanned for exact values, with a canary."""
    report = run.report
    report.section("ATTACK 2 -- secret leak scan by value (positive control: a planted canary)")
    values = dict(run.secrets.labelled())
    for index, value in enumerate(fabricated, start=1):
        values[f"fabricated key #{index}"] = value
    canary = "aura-canary-" + secrets.token_hex(16)
    values["positive control canary"] = canary
    planted = run.work / "logs" / "canary.log"
    planted.write_text(f"planted: {canary}\n")
    files = [p for p in run.work.rglob("*") if p.is_file() and p.suffix != ".db"] + [
        p for p in extra_files if p.exists()
    ]
    for history in (".bash_history", ".zsh_history", ".python_history"):
        if (Path.home() / history).exists():
            files.append(Path.home() / history)
    transcript = "\n".join(report.lines)
    hits: dict[str, list[str]] = {label: [] for label in values}
    for path in files:
        content = path.read_bytes().decode("utf-8", "replace")
        for label, value in values.items():
            if value in content:
                hits[label].append(path.name)
    for label, value in values.items():
        if value in transcript:
            hits[label].append("<transcript>")
    report.note(f"  files scanned: {len(files)}, values searched: {len(values)}")
    for label, where in hits.items():
        if label == "positive control canary":
            report.check(
                "positive control: the canary is found exactly where it was planted",
                where == [planted.name],
                str(where),
            )
        else:
            report.check(f"{label}: no occurrence", not where, ", ".join(where))
    planted.unlink()


def cleanup(run: Run) -> None:
    """Delete clocks (and with them their customers and subscriptions), cancel the rest, archive the extra price."""
    report = run.report
    report.section("CLEANUP -- every Stripe object this run created")
    for object_id, label in reversed(run.cli.created):
        try:
            if object_id.startswith("clock_"):
                run.cli.call("delete", f"/v1/test_helpers/test_clocks/{object_id}")
            elif object_id.startswith("cus_"):
                run.cli.call("delete", f"/v1/customers/{object_id}")
            elif object_id.startswith("price_"):
                run.cli.post(f"/v1/prices/{object_id}", {"active": "false"})
            elif object_id.startswith("prod_"):
                run.cli.post(f"/v1/products/{object_id}", {"active": "false"})
            else:
                continue
            report.note(f"  removed/archived {object_id} ({label})")
        except RuntimeError as exc:
            report.note(f"  could not remove {object_id} ({label}): {exc}")
    clocks = [o for o, _ in run.cli.created if o.startswith("clock_")]
    remaining = run.cli.get("/v1/test_helpers/test_clocks", {"limit": "100"})["data"]
    report.check(
        "none of this run's test clocks remains", not {c["id"] for c in remaining} & set(clocks)
    )


@contextmanager
def running_stack(work: Path) -> Iterator[Stack]:
    """Start the local stack and always stop it again."""
    stack = Stack(work)
    try:
        stack.start("bot", "discord", "backend", "frontend", "capture")
        yield stack
    finally:
        stack.stop()


def main() -> int:
    """Run the verification and write the transcript.

    Returns
    -------
    int
        0 when every check passed, 1 otherwise, 2 when a safety precondition refused the run.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "reports" / "phase-4c-stripe-verification-rerun.txt")
    )
    parser.add_argument(
        "--keep-objects", action="store_true", help="skip the Stripe cleanup (for inspection)"
    )
    parser.add_argument(
        "--managed-payments-default",
        choices=["on", "off"],
        help="the state the operator set in the Dashboard; the run fails if Stripe disagrees",
    )
    parser.add_argument("--serve", choices=["bot", "capture"], help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if arguments.serve is not None:
        work = Path(os.environ["SBX_WORK_DIR"])
        (serve_bot_harness if arguments.serve == "bot" else serve_capture_hop)(work)
        return 0

    bag = Secrets.load()
    report = Report(bag)
    work = Path(tempfile.mkdtemp(prefix="aura-stripe-verify-"))
    for sub in ("logs", "captures", "raw"):
        (work / sub).mkdir(mode=0o700)
    report.note("PHASE 4c-F01 -- STRIPE SANDBOX VERIFICATION (automated rerun)")
    report.note(f"  started {datetime.now(UTC):%Y-%m-%d %H:%M:%S} UTC; work directory {work}")
    cli = StripeCli(report)
    fabricated: list[str] = []
    try:
        run = Run(bag, report, cli, Stack(work), work)
        if arguments.managed_payments_default is not None:
            run.expected_managed_payments_default = arguments.managed_payments_default == "on"
        step_safety(run)
        with running_stack(work) as stack:
            run.stack = stack
            step_permissions(run)
            step_checkout(run)
            clock, subscription = step_first_subscription_and_tamper(run)
            step_lifecycle(run, clock, subscription)
            step_cancel_at_period_end(run)
            step_wrong_price_retry_race(run)
            step_statuses(run)
            fabricated = step_wrong_credentials(run)
    except SandboxRefusal as refusal:
        report.note(f"REFUSED: {refusal}")
        Path(arguments.out).write_text("\n".join(report.lines) + "\n", encoding="utf-8")
        return 2
    finally:
        if not arguments.keep_objects and cli.created:
            with suppress(Exception):
                cleanup(Run(bag, report, cli, Stack(work), work))
    output = Path(arguments.out)
    step_leak_scan(Run(bag, report, cli, Stack(work), work), fabricated, [output])
    report.section("SUMMARY")
    report.note(f"  checks passed : {report.passed}")
    report.note(f"  checks failed : {report.failed}")
    report.note(f"  not verifiable: {len(report.unverifiable)}")
    report.note("  RESULT: " + ("ALL CHECKS PASSED" if report.failed == 0 else "FAILURES PRESENT"))
    output.write_text("\n".join(report.lines) + "\n", encoding="utf-8")
    print(f"\nTranscript written to {output}; logs and raw objects in {work}")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
