"""Phase 4b live verification: a real OAuth2 round trip over real sockets.

Runs the actual backend and a Discord stand-in as two uvicorn processes on
real ports, then drives the whole flow with curl -- so every assertion below
is made against bytes that crossed a socket, not against a function's return
value. That distinction is the brief's requirement for two of its four
attacks ("no Discord token in any response to the browser -- check at the
network level, not just by reading code" and "check the cookie flags in the
actual HTTP response").

curl, specifically, rather than a Python HTTP client: it has no cookie policy
opinions to accidentally satisfy the test, it prints raw response headers
verbatim, and the transcript it produces is something a reader can re-run by
hand.

Cookies are carried forward by explicitly re-sending the values parsed out of
Set-Cookie, rather than by curl's cookie jar. The jar refuses to send a
`Secure` cookie back over http, so using it would have forced this run to
disable the very flag it exists to verify. A browser on localhost sends these
cookies (localhost is a trustworthy origin); doing it explicitly here matches
that without weakening the server's configuration.

Usage:  python scripts/verify_oauth_flow.py [--out reports/phase-4b-verification.txt]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "web" / "backend"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# web/backend is not a package this repository installs; it is put on the path
# by pytest.ini for the test suite and here for a direct `python scripts/...`
# run, so this script works either way without a PYTHONPATH incantation.
for extra_path in (str(BACKEND_DIR), str(SCRIPTS_DIR)):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from verify_oauth_fixtures import (  # imported after the sys.path setup above
    BOT_TOKEN,
    CLIENT_SECRET,
    MODERATOR_ID,
    PLAIN_MEMBER_ID,
)


@dataclass
class CurlResult:
    """One curl invocation: the command, the raw response, and the parsed parts."""

    command: list[str]
    raw: str
    status: int
    headers: list[tuple[str, str]]
    body: str

    def header(self, name: str) -> str | None:
        for header_name, value in self.headers:
            if header_name.lower() == name.lower():
                return value
        return None

    def set_cookie(self, cookie_name: str) -> str | None:
        for header_name, value in self.headers:
            if header_name.lower() == "set-cookie" and value.split("=", 1)[0] == cookie_name:
                return value
        return None


@dataclass
class Report:
    """Accumulates the transcript and the pass/fail tally."""

    lines: list[str] = field(default_factory=list)
    passed: int = 0
    failed: int = 0

    def section(self, title: str) -> None:
        self.lines.append("")
        self.lines.append("=" * 78)
        self.lines.append(title)
        self.lines.append("=" * 78)

    def note(self, text: str = "") -> None:
        self.lines.append(text)

    def transcript(self, result: CurlResult) -> None:
        self.lines.append(f"$ {' '.join(result.command)}")
        self.lines.append("")
        for line in result.raw.splitlines():
            self.lines.append(f"  | {line}")
        self.lines.append("")

    def check(self, description: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            self.lines.append(f"  [PASS] {description}")
        else:
            self.failed += 1
            self.lines.append(f"  [FAIL] {description}" + (f" -- {detail}" if detail else ""))


def free_port() -> int:
    """Ask the OS for an unused port rather than guessing one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_until_listening(port: int, *, process: subprocess.Popen[bytes], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server on port {port} exited with code {process.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"server on port {port} never started listening")


def start_server(factory: str, port: int, environment: dict[str, str]) -> subprocess.Popen[bytes]:
    child_environment = dict(os.environ)
    child_environment.update(environment)
    child_environment["PYTHONPATH"] = os.pathsep.join(
        [str(BACKEND_DIR), str(SCRIPTS_DIR), child_environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            factory,
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(REPO_ROOT),
        env=child_environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def curl(*arguments: str) -> CurlResult:
    """Run curl with raw headers included, and split the response into parts."""
    command = ["curl", "--silent", "--show-error", "--include", "--no-buffer", *arguments]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    raw = completed.stdout
    # text=True turns on universal newlines, so curl's CRLF line endings have
    # already been translated to LF by the time this sees them -- splitting on
    # CRLFCRLF would silently never match and leave every body empty.
    separator = "\r\n\r\n" if "\r\n\r\n" in raw else "\n\n"
    head, _, body = raw.partition(separator)
    head_lines = head.splitlines()
    status = int(head_lines[0].split()[1]) if head_lines else 0
    headers = [
        (line.split(":", 1)[0].strip(), line.split(":", 1)[1].strip())
        for line in head_lines[1:]
        if ":" in line
    ]
    # The printed command redacts nothing: these are throwaway fixture
    # credentials, and a transcript that hides what was sent is not evidence.
    return CurlResult(command=command, raw=raw.strip(), status=status, headers=headers, body=body)


def cookie_value(set_cookie_header: str) -> str:
    return set_cookie_header.split(";", 1)[0].split("=", 1)[1]


def flags_of(set_cookie_header: str) -> dict[str, bool]:
    lowered = set_cookie_header.lower()
    return {
        "HttpOnly": "httponly" in lowered,
        "Secure": "secure" in lowered,
        "SameSite=Lax": "samesite=lax" in lowered,
        "Path=/": "path=/" in lowered,
    }


def run_verification(report: Report, base_url: str, discord_url: str) -> None:
    secrets_seen: dict[str, str] = {
        "OAuth client secret": CLIENT_SECRET,
        "Bot token": BOT_TOKEN,
    }
    transcript_for_leak_scan: list[str] = []

    def record(result: CurlResult) -> CurlResult:
        report.transcript(result)
        transcript_for_leak_scan.append(result.raw)
        return result

    def issue_code(user_id: str) -> str:
        response = subprocess.run(
            [
                "curl",
                "--silent",
                "-X",
                "POST",
                f"{discord_url}/__fake__/issue-code",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps({"user_id": user_id}),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return str(json.loads(response.stdout)["code"])

    # ---------------------------------------------------------------- step 1
    report.section("STEP 1 -- GET /api/auth/login  (start of the flow)")
    login = record(curl(f"{base_url}/api/auth/login"))

    location = login.header("location") or ""
    state_cookie_header = login.set_cookie("aura_oauth_state") or ""
    state_in_url = ""
    match = re.search(r"[?&]state=([^&]+)", location)
    if match:
        state_in_url = match.group(1)

    report.note()
    report.check("redirects (307) to Discord's authorize page", login.status == 307, location)
    report.check("authorize URL is on discord.com", location.startswith("https://discord.com/oauth2/authorize"), location)
    report.check(
        "requests exactly the scopes identify+guilds",
        re.search(r"scope=guilds\+identify|scope=identify\+guilds", location) is not None,
        location,
    )
    report.check("carries a state parameter", bool(state_in_url), location)
    report.check("sets an aura_oauth_state cookie", bool(state_cookie_header), "no Set-Cookie")
    for flag, present in flags_of(state_cookie_header).items():
        report.check(f"state cookie carries {flag}", present, state_cookie_header)
    report.check(
        "the state in the cookie equals the state in the URL (browser binding)",
        bool(state_cookie_header) and cookie_value(state_cookie_header) == state_in_url,
    )
    report.check(
        "no client secret or bot token appears in the authorize redirect",
        all(secret not in login.raw for secret in secrets_seen.values()),
    )

    state_cookie = cookie_value(state_cookie_header) if state_cookie_header else ""

    # ---------------------------------------------------------------- step 2
    report.section("STEP 2 -- GET /api/auth/callback with a MISSING state (attack 1a)")
    missing_state = record(
        curl(
            f"{base_url}/api/auth/callback?code={issue_code(MODERATOR_ID)}",
            "-H",
            f"Cookie: aura_oauth_state={state_cookie}",
        )
    )
    report.note()
    report.check("refused with HTTP 400", missing_state.status == 400, missing_state.raw)
    report.check('error code is "invalid_state"', '"invalid_state"' in missing_state.body)
    report.check("no session cookie is issued", missing_state.set_cookie("aura_session") is None)

    # ---------------------------------------------------------------- step 3
    report.section("STEP 3 -- GET /api/auth/callback with a TAMPERED state (attack 1b)")
    tampered = state_in_url[:-1] + ("A" if state_in_url[-1] != "A" else "B")
    tampered_state = record(
        curl(
            f"{base_url}/api/auth/callback?code={issue_code(MODERATOR_ID)}&state={tampered}",
            "-H",
            f"Cookie: aura_oauth_state={state_cookie}",
        )
    )
    report.note()
    report.check("refused with HTTP 400", tampered_state.status == 400, tampered_state.raw)
    report.check('error code is "invalid_state"', '"invalid_state"' in tampered_state.body)
    report.check("no session cookie is issued", tampered_state.set_cookie("aura_session") is None)

    # ---------------------------------------------------------------- step 4
    report.section("STEP 4 -- GET /api/auth/callback with a valid state but NO state COOKIE")
    report.note("(the CSRF case: an attacker's genuinely-issued state, replayed in a")
    report.note(" browser that never received the matching cookie)")
    report.note()
    no_cookie = record(
        curl(f"{base_url}/api/auth/callback?code={issue_code(MODERATOR_ID)}&state={state_in_url}")
    )
    report.note()
    report.check("refused with HTTP 400", no_cookie.status == 400, no_cookie.raw)
    report.check('error code is "invalid_state"', '"invalid_state"' in no_cookie.body)
    report.check("no session cookie is issued", no_cookie.set_cookie("aura_session") is None)

    # -------------------------------------------------------------- step 4b
    report.section("STEP 4b -- a REPEATED state parameter (parameter pollution)")
    report.note("Starlette resolves a repeated query parameter to its LAST occurrence.")
    report.note("A callback carrying state twice is refused outright, whichever value")
    report.note("would have won -- Discord never sends one, so there is nothing to guess.")
    report.note()
    polluted_state = record(
        curl(
            f"{base_url}/api/auth/callback?code={issue_code(MODERATOR_ID)}"
            f"&state=forged&state={state_in_url}",
            "-H",
            f"Cookie: aura_oauth_state={state_cookie}",
        )
    )
    report.note()
    report.check("refused with HTTP 400", polluted_state.status == 400, polluted_state.raw)
    report.check('error code is "invalid_state"', '"invalid_state"' in polluted_state.body)
    report.check("no session cookie is issued", polluted_state.set_cookie("aura_session") is None)

    # ---------------------------------------------------------------- step 5
    report.section("STEP 5 -- GET /api/auth/callback, honest flow (moderator)")
    report.note("NOTE: steps 2-4 each consumed a state, so a fresh login leg runs first.")
    report.note()
    fresh_login = record(curl(f"{base_url}/api/auth/login"))
    fresh_state_header = fresh_login.set_cookie("aura_oauth_state") or ""
    fresh_state = cookie_value(fresh_state_header)
    code = issue_code(MODERATOR_ID)

    callback = record(
        curl(
            f"{base_url}/api/auth/callback?code={code}&state={fresh_state}",
            "-H",
            f"Cookie: aura_oauth_state={fresh_state}",
        )
    )
    session_cookie_header = callback.set_cookie("aura_session") or ""
    report.note()
    report.check("redirects (303) after a successful login", callback.status == 303, callback.raw)
    report.check(
        "redirect target is exactly the configured post-login URL, with no query string",
        callback.header("location") == "http://localhost:3000/",
        str(callback.header("location")),
    )
    report.check("issues an aura_session cookie", bool(session_cookie_header))
    for flag, present in flags_of(session_cookie_header).items():
        report.check(f"session cookie carries {flag}", present, session_cookie_header)
    report.check("clears the spent state cookie", callback.set_cookie("aura_oauth_state") is not None)
    report.check("response is marked Cache-Control: no-store", callback.header("cache-control") == "no-store")

    session_cookie = cookie_value(session_cookie_header) if session_cookie_header else ""

    # ---------------------------------------------------------------- step 6
    report.section("STEP 6 -- replaying the SAME state a second time (attack 1c)")
    replay = record(
        curl(
            f"{base_url}/api/auth/callback?code={issue_code(MODERATOR_ID)}&state={fresh_state}",
            "-H",
            f"Cookie: aura_oauth_state={fresh_state}",
        )
    )
    report.note()
    report.check("a consumed state is refused (single-use)", replay.status == 400, replay.raw)
    report.check('error code is "invalid_state"', '"invalid_state"' in replay.body)

    # ---------------------------------------------------------------- step 7
    report.section("STEP 7 -- GET /api/me  (moderator's session)")
    me = record(curl(f"{base_url}/api/me", "-H", f"Cookie: aura_session={session_cookie}"))
    report.note()
    report.check("HTTP 200", me.status == 200, me.raw)
    report.check("returns the signed-in identity", '"id": "5000"' in me.body or '"id":"5000"' in me.body, me.body)

    # ---------------------------------------------------------------- step 8
    report.section("STEP 8 -- GET /api/guilds  (attack 3: the two-condition filter)")
    report.note("Fixture: user manages 1000 (Aura present) and 2000 (Aura ABSENT);")
    report.note("         user is a plain member of 3000 (Aura present).")
    report.note("Expected: exactly guild 1000.")
    report.note()
    guilds = record(curl(f"{base_url}/api/guilds", "-H", f"Cookie: aura_session={session_cookie}"))
    report.note()
    report.check("HTTP 200", guilds.status == 200, guilds.raw)
    report.check('includes guild 1000 (manageable AND Aura present)', '"1000"' in guilds.body)
    report.check('EXCLUDES guild 2000 (manageable but Aura absent)', '"2000"' not in guilds.body, guilds.body)
    report.check('EXCLUDES guild 3000 (Aura present but not manageable)', '"3000"' not in guilds.body, guilds.body)

    # ---------------------------------------------------------------- step 9
    report.section("STEP 9 -- a user with MANAGE_GUILD nowhere (attack 3, empty case)")
    member_login = record(curl(f"{base_url}/api/auth/login"))
    member_state = cookie_value(member_login.set_cookie("aura_oauth_state") or "")
    member_callback = record(
        curl(
            f"{base_url}/api/auth/callback?code={issue_code(PLAIN_MEMBER_ID)}&state={member_state}",
            "-H",
            f"Cookie: aura_oauth_state={member_state}",
        )
    )
    member_session = cookie_value(member_callback.set_cookie("aura_session") or "")
    member_guilds = record(
        curl(f"{base_url}/api/guilds", "-H", f"Cookie: aura_session={member_session}")
    )
    report.note()
    report.check("HTTP 200, not an error", member_guilds.status == 200, member_guilds.raw)
    report.check("the list is empty", member_guilds.body.strip() == "[]", member_guilds.body)
    report.check(
        "no other user's guild appears",
        all(guild_id not in member_guilds.body for guild_id in ("1000", "2000", "3000")),
        member_guilds.body,
    )

    # --------------------------------------------------------------- step 10
    report.section("STEP 10 -- unauthenticated access")
    anonymous_me = record(curl(f"{base_url}/api/me"))
    anonymous_guilds = record(curl(f"{base_url}/api/guilds"))
    forged = record(curl(f"{base_url}/api/me", "-H", "Cookie: aura_session=forged-identifier"))
    report.note()
    report.check("/api/me without a cookie is 401", anonymous_me.status == 401)
    report.check("/api/guilds without a cookie is 401", anonymous_guilds.status == 401)
    report.check("a forged session identifier is 401", forged.status == 401)

    # --------------------------------------------------------------- step 11
    report.section("STEP 11 -- POST /api/auth/logout")
    logout = record(
        curl("-X", "POST", f"{base_url}/api/auth/logout", "-H", f"Cookie: aura_session={session_cookie}")
    )
    after_logout = record(
        curl(f"{base_url}/api/me", "-H", f"Cookie: aura_session={session_cookie}")
    )
    logout_by_get = record(curl(f"{base_url}/api/auth/logout"))
    report.note()
    report.check("logout answers 204", logout.status == 204, logout.raw)
    report.check("logout clears the session cookie", logout.set_cookie("aura_session") is not None)
    report.check("the old identifier no longer authenticates", after_logout.status == 401)
    report.check("logout is not reachable by GET (CSRF)", logout_by_get.status == 405, logout_by_get.raw)

    # --------------------------------------------------------------- step 12
    report.section("STEP 12 -- ATTACK 2: no Discord token anywhere on the wire")
    report.note("Every byte of every response captured above is scanned for the")
    report.note("credentials that existed during the run. The token values are read")
    report.note("from the stand-in Discord's own state, so this cannot pass by")
    report.note("scanning for the wrong strings.")
    report.note()

    issued = subprocess.run(
        ["curl", "--silent", f"{discord_url}/__fake__/issued-tokens"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        issued_tokens = json.loads(issued.stdout)
    except json.JSONDecodeError:
        issued_tokens = {"access_tokens": [], "refresh_tokens": []}

    for index, token in enumerate(issued_tokens.get("access_tokens", [])):
        secrets_seen[f"Discord access token #{index + 1}"] = token
    for index, token in enumerate(issued_tokens.get("refresh_tokens", [])):
        secrets_seen[f"Discord refresh token #{index + 1}"] = token

    report.note(f"  credentials in play during this run: {len(secrets_seen)}")
    for label in secrets_seen:
        report.note(f"    - {label}")
    report.note(f"  responses captured and scanned: {len(transcript_for_leak_scan)}")
    report.note()

    whole_transcript = "\n".join(transcript_for_leak_scan)
    report.check(
        "at least one Discord token was actually issued (otherwise this check is vacuous)",
        len(issued_tokens.get("access_tokens", [])) > 0,
    )
    for label, secret in secrets_seen.items():
        report.check(f"{label} never appears in any response", secret not in whole_transcript)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "reports" / "phase-4b-verification.txt"),
        help="where to write the transcript",
    )
    arguments = parser.parse_args()

    discord_port = free_port()
    backend_port = free_port()
    discord_url = f"http://127.0.0.1:{discord_port}"
    base_url = f"http://127.0.0.1:{backend_port}"

    report = Report()
    report.note("PHASE 4b -- LIVE OAUTH2 VERIFICATION")
    report.note("")
    report.note("Two real uvicorn processes on real ports; every request below issued by")
    report.note("curl and reproduced verbatim, headers included.")
    report.note("")
    report.note(f"  backend (the real aura_web app) : {base_url}")
    report.note(f"  Discord stand-in                : {discord_url}")

    discord_process = start_server("verify_oauth_fixtures:build_fake", discord_port, {})
    backend_process: subprocess.Popen[bytes] | None = None
    try:
        wait_until_listening(discord_port, process=discord_process)
        backend_process = start_server(
            "verify_oauth_fixtures:build_backend",
            backend_port,
            {
                "VERIFY_DISCORD_API_BASE": f"{discord_url}/api/v10",
                "VERIFY_REDIRECT_URI": "http://localhost:3000/api/auth/callback",
                "VERIFY_POST_LOGIN_URL": "http://localhost:3000/",
            },
        )
        wait_until_listening(backend_port, process=backend_process)
        run_verification(report, base_url, discord_url)
    finally:
        for process in (backend_process, discord_process):
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    process.kill()

    report.section("SUMMARY")
    report.note(f"  checks passed : {report.passed}")
    report.note(f"  checks failed : {report.failed}")
    report.note()
    report.note("  RESULT: " + ("ALL CHECKS PASSED" if report.failed == 0 else "FAILURES PRESENT"))

    output_path = Path(arguments.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report.lines) + "\n", encoding="utf-8")

    print("\n".join(report.lines))
    print(f"\nTranscript written to {output_path}")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
