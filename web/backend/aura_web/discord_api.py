"""Every outbound call this service makes to Discord, and nothing else.

One module so there is exactly one place where a Discord response is turned
into something the rest of the service trusts. Each response is validated
field by field rather than passed through: an upstream that changes a type,
omits a field, or is impersonated by whatever sits between us and it must
produce a clean refusal here, not a surprising value three layers up. That
is the same reasoning aura.extraction's JSON parsing already applies to model
output, applied to an API instead of an LLM.

The exception hierarchy separates the two failures that need different
answers at the HTTP boundary: DiscordAuthError means this credential is no
longer good (re-authenticate the user), DiscordUnavailableError means we
could not find out (fail the request, keep the session).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx

from aura_web.config import DISCORD_GUILD_PAGE_SIZE
from aura_web.permissions import (
    parse_snowflake,
    sanitize_guild_name,
    sanitize_icon_hash,
)
from aura_web.sessions import DiscordTokens, DiscordUser, utc_now

logger = logging.getLogger(__name__)

# The scopes this service requests, and the exact set it verifies it received.
# Minimal by requirement: identify is what /users/@me needs, guilds is what
# /users/@me/guilds needs, and there is no third thing 4b does.
REQUIRED_SCOPES = frozenset({"identify", "guilds"})

# A bot in more than this many guilds would page forever if Discord ever
# returned a cursor that does not advance. 100 pages of 200 is 20,000 guilds,
# far past anything Aura will see, so hitting it means the loop is broken
# rather than the deployment being large.
MAX_GUILD_PAGES = 100

# Discord's documented default for access tokens is a week; this only applies
# when a response omits or mangles expires_in, and is short enough that the
# refresh path gets exercised rather than a wrong long life being trusted.
FALLBACK_TOKEN_LIFETIME_SECONDS = 3600

# A guild name we could not read. Never shown in place of a name we could.
UNKNOWN_GUILD_NAME = "Unknown server"


class DiscordAPIError(Exception):
    """Base class for every failure talking to Discord."""


class DiscordAuthError(DiscordAPIError):
    """Discord rejected the credential: a bad code, a revoked or expired token.

    Distinct from unavailability because the correct response differs. This
    one means the user (or the bot) must authenticate again; retrying with the
    same credential will never work.
    """


class DiscordUnavailableError(DiscordAPIError):
    """Discord could not be reached, or answered in a way we cannot act on.

    Covers network failures, timeouts, 5xx, 429 and malformed payloads. The
    caller must fail the request rather than guess -- a guess here would mean
    showing a guild list built from incomplete knowledge of who may see what.
    """


@dataclass(frozen=True)
class PartialGuild:
    """One guild as /users/@me/guilds describes it, after validation.

    ``permissions`` stays exactly as Discord sent it (a string) rather than
    being parsed here: this module's job is to establish that the response was
    well-formed, and aura_web.permissions' job is to decide what the bitmask
    means. Splitting them keeps the authorization decision unit-testable
    without an HTTP layer.
    """

    id: str
    name: str
    icon: str | None
    permissions: str


def _parse_json(response: httpx.Response, *, context: str) -> object:
    try:
        return response.json()
    except ValueError as exc:
        raise DiscordUnavailableError(
            f"Discord returned a non-JSON body for {context} (HTTP {response.status_code})"
        ) from exc


def _raise_for_status(response: httpx.Response, *, context: str) -> None:
    """Translate an HTTP status into this module's two-way split.

    401 and 403 are credential problems; 429 is deliberately grouped with the
    5xx family rather than retried here, because a retry loop inside a request
    handler turns Discord's rate limit into our own latency problem and, under
    load, into an amplifier pointed at the route Discord already throttles
    hardest.
    """
    if response.status_code in (401, 403):
        raise DiscordAuthError(f"Discord rejected the credential for {context} (HTTP {response.status_code})")
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "unknown")
        raise DiscordUnavailableError(
            f"Discord rate-limited {context} (retry after {retry_after}s)"
        )
    if response.status_code >= 400:
        raise DiscordUnavailableError(
            f"Discord returned HTTP {response.status_code} for {context}"
        )


def _coerce_expires_in(raw: object) -> int:
    """Turn Discord's expires_in into a positive number of seconds.

    Anything non-numeric, negative or absurd falls back to a short lifetime
    rather than being trusted. The dangerous direction is a too-LONG life: it
    would park an already-dead token in a session and surface as an
    unexplained 401 much later, so every unreadable value is treated as short.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return FALLBACK_TOKEN_LIFETIME_SECONDS
    seconds = int(raw)
    if seconds <= 0:
        return FALLBACK_TOKEN_LIFETIME_SECONDS
    return min(seconds, 30 * 24 * 3600)


def _parse_token_payload(payload: object, *, now: datetime, require_scopes: bool) -> DiscordTokens:
    if not isinstance(payload, dict):
        raise DiscordUnavailableError("Discord's token response was not a JSON object")

    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise DiscordUnavailableError("Discord's token response had no usable access_token")

    refresh_token = payload.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        refresh_token = None

    if require_scopes:
        granted = payload.get("scope")
        granted_scopes = set(granted.split()) if isinstance(granted, str) else set()
        missing = REQUIRED_SCOPES - granted_scopes
        if missing:
            # Reachable by a user who hand-edits the authorize URL to drop a
            # scope. Caught here, at the exchange, rather than as a confusing
            # 403 on the first guild listing minutes later.
            raise DiscordAuthError(
                f"Discord granted scopes {sorted(granted_scopes)}, missing {sorted(missing)}"
            )

    return DiscordTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=now + timedelta(seconds=_coerce_expires_in(payload.get("expires_in"))),
    )


def _parse_user_payload(payload: object) -> DiscordUser:
    if not isinstance(payload, dict):
        raise DiscordUnavailableError("Discord's /users/@me response was not a JSON object")

    user_id = parse_snowflake(payload.get("id"))
    if user_id is None:
        raise DiscordUnavailableError("Discord's /users/@me response had no usable id")

    raw_username = payload.get("username")
    username = sanitize_guild_name(raw_username, fallback=f"user-{user_id}")
    raw_global_name = payload.get("global_name")
    global_name = (
        sanitize_guild_name(raw_global_name, fallback="") if isinstance(raw_global_name, str) else None
    )

    return DiscordUser(
        id=user_id,
        username=username,
        global_name=global_name or None,
        avatar=sanitize_icon_hash(payload.get("avatar")),
    )


def _parse_guild_page(payload: object, *, context: str) -> list[PartialGuild]:
    if not isinstance(payload, list):
        raise DiscordUnavailableError(f"Discord's {context} response was not a JSON array")

    guilds: list[PartialGuild] = []
    for entry in payload:
        if not isinstance(entry, dict):
            logger.warning("Skipping a non-object entry in Discord's %s response", context)
            continue
        guild_id = parse_snowflake(entry.get("id"))
        if guild_id is None:
            logger.warning("Skipping a %s entry with an unusable id", context)
            continue
        raw_permissions = entry.get("permissions")
        # Normalised to a string here so PartialGuild has one type regardless
        # of which shape the API used; the *meaning* is still decided in
        # aura_web.permissions, which accepts both anyway.
        permissions = raw_permissions if isinstance(raw_permissions, str) else str(raw_permissions)
        guilds.append(
            PartialGuild(
                id=guild_id,
                name=sanitize_guild_name(entry.get("name"), fallback=UNKNOWN_GUILD_NAME),
                icon=sanitize_icon_hash(entry.get("icon")),
                permissions=permissions,
            )
        )
    return guilds


class DiscordClient:
    """Thin, validating wrapper over the handful of Discord endpoints 4b needs.

    Takes an httpx.AsyncClient rather than building one, so the application
    owns exactly one connection pool for its lifetime and tests can drive a
    transport without patching module globals.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        api_base: str,
        client_id: str,
        client_secret: str,
        bot_token: str,
    ) -> None:
        self._http = http
        self._api_base = api_base.rstrip("/")
        self._client_id = client_id
        self._client_secret = client_secret
        self._bot_token = bot_token

    async def exchange_code(self, code: str, redirect_uri: str) -> DiscordTokens:
        """Trade an authorization code for a token pair.

        Client credentials go in HTTP Basic auth rather than the form body.
        Both are documented and accepted; Basic keeps the secret out of any
        request-body logging that httpx, a proxy, or a future debugging patch
        might do, and the body is the thing most likely to get logged.
        """
        payload = await self._post_token_endpoint(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            },
            context="the authorization-code exchange",
        )
        return _parse_token_payload(payload, now=utc_now(), require_scopes=True)

    async def refresh_tokens(self, refresh_token: str) -> DiscordTokens:
        """Exchange a refresh token for a fresh pair.

        Scopes are not re-verified here: the exchange already proved them, and
        Discord's refresh response is documented to carry the same set. A
        missing-scope failure at this point would log the user out for a field
        that cannot have changed.
        """
        payload = await self._post_token_endpoint(
            {"grant_type": "refresh_token", "refresh_token": refresh_token},
            context="the refresh-token exchange",
        )
        return _parse_token_payload(payload, now=utc_now(), require_scopes=False)

    async def revoke_token(self, token: str) -> None:
        """Best-effort revocation of a token pair at logout.

        Failures are logged and swallowed on purpose. Logout's contract to the
        user is "this browser is no longer logged in", which the session
        deletion has already satisfied by the time this runs; letting
        Discord's availability decide whether logout succeeds would make the
        one action a worried user takes the one most likely to fail.
        """
        try:
            response = await self._http.post(
                f"{self._api_base}/oauth2/token/revoke",
                data={"token": token, "token_type_hint": "access_token"},
                auth=(self._client_id, self._client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if response.status_code >= 400:
                logger.warning(
                    "Discord refused a token revocation at logout (HTTP %d)", response.status_code
                )
        except httpx.HTTPError as exc:
            logger.warning("Could not reach Discord to revoke a token at logout: %s", exc)

    async def fetch_current_user(self, access_token: str) -> DiscordUser:
        """Read the identity behind a user access token (the ``identify`` scope)."""
        payload = await self._get(
            "/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
            context="/users/@me",
        )
        return _parse_user_payload(payload)

    async def fetch_user_guilds(self, access_token: str) -> list[PartialGuild]:
        """List the guilds a user belongs to, with their permission bitmask in each."""
        return await self._paginate_guilds(
            headers={"Authorization": f"Bearer {access_token}"},
            context="/users/@me/guilds (user)",
        )

    async def fetch_bot_guild_ids(self) -> frozenset[str]:
        """List the guild IDs Aura itself is a member of.

        This is the question CLAUDE.md's knowledge model cannot answer and
        Discord can. Aura's database has no membership table -- every guild_id
        in it is a side effect of activity (a fact extracted, a channel
        configured), so a freshly invited guild has no rows and a guild Aura
        was removed from keeps its rows forever. Deriving membership from that
        would be wrong in both directions; asking Discord is right in both.
        The trade-off it accepts -- this container holds the bot token -- is
        documented in web/README.md.
        """
        guilds = await self._paginate_guilds(
            headers={"Authorization": f"Bot {self._bot_token}"},
            context="/users/@me/guilds (bot)",
        )
        return frozenset(guild.id for guild in guilds)

    async def _post_token_endpoint(self, data: dict[str, str], *, context: str) -> object:
        try:
            response = await self._http.post(
                f"{self._api_base}/oauth2/token",
                data=data,
                auth=(self._client_id, self._client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except httpx.HTTPError as exc:
            raise DiscordUnavailableError(f"Could not reach Discord for {context}: {exc}") from exc

        if response.status_code == 400:
            # The token endpoint answers a reused, expired or forged code with
            # 400 invalid_grant rather than 401, so this status has to join the
            # auth family explicitly or a replayed callback would read as an
            # outage and get retried.
            raise DiscordAuthError(f"Discord rejected {context} (HTTP 400)")
        _raise_for_status(response, context=context)
        return _parse_json(response, context=context)

    async def _get(self, path: str, *, headers: dict[str, str], context: str) -> object:
        try:
            response = await self._http.get(f"{self._api_base}{path}", headers=headers)
        except httpx.HTTPError as exc:
            raise DiscordUnavailableError(f"Could not reach Discord for {context}: {exc}") from exc
        _raise_for_status(response, context=context)
        return _parse_json(response, context=context)

    async def _paginate_guilds(self, *, headers: dict[str, str], context: str) -> list[PartialGuild]:
        """Walk Discord's ``after``-cursor pagination to the end of a guild list.

        Three independent stop conditions, because relying on any one alone
        makes a broken or hostile upstream into an unbounded loop inside a
        request handler: a short page, an empty page, and a page-count
        ceiling. The cursor advances to the numerically largest ID seen rather
        than the last one in the array -- Discord orders these ascending, but
        an out-of-order page would otherwise send the cursor backwards and
        re-request the same page forever.
        """
        collected: dict[str, PartialGuild] = {}
        after: str | None = None

        for _ in range(MAX_GUILD_PAGES):
            query = f"?limit={DISCORD_GUILD_PAGE_SIZE}"
            if after is not None:
                query += f"&after={after}"
            page = _parse_guild_page(
                await self._get(f"/users/@me/guilds{query}", headers=headers, context=context),
                context=context,
            )
            if not page:
                return list(collected.values())

            highest_seen = after
            for guild in page:
                # Deduplicated by ID: a page boundary that shifts while we
                # read it can hand back the same guild twice, and a duplicate
                # would otherwise reach the browser as two identical cards.
                collected[guild.id] = guild
                if highest_seen is None or int(guild.id) > int(highest_seen):
                    highest_seen = guild.id

            if len(page) < DISCORD_GUILD_PAGE_SIZE:
                return list(collected.values())
            if highest_seen == after:
                logger.warning("Discord's %s pagination stopped advancing; stopping early", context)
                return list(collected.values())
            after = highest_seen

        logger.warning(
            "Discord's %s pagination exceeded %d pages; returning what was collected",
            context,
            MAX_GUILD_PAGES,
        )
        return list(collected.values())
