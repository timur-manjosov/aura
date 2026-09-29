"""Reading request bodies with a hard size ceiling, and parsing JSON without ambiguity.

Two routes read bodies in this service and both face the internet: Stripe's
webhook (unauthenticated until its signature checks out) and the billing
actions (authenticated, but sent by a browser). Neither may let a caller decide
how much memory a request costs, and neither may accept JSON two parsers could
read differently.
"""

from __future__ import annotations

import json
from typing import Any

from starlette.requests import Request


class BodyTooLargeError(Exception):
    """The request body exceeds the route's ceiling."""


async def read_bounded_body(request: Request, limit: int) -> bytes:
    """Read the raw body, refusing it the moment it exceeds `limit` bytes.

    Parameters
    ----------
    request
        The incoming request.
    limit
        Maximum bytes to accept.

    Returns
    -------
    bytes
        The raw body.

    Raises
    ------
    BodyTooLarge
        The moment the stream exceeds `limit`, so an oversized body is never
        fully buffered.

    Notes
    -----
    A declared Content-Length over the limit is refused before a byte is read;
    a body that lies about its length, or streams without one, is refused as
    soon as it crosses the limit rather than after being buffered in full.
    """
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise BodyTooLargeError
    received = bytearray()
    async for chunk in request.stream():
        received.extend(chunk)
        if len(received) > limit:
            raise BodyTooLargeError
    return bytes(received)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # {"guild_id": "a", "guild_id": "b"} is last-one-wins in Python and
            # first-one-wins in other parsers; a proxy validating one reading
            # and this service acting on the other is the textbook smuggling
            # shape. Neither reading is honoured.
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name}")


def parse_json_object(raw: bytes) -> dict[str, Any] | None:
    """Parse a JSON object strictly: UTF-8, no duplicate keys, no NaN/Infinity. None if not.

    Parameters
    ----------
    raw
        The raw request body.

    Returns
    -------
    dict[str, Any] or None
        The parsed object, or None when it is not valid UTF-8, not JSON, not a
        JSON *object*, carries duplicate keys, or contains NaN/Infinity.
    """
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None
