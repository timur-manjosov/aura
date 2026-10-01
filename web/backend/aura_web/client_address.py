"""Which address a request really came from, believed only from the frontend.

THE CHAIN, as measured against the deployed versions (Caddy 2.11, Next.js 16):

    browser --TLS--> host Caddy --> frontend container (Next.js) --> this backend

  * Caddy, configured with no ``trusted_proxies``, discards any
    X-Forwarded-For the client sent -- one header or several -- and writes its
    own: exactly one entry, the client's address. It passes ``Forwarded`` and
    ``X-Real-IP`` through untouched, so neither may ever be read here.
  * Next.js's rewrite proxy forwards X-Forwarded-For unchanged and appends
    nothing; a request that reached it without one (only possible from the
    host itself, on the loopback port) arrives here without one.
  * This service's connection peer is therefore always the frontend
    container -- the one party whose X-Forwarded-For is believed, because its
    address is pinned in web/docker-compose.yml and nothing else on that
    network can hold it.

THE RULE: X-Forwarded-For is read only when the connection's peer is a
configured trusted proxy, and then from the right: entries that are themselves
trusted proxies are skipped, and the first entry that is not is the client.
The rightmost untrusted entry is the last one a party this service trusts
actually wrote; everything to its left was written by someone it does not, and
reading the leftmost entry instead -- the tempting choice -- is exactly what
lets a client pick its own rate-limit identity. Anything unparseable stops the
walk and falls back to the peer: a malformed chain is not evidence of who sent
the request, so it earns the frontend's own shared identity, never one the
sender chose.

The resolved address replaces ``scope["client"]`` before anything else runs,
so uvicorn's access log, this service's log lines and the rate limiter all
see the same address. Imports nothing from the rest of this package.
"""

from __future__ import annotations

from collections.abc import Iterable
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Final

from starlette.types import ASGIApp, Receive, Scope, Send

# The only forwarding header read. `Forwarded` (RFC 7239) and `X-Real-IP` are
# passed through by Caddy exactly as the client sent them; they are named here
# so a reader looking for them finds the decision rather than an omission.
FORWARDED_FOR_HEADER: Final[bytes] = b"x-forwarded-for"
IGNORED_FORWARDING_HEADERS: Final[frozenset[bytes]] = frozenset({b"forwarded", b"x-real-ip"})

# A chain longer than this is not one any deployment of this service produces
# (it has one hop), so the walk stops there and falls back to the peer rather
# than parsing an attacker-sized header entry by entry.
MAX_FORWARDED_HOPS: Final[int] = 16


def parse_address(value: str) -> IPv4Address | IPv6Address | None:
    """Parse one address, folding an IPv4-mapped IPv6 form to plain IPv4.

    Parameters
    ----------
    value
        A candidate address, already stripped of surrounding whitespace.

    Returns
    -------
    IPv4Address | IPv6Address | None
        The address, or None for anything that is not exactly one IP address
        (a hostname, a port suffix, brackets, a zone index, an empty string).
    """
    try:
        address = ip_address(value)
    except ValueError:
        return None
    if isinstance(address, IPv6Address):
        if address.scope_id is not None:
            return None
        if address.ipv4_mapped is not None:
            return address.ipv4_mapped
    return address


def resolve_client_address(
    peer: str | None,
    forwarded_for: Iterable[str],
    trusted_proxies: frozenset[IPv4Address | IPv6Address],
) -> str | None:
    """Return the address a request is attributed to.

    Parameters
    ----------
    peer
        The connection's peer address as the server reports it, or None when
        the server reports none.
    forwarded_for
        Every X-Forwarded-For header value on the request, in arrival order.
    trusted_proxies
        The proxy addresses whose X-Forwarded-For is believed.

    Returns
    -------
    str | None
        The rightmost X-Forwarded-For entry that is not itself a trusted proxy,
        when the peer is a trusted proxy and every entry walked parses; the
        peer unchanged in every other case, including a peer that is not an IP
        address at all. None only when `peer` is None.
    """
    if peer is None:
        return None
    peer_address = parse_address(peer.strip())
    if peer_address is None or peer_address not in trusted_proxies:
        return peer

    # Several header lines are one list (RFC 9110 §5.3), in order.
    hops = [hop.strip() for value in forwarded_for for hop in value.split(",")]
    if len(hops) > MAX_FORWARDED_HOPS:
        return peer
    for hop in reversed(hops):
        hop_address = parse_address(hop)
        if hop_address is None:
            return peer
        if hop_address in trusted_proxies:
            continue
        return str(hop_address)
    return peer


def forwarded_for_values(scope: Scope) -> list[str]:
    """Return every X-Forwarded-For header value of an ASGI request, in order.

    Parameters
    ----------
    scope
        An ASGI HTTP scope.

    Returns
    -------
    list[str]
        The raw values, latin-1 decoded as ASGI headers are specified to be.
    """
    return [
        value.decode("latin-1")
        for name, value in scope.get("headers", [])
        if name.lower() == FORWARDED_FOR_HEADER
    ]


class ClientAddressMiddleware:
    """Replace ``scope["client"]`` with the resolved client address, first thing.

    Parameters
    ----------
    app
        The ASGI application to wrap.
    trusted_proxies
        The proxy addresses whose X-Forwarded-For is believed; empty trusts
        none, which leaves every scope unchanged.

    Notes
    -----
    A pure ASGI middleware, not Starlette's BaseHTTPMiddleware, so it neither
    touches nor buffers the body: it must run before the rate limiter, which
    in turn must decide before any body is read.

    uvicorn's own proxy-header handling is switched off (aura_web.__main__) so
    this is the only place that ever reads X-Forwarded-For. With both on, the
    two rules would run one after the other and the result would depend on
    which saw the header first.
    """

    def __init__(
        self, app: ASGIApp, *, trusted_proxies: frozenset[IPv4Address | IPv6Address]
    ) -> None:
        self.app = app
        self.trusted_proxies = trusted_proxies

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Resolve the client address of an HTTP request, then hand it on.

        Parameters
        ----------
        scope, receive, send
            The ASGI connection.
        """
        if scope["type"] == "http":
            client = scope.get("client")
            if client is not None:
                peer_host = client[0]
                resolved = resolve_client_address(
                    peer_host, forwarded_for_values(scope), self.trusted_proxies
                )
                if resolved is not None and resolved != peer_host:
                    # The client's own port is not known past a proxy; 0 says
                    # so rather than pairing its address with the proxy's port.
                    scope["client"] = (resolved, 0)
        await self.app(scope, receive, send)
