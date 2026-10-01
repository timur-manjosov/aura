"""Run the web backend with uvicorn: ``python -m aura_web``.

Mirrors ``python -m aura.main`` in the bot container, so both services in this
repository start the same way.
"""

from __future__ import annotations

import os
import sys
from contextlib import suppress
from typing import Final

import uvicorn
from fastapi import FastAPI

from aura_web.app import build_app

# uvicorn.run's own exit status for a server that never finished starting (a
# failed lifespan startup, a port already taken), kept so the container's exit
# code means what it meant before this module built the server itself.
STARTUP_FAILURE_EXIT_CODE: Final[int] = 3


def uvicorn_config(app: FastAPI, *, host: str, port: int) -> uvicorn.Config:
    """Return the server configuration the backend is run with.

    Parameters
    ----------
    app
        The application to serve.
    host, port
        Where to listen.

    Returns
    -------
    uvicorn.Config
        uvicorn's own logging configuration left alone, its access log on, and
        its proxy-header handling off.

    Notes
    -----
    The access log is left on: it is the record that shows a 400 on the
    callback route, which is what a rejected state looks like from outside. It
    names the client aura_web.client_address resolved, because that middleware
    rewrites the address before the log line reads it.

    ``proxy_headers`` is off so X-Forwarded-For is read in exactly one place,
    with exactly one rule (aura_web.client_address). uvicorn's own handling
    defaults to on and trusts 127.0.0.1; with both, the address a request is
    limited and logged under would depend on which rule saw the header first.
    """
    return uvicorn.Config(app, host=host, port=port, log_config=None, proxy_headers=False)


def main() -> None:
    """Serve the application on the configured host and port."""
    config = uvicorn_config(
        build_app(),
        host=os.environ.get("AURA_WEB_HOST", "0.0.0.0"),
        port=int(os.environ.get("AURA_WEB_PORT", "8080")),
    )
    server = uvicorn.Server(config)
    with suppress(KeyboardInterrupt):
        server.run()
    if not server.started:
        sys.exit(STARTUP_FAILURE_EXIT_CODE)


if __name__ == "__main__":
    main()
