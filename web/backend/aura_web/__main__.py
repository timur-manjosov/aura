"""Run the web backend with uvicorn: ``python -m aura_web``.

Mirrors ``python -m aura.main`` in the bot container, so both services in this
repository start the same way.
"""
from __future__ import annotations

import os

import uvicorn

from aura_web.app import build_app


def main() -> None:
    """Serve the application on the configured host and port."""
    uvicorn.run(
        build_app(),
        host=os.environ.get("AURA_WEB_HOST", "0.0.0.0"),
        port=int(os.environ.get("AURA_WEB_PORT", "8080")),
        # uvicorn's own access log is left on: it is the record that shows a
        # 400 on the callback route, which is what a rejected state looks like
        # from outside.
        log_config=None,
    )


if __name__ == "__main__":
    main()
