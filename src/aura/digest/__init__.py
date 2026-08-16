"""The periodic digest (CLAUDE.md's FOURTH trigger): what has changed in this
server's knowledge model since the last time Aura said so.

One mechanism, four triggers -- this is the fourth, and like the other three it
reads nothing but the four-part knowledge model: a fact's distilled sentence,
its timestamp, its status and successor chain. Nothing new is stored about a
fact to make digests possible, and nothing here writes to the knowledge model at
all.

**No LLM call anywhere in this package.** Every word a digest contains was
already written and structured before the digest existed, so there is nothing
here for a model to reason about; see aura.digest.builder for the full argument.
The consequence worth naming: this is the first thing Aura posts publicly with
no generated text in it, which is why it needs no grounding check -- there is
nothing that could have been invented.

The three pieces, deliberately separated so the interesting one needs no Discord
connection to verify:

  * builder   -- what changed, as plain data. Pure logic over database reads.
  * formatter -- that data as one localized embed, bounded in every direction.
  * scheduler -- when it happens, and the bookkeeping that makes "when" survive
                 a restart. Its durable half lives in aura.db.digest_state.
"""
from aura.digest.builder import DigestChange, DigestContent, build_digest
from aura.digest.formatter import build_digest_embed, digest_locale
from aura.digest.gateway import ClientDigestGateway, DigestGateway
from aura.digest.intervals import DigestInterval, describe_interval
from aura.digest.scheduler import run_digest_scheduler, send_due_digests, window_start

__all__ = [
    "ClientDigestGateway",
    "DigestChange",
    "DigestContent",
    "DigestGateway",
    "DigestInterval",
    "build_digest",
    "build_digest_embed",
    "describe_interval",
    "digest_locale",
    "run_digest_scheduler",
    "send_due_digests",
    "window_start",
]
