"""Onboarding (CLAUDE.md's THIRD trigger): a new member receives the current
knowledge state as a summary, instead of a static welcome message.

One mechanism, four triggers -- this is the third, and like the other three it
reads nothing but the four-part knowledge model: a fact's distilled sentence,
its timestamp, its category (read back through pending_facts, exactly as the
digest's milestone section does) and, structurally, its status -- only ACTIVE
facts are ever eligible. Nothing new is stored about a fact to make onboarding
possible, and nothing here writes to the knowledge model at all.

**No LLM call in the base case.** Everything an onboarding message says was
already written and structured before this code runs; see aura.onboarding.
builder for the full argument, which is the digest's argument (aura.digest.
builder) applied to a full snapshot instead of a window, plus the one
exception this sub-phase's brief asked to be checked rather than assumed away.

**Posts to a channel, never a DM.** An unsolicited private message to someone
who just joined a server they do not yet trust is a stronger interruption than
a channel post, which is consistent with the "deliberately conservative"
posture Trigger 2, the grounding check, and the digest all already take.

The four pieces, deliberately separated so the interesting ones need no
Discord connection to verify:

  * builder   -- what a new member should see, as plain data. Pure logic over
                 database reads.
  * formatter -- that data as one localized embed, sharing its
                 security-critical rendering (link-hijack escaping, field
                 truncation) with the digest through aura.rendering.
  * gateway   -- turning a stored channel ID into somewhere to post.
  * listener  -- the on_member_join orchestration: config check, content,
                 channel, the atomic claim against duplicate/mass joins
                 (aura.db.onboarding_state), and the send.
"""
from aura.onboarding.builder import OnboardingContent, build_onboarding_content
from aura.onboarding.formatter import build_onboarding_embed, onboarding_locale
from aura.onboarding.gateway import ClientOnboardingGateway, OnboardingGateway
from aura.onboarding.listener import handle_member_join

__all__ = [
    "ClientOnboardingGateway",
    "OnboardingContent",
    "OnboardingGateway",
    "build_onboarding_content",
    "build_onboarding_embed",
    "handle_member_join",
    "onboarding_locale",
]
