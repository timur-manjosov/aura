"""Trigger 2's policy: when an eligible message becomes an actual public post.

This is the first place in Aura's whole existence where money is spent and a
message is posted unprompted. Everything upstream (question_detector, gate,
proactive_state) is free and silent; the gate's ELIGIBLE verdict is the line
this module sits behind.

The core policy distinction, kept explicit here rather than left implicit in
code: **Trigger 1 (/aura-ask) answers an explicit request, so a wrong or
unconfident answer is a bounded risk the asker opted into. Trigger 2 is
unprompted, so an unconfident answer nobody asked for damages trust in the bot
as a whole.** Trigger 2 must therefore be willing to stay silent far more often
than Trigger 1 -- but when it is genuinely confident it should post, because
that is exactly what earns Aura its reputation as useful rather than annoying.

Both triggers call the same synthesis function (aura.synthesis.synthesize_answer)
-- one mechanism, not two. What lives here is only the *policy* applied to its
result: the hard code-gate below, and the distinguishable, transparent framing
of the post.

**The hard code-gate.** A message is posted only if every one of these agrees,
and any single "no" is silence:
  1. The channel is proactive-enabled (checked as the pipeline's first gate,
     and RE-checked here right before posting, in case a moderator toggled it
     off mid-flight).
  2. An LLM is configured for this trigger.
  3. Synthesis returned a well-formed result.
  4. The result's own answers_question self-assessment is true, AND it actually
     cited at least one fact.
The numeric Stage 1/2 gates (question-likeness, similarity) and the budget gate
were already satisfied upstream, computed from the message
geometry alone and entirely independent of anything the LLM concludes here.
That independence is the defense against prompt injection: a message crafted to
flip answers_question to true can, at most, affect that one field -- it can
never make a message that failed the numeric gates reach this code at all,
because this code only runs behind the gate's ELIGIBLE verdict.
"""
from __future__ import annotations

import logging

import aiosqlite
import discord
from fastembed import TextEmbedding
from pydantic import BaseModel

from aura.config import ModelComponent, Settings
from aura.db.models import Fact
from aura.db.proactive_channel_config import is_channel_enabled
from aura.embeddings import SYNTHESIS_FACT_LIMIT, find_similar_facts
from aura.i18n import DEFAULT_LOCALE, t
from aura.synthesis import SynthesisResult, synthesize_answer

logger = logging.getLogger(__name__)

# Discord's own embed-description hard cap, same limit /aura-ask truncates to.
_ANSWER_DISPLAY_LIMIT = 4096

# A deliberately distinct colour so an unsolicited proactive answer never looks
# like a plain /aura-ask reply (which carries no colour). This is a transparency
# measure, not decoration -- see the localized framing label and footer below.
_PROACTIVE_EMBED_COLOR = discord.Color.blurple()


class ProactiveResponseOutcome(BaseModel):
    """What synthesis decided for one eligible message, for the debug trail.

    answers_question mirrors the LLM's self-assessment, or is None when
    synthesis produced no usable result at all (not configured, call failed,
    knowledge model changed out from under it). posted is always definite:
    True only if a message was genuinely sent.
    """

    answers_question: bool | None
    posted: bool


def _proactive_locale(guild: discord.Guild) -> str:
    """The locale a proactive post is written and framed in.

    A proactive answer has no asking user whose interaction.locale we could
    read, so it uses the guild's own preferred locale -- the best server-scoped
    signal available. Per-message language detection (answering each member in
    the language they wrote in) is deliberately out of scope for this phase and
    left for Phase 2b. Falls back to Aura's default locale if a guild somehow
    reports none, matching t()'s own mandatory-fallback rule.
    """
    preferred = getattr(guild, "preferred_locale", None)
    return str(preferred) if preferred else DEFAULT_LOCALE


def _build_proactive_embed(
    result: SynthesisResult, facts: list[Fact], locale: str
) -> discord.Embed:
    """Build the visibly-distinct embed for an unsolicited proactive answer.

    Distinguishable from an /aura-ask reply three ways, all localized: a
    coloured embed (ask replies have none), an author line framing it as Aura
    volunteering information, and a footer stating it was automatic and how to
    turn it off. Server members must be able to tell at a glance that nobody
    asked Aura this -- it spoke up on its own.
    """
    answer = result.answer
    if len(answer) > _ANSWER_DISPLAY_LIMIT:
        answer = answer[: _ANSWER_DISPLAY_LIMIT - 1] + "…"

    embed = discord.Embed(description=answer, color=_PROACTIVE_EMBED_COLOR)
    embed.set_author(name=t("proactive_reply_label", locale))

    cited_facts = [fact for fact in facts if fact.id in result.used_fact_ids]
    if cited_facts:
        links = "\n".join(
            f"https://discord.com/channels/{fact.guild_id}/{fact.channel_id}/{fact.message_id}"
            for fact in cited_facts
        )
        embed.add_field(name=t("ask_sources_label", locale), value=links, inline=False)

    embed.set_footer(text=t("proactive_reply_footer", locale))
    return embed


async def respond_with_synthesis(
    message: discord.Message,
    *,
    db: aiosqlite.Connection,
    model: TextEmbedding,
    settings: Settings,
) -> ProactiveResponseOutcome:
    """Synthesize an answer for an already-eligible message and post it, if confident.

    Called only behind the gate's ELIGIBLE verdict -- by which point a budget
    slot has already been spent for this message, whatever happens next. Returns
    a ProactiveResponseOutcome the caller records onto the message's trail;
    never raises for an expected failure (missing config, empty facts, a failed
    LLM call, a rejected post), so the caller's single record path always runs.

    The order of checks is the hard code-gate documented at module level:
    configured -> facts still present -> synthesis succeeded -> model confident
    and cited -> channel still enabled -> post.
    """
    guild = message.guild
    assert guild is not None  # only reached for a guild message (see should_classify)
    channel = message.channel
    locale = _proactive_locale(guild)

    if not settings.is_llm_configured(ModelComponent.PROACTIVE):
        # Proactive relief is not operational here: no funded LLM. The eligible
        # message already spent a budget slot (bounded by the daily cap, no
        # money, no post), and it shows in the debug trail as ELIGIBLE with no
        # synthesis outcome -- which is exactly the signal a moderator needs
        # that a channel was enabled but no model is configured.
        return ProactiveResponseOutcome(answers_question=None, posted=False)

    # PROACTIVE_SIMILARITY_THRESHOLD, not the direct-query SIMILARITY_THRESHOLD.
    # Through Phase 2b-3 this filtered on the latter (0.40) while the gate that
    # authorized the call used the former (0.30), and the mismatch was a live
    # defect rather than a stylistic one: any message whose best fact scored
    # between the two bars was granted an escalation slot upstream and then
    # found nothing to answer from here -- a spent slot, permanent silence, and
    # a trail row indistinguishable from a failed LLM call. Measured at 45 of
    # 580 cases on the Phase 2b-2 corpus, so roughly one eligible message in
    # eight. Trigger 2 now uses one bar end to end, which is what makes the
    # gate's verdict mean what it says.
    #
    # Bounded explicitly rather than by find_similar_facts' default: for an
    # unprompted call the number of facts entering the prompt is a cost ceiling
    # (see SYNTHESIS_FACT_LIMIT), and a ceiling should be stated where it is
    # relied upon.
    results = await find_similar_facts(
        db, model, guild_id=guild.id, query=message.content, top_k=SYNTHESIS_FACT_LIMIT
    )
    relevant_facts = [
        fact for fact, score in results if score >= settings.proactive_similarity_threshold
    ]
    if not relevant_facts:
        # The knowledge model moved between the gate and here (a fact was
        # superseded, say). No basis to answer; stay silent.
        return ProactiveResponseOutcome(answers_question=None, posted=False)

    # --- PROACTIVE_MODEL selection (CLAUDE.md: reason about the task, don't
    #     restate the criteria; document the evidence) -------------------------
    #
    # This call's real shape is a structured JSON classification
    # (answers_question) plus a short cited synthesis, over a moderate
    # multilingual load (Aura's nine locales) -- NOT heavy code-reasoning. Per
    # this phase's core policy, an unprompted false-positive costs more
    # reputationally than an explicit-request one, so the trait that matters
    # most is a trustworthy, well-CALIBRATED answers_question -- structured-
    # output reliability and calibration weigh more here than raw speed, and
    # Trigger 2 is on no user's critical path, so latency barely matters.
    #
    # RESOLVED by measurement. The bake-off Phase 2a-3 could not afford is now
    # run: 12 hand-picked cases across 7 of the 9 locales (en-US, de, ja, tr,
    # pt-BR, es-ES, ko), driving this exact call path. Raw per-case results are
    # in reports/model-bakeoff.txt; scripts/model_bakeoff.py re-runs it.
    #
    # Live OpenRouter pricing was re-checked first, and one of the phase's
    # assumptions had already gone stale: gpt-5.4-mini is $0.75/$4.50 per Mtok,
    # not the ~$0.20/$1.25 recorded then, which removes most of the cost gap
    # that made Haiku a "fallback only" option.
    #   * google/gemini-3.1-flash-lite-preview  $0.25/$1.50   10/12
    #   * openai/gpt-5.4-mini                   $0.75/$4.50   10/12
    #   * anthropic/claude-haiku-4.5            $1.00/$5.00   12/12
    #
    # All three handle the JSON schema and all nine-locale output fine; the two
    # cheaper ones lose on CALIBRATION, which is the trait this trigger actually
    # depends on. Both failed the same two cases -- a fact that only partially
    # answers the question, and two contradictory active facts -- by answering
    # confidently where the correct move is to decline. Repeating just those two
    # cases 3x each: gemini 0/6, gpt-5.4-mini 4/6 (it flip-flops run to run),
    # haiku 6/6. A model that is right about easy cases and optimistic about
    # ambiguous ones is precisely wrong for a trigger nobody asked to hear from.
    #
    # Cost does not overturn that, because PROACTIVE_DAILY_CAP already bounds
    # it: measured at ~420 in / ~55 out tokens per call, the daily cap of 20
    # puts the three at $0.10, $0.28 and $0.42 per guild per month. The whole
    # spread is 32 cents; one avoided wrong public post is worth more.
    # Latency (haiku median 1.50s vs gemini 0.65s) is irrelevant here -- no user
    # is waiting on Trigger 2 at all.
    #
    # This is the same model as SYNTHESIS_MODEL, which is a legitimate outcome
    # rather than a decision left unmade: /aura-ask was evaluated on its own
    # merits (see .env.example) and the same model won there too, for different
    # reasons. The two values stay separate so they CAN diverge later.
    #
    # One bug this bake-off surfaced is already fixed rather than logged: Haiku
    # returned its JSON inside a ```json fence on 12 of 12 calls, which made
    # every Anthropic model score 0/12 until aura.synthesis learned to unwrap it.
    # That was a fault in Aura, not in the model -- see _parse_json_response.
    proactive_model = settings.resolve_model(ModelComponent.PROACTIVE)
    assert proactive_model is not None  # guaranteed by is_llm_configured() above
    result = await synthesize_answer(
        relevant_facts, message.content, locale, model=proactive_model
    )

    if result is None:
        return ProactiveResponseOutcome(answers_question=None, posted=False)

    # The hard code-gate. answers_question is an additional requirement on top
    # of the numeric gates, never a substitute -- and a claim to answer with no
    # cited fact is treated as not-confident, since a confident answer that
    # draws from nothing is a contradiction.
    if not result.answers_question or not result.used_fact_ids:
        return ProactiveResponseOutcome(answers_question=result.answers_question, posted=False)

    # Freshest-setting check: re-read the channel switch right before posting,
    # not the value the pipeline saw seconds ago, so a moderator who toggles the
    # channel off mid-synthesis is obeyed. The slot stays spent -- that is the
    # documented direction for a budget whose job is bounding cost.
    if not await is_channel_enabled(db, channel_id=channel.id):
        return ProactiveResponseOutcome(answers_question=result.answers_question, posted=False)

    embed = _build_proactive_embed(result, relevant_facts, locale)
    try:
        await channel.send(embed=embed)
    except Exception:
        # A post can fail for reasons entirely outside Aura's control: the
        # channel was deleted, send permissions were revoked, Discord returned
        # an error. Fail closed -- no post recorded -- and let the caller record
        # the outcome. CancelledError is a BaseException and still propagates.
        logger.exception(
            "Proactive post failed in channel %s", getattr(channel, "id", "<unknown>")
        )
        return ProactiveResponseOutcome(answers_question=result.answers_question, posted=False)

    return ProactiveResponseOutcome(answers_question=result.answers_question, posted=True)
