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
  5. The independent grounding check agrees the cited facts actually support the
     answer that was written (see aura.grounding). A rejection, a timeout, and
     any other failure of that check are all silence here -- unlike /aura-ask,
     which replies honestly in the same two situations, because somebody asked
     it. Nobody asked Trigger 2, so an unsolicited "I could not verify my own
     answer" would be exactly the unwanted interruption CLAUDE.md's
     "deliberately conservative" instruction exists to prevent: it carries no
     information a reader can use and costs the same attention a real answer
     would. It is logged at WARNING instead, which is where an operator can see
     it without a channel having to.
The numeric Stage 1/2 gates (question-likeness, similarity) and the budget gate
were already satisfied upstream, computed from the message
geometry alone and entirely independent of anything the LLM concludes here.
That independence is the defense against prompt injection: a message crafted to
flip answers_question to true can, at most, affect that one field -- it can
never make a message that failed the numeric gates reach this code at all,
because this code only runs behind the gate's ELIGIBLE verdict.

**The v2 answer format** (PROACTIVE_ANSWER_FORMAT, default legacy). Selected,
the same gate runs on the PROACTIVE VARIANT of the structured answer contract
(P5): the model first names what the message is (`message_kind`), and it posts
only when that is a sincere request, the answer answers the question, cites a
fact, and involves neither a same-detail conflict nor an "unclear if same" pair
(aura.answer_contract.ContractAnswer.answers_unprompted); the statement check of
aura.answer_check replaces the legacy grounding check; and the post is a
proactive answer card with at most two points. PROACTIVE_PROVIDERS and its
siblings route the answer, PROACTIVE_MAX_OUTPUT_TOKENS bounds it. Everything upstream -- the gate,
the thresholds, the budget, the facts retrieved -- is identical in both formats.

**Late answers (P5c), both formats.** PROACTIVE_REQUEST_TIMEOUT_SECONDS, when
set, is a hard deadline on the answer call (the client's own timeout is per
read on OpenRouter, measured in P5c, so it bounds nothing on its own). And the
caller (aura.proactive.listener) hands in a freshness watch: right before the
post, after the channel re-check, an answer whose conversation moved on -- a
different member wrote in the channel, or the question was edited or deleted,
since the grace period ended -- or that took longer than
`answer_deadline_seconds` after the grace period is silence and an INFO line.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Final, Protocol

import aiosqlite
import discord
from fastembed import TextEmbedding
from pydantic import BaseModel

from aura.answer_card import (
    build_answer_card,
    card_to_embed,
    card_to_layout_view,
    components_v2_available,
    label_legacy_embed,
    with_answer_labels,
)
from aura.answer_check import build_statements, verify_answer_v2
from aura.answer_contract import synthesize_contract_answer
from aura.config import AnswerFormat, CardStyle, ModelComponent, Settings
from aura.db.models import Fact
from aura.db.proactive_channel_config import is_channel_enabled
from aura.discord_context import channel_display_name, fact_channel_names
from aura.embeddings import SYNTHESIS_FACT_LIMIT, find_similar_facts
from aura.grounding import (
    PROACTIVE_GROUNDING_TIMEOUT_SECONDS,
    GroundingOutcome,
    verify_answer_grounded,
)
from aura.i18n import DEFAULT_LOCALE, t
from aura.links_service import expand_with_linked_facts
from aura.llm_request_options import openrouter_extra_body, parse_provider_list
from aura.rendering import source_link
from aura.synthesis import SynthesisResult, synthesize_answer

logger = logging.getLogger(__name__)

# Discord's own embed-description hard cap, same limit /aura-ask truncates to.
_ANSWER_DISPLAY_LIMIT = 4096

# The client timeout both answer calls pass when PROACTIVE_REQUEST_TIMEOUT_SECONDS
# is unset (aura.synthesis and aura.answer_contract, both 30; a test keeps the
# three equal). Used here only to state the posting deadline.
DEFAULT_ANSWER_TIMEOUT_SECONDS: Final = 30.0

# Added to the two calls' own limits for the posting deadline: the retrieval,
# the channel re-check and the scheduling around them. Small next to the two
# calls, large enough never to cut an answer that met both of them.
_ANSWER_DEADLINE_SLACK_SECONDS: Final = 15.0


class AnswerFreshness(Protocol):
    """Whether an answer may still be posted (aura.proactive.grace.AnswerWatch)."""

    def stale_reason(self, *, deadline_seconds: float) -> str | None:
        """Return why the answer is stale, or None when it may be posted."""
        ...


def answer_deadline_seconds(settings: Settings) -> float:
    """Return how long after the grace period an unprompted answer may still be posted.

    Parameters
    ----------
    settings
        Loaded configuration: PROACTIVE_REQUEST_TIMEOUT_SECONDS.

    Returns
    -------
    float
        The answer call's limit (PROACTIVE_REQUEST_TIMEOUT_SECONDS, or the 30
        seconds of before when it is unset) plus the check's deadline
        (PROACTIVE_GROUNDING_TIMEOUT_SECONDS) plus a 15-second allowance: 75 s
        by default, 105 s at a 60-second timeout.

    Notes
    -----
    An answer that met both calls' limits is never cut by this; it exists for
    the one that did not -- a client timeout the provider kept alive, a stalled
    event loop or database -- so that no setting and no library can make a
    proactive post arrive later than this.
    """
    timeout = settings.proactive_request_timeout_seconds
    call_limit = timeout if timeout is not None else DEFAULT_ANSWER_TIMEOUT_SECONDS
    return call_limit + PROACTIVE_GROUNDING_TIMEOUT_SECONDS + _ANSWER_DEADLINE_SLACK_SECONDS


async def _within_deadline[T](call: Awaitable[T], settings: Settings) -> T:
    """Await the proactive answer call, under its hard deadline when one is configured.

    Raises
    ------
    TimeoutError
        When PROACTIVE_REQUEST_TIMEOUT_SECONDS is set and the call ran past it.
    """
    timeout = settings.proactive_request_timeout_seconds
    if timeout is None:
        return await call
    return await asyncio.wait_for(call, timeout=timeout)


def _stale(freshness: AnswerFreshness | None, settings: Settings, channel: object) -> bool:
    """Report (and log) whether the answer must not be posted any more."""
    if freshness is None:
        return False
    reason = freshness.stale_reason(deadline_seconds=answer_deadline_seconds(settings))
    if reason is None:
        return False
    logger.info(
        "Proactive answer withheld in channel %s: %s",
        getattr(channel, "id", "<unknown>"),
        reason,
    )
    return True


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


def _proactive_route(settings: Settings, model: str) -> dict[str, object] | None:
    """Return PROACTIVE_MODEL's OpenRouter request options, or None when none are set.

    Parameters
    ----------
    settings
        Loaded configuration: PROACTIVE_PROVIDERS, PROACTIVE_REASONING and
        PROACTIVE_DENY_DATA_COLLECTION.
    model
        The resolved proactive model.

    Returns
    -------
    dict[str, object] or None
        The `extra_body` both answer formats send; None (nothing extra) when
        every option is unset or the model is not routed through OpenRouter.
    """
    return openrouter_extra_body(
        model,
        providers=parse_provider_list(settings.proactive_providers),
        deny_data_collection=settings.proactive_deny_data_collection,
        reasoning=settings.proactive_reasoning,
    )


def _log_deadline_passed(channel: object, settings: Settings) -> None:
    """Log that the answer call ran past PROACTIVE_REQUEST_TIMEOUT_SECONDS."""
    logger.warning(
        "Proactive answer withheld in channel %s: the answer call ran past its "
        "%.0f-second deadline (PROACTIVE_REQUEST_TIMEOUT_SECONDS)",
        getattr(channel, "id", "<unknown>"),
        settings.proactive_request_timeout_seconds,
    )


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
        links = "\n".join(source_link(fact) for fact in cited_facts)
        embed.add_field(name=t("ask_sources_label", locale), value=links, inline=False)

    embed.set_footer(text=t("proactive_reply_footer", locale))
    return embed


async def _respond_in_v2(
    message: discord.Message,
    *,
    db: aiosqlite.Connection,
    settings: Settings,
    synthesis_facts: list[Fact],
    locale: str,
    proactive_model: str,
    freshness: AnswerFreshness | None,
) -> ProactiveResponseOutcome:
    """Trigger 2 in the v2 answer format: the same hard code-gate, an answer card.

    Parameters
    ----------
    message
        The message that cleared the gate.
    db
        Open database connection, for the channel re-check.
    settings
        Loaded configuration: the checker and the card style.
    synthesis_facts
        The retrieved facts plus their links.
    locale
        The guild's locale.
    proactive_model
        The resolved proactive model. It is sent with its own PROACTIVE_*
        route, never the ANSWER_V2_* one, which describes ANSWER_V2_MODEL;
        the check is sent with its own route, since both triggers use the
        same checker.
    freshness
        The caller's watch over the conversation; None skips the freshness
        check (direct callers, tests).

    Returns
    -------
    ProactiveResponseOutcome
        What happened; never raises for an expected failure.

    Notes
    -----
    The order is the legacy one: synthesis, the model's own verdict, the check,
    the channel re-check, the freshness check, the post. A refused or failed
    check is silence and a WARNING, as in the legacy format; an answer call
    past its deadline is silence and a WARNING. Every post disables mentions.
    """
    channel = message.channel
    try:
        answer = await _within_deadline(
            synthesize_contract_answer(
                synthesis_facts,
                message.content,
                locale,
                model=proactive_model,
                settings=settings,
                extra_body=_proactive_route(settings, proactive_model),
                # P5: the proactive variant of the contract -- the model first
                # says what the message is, and only a sincere request may be
                # answered (ContractAnswer.answers_unprompted); it is told the
                # posting date so a fact about a date already past is not
                # offered as current.
                proactive_posted_at=message.created_at,
                max_output_tokens=settings.proactive_max_output_tokens,
                timeout_seconds=settings.proactive_request_timeout_seconds,
            ),
            settings,
        )
    except TimeoutError:
        _log_deadline_passed(channel, settings)
        return ProactiveResponseOutcome(answers_question=None, posted=False)
    if answer is None:
        return ProactiveResponseOutcome(answers_question=None, posted=False)
    if not answer.answers_unprompted:
        return ProactiveResponseOutcome(answers_question=answer.answers_question, posted=False)

    guild = message.guild
    assert guild is not None  # only reached for a guild message
    facts_by_id = {fact.id: fact for fact in synthesis_facts}
    card = build_answer_card(
        answer,
        synthesis_facts,
        question=None,
        locale=locale,
        channel_names=fact_channel_names(
            guild, {facts_by_id[fact_id].channel_id for fact_id in answer.used_fact_ids}
        ),
        proactive=True,
    )
    assert card.checked_lead is not None  # always set on an answer card
    grounding = await verify_answer_v2(
        build_statements(
            card.checked_lead,
            [(point.text, point.fact_ids) for point in card.checked_points],
            card.cited_fact_ids,
        ),
        [facts_by_id[fact_id] for fact_id in card.cited_fact_ids],
        settings=settings,
        timeout_seconds=PROACTIVE_GROUNDING_TIMEOUT_SECONDS,
    )
    if grounding is not GroundingOutcome.GROUNDED:
        logger.warning(
            "Proactive answer withheld in channel %s: v2 answer check returned %s",
            getattr(channel, "id", "<unknown>"),
            grounding.value,
        )
        return ProactiveResponseOutcome(answers_question=answer.answers_question, posted=False)

    if not await is_channel_enabled(db, channel_id=channel.id):
        return ProactiveResponseOutcome(answers_question=answer.answers_question, posted=False)
    if _stale(freshness, settings, channel):
        return ProactiveResponseOutcome(answers_question=answer.answers_question, posted=False)

    # P7a: the AI label and the privacy line, added after the check (they are
    # template text, never checked) and only when switched on.
    card = with_answer_labels(
        card,
        locale=locale,
        ai_label=settings.ai_label_enabled,
        privacy_line=settings.privacy_info_enabled,
    )
    try:
        if settings.answer_card_style is CardStyle.CONTAINER and components_v2_available():
            await channel.send(
                view=card_to_layout_view(card), allowed_mentions=discord.AllowedMentions.none()
            )
        else:
            await channel.send(
                embed=card_to_embed(card), allowed_mentions=discord.AllowedMentions.none()
            )
    except Exception:
        logger.exception("Proactive post failed in channel %s", getattr(channel, "id", "<unknown>"))
        return ProactiveResponseOutcome(answers_question=answer.answers_question, posted=False)
    return ProactiveResponseOutcome(answers_question=answer.answers_question, posted=True)


async def respond_with_synthesis(
    message: discord.Message,
    *,
    db: aiosqlite.Connection,
    model: TextEmbedding,
    settings: Settings,
    freshness: AnswerFreshness | None = None,
) -> ProactiveResponseOutcome:
    """Synthesize an answer for an already-eligible message and post it, if confident.

    Parameters
    ----------
    message
        The message that cleared the gate.
    db
        Open database connection.
    model
        The loaded embedding model.
    settings
        Loaded configuration: which model to use, and whether one is configured
        at all.
    freshness
        The listener's watch over the conversation since the grace period
        ended (aura.proactive.grace.AnswerWatch); None skips the freshness
        check.

    Returns
    -------
    ProactiveResponseOutcome
        What happened, for the caller to record onto the message's trail. Never
        raises for an expected failure -- missing config, empty facts, a failed
        LLM call, a rejected post -- so the caller's single record path always
        runs.

    Notes
    -----
    Called only behind the gate's ELIGIBLE verdict, by which point a budget slot
    has already been spent for this message, whatever happens next.

    The order of checks is the hard code-gate documented at module level:
    configured -> facts still present -> synthesis succeeded (within its
    deadline) -> model confident and cited -> channel still enabled ->
    conversation still fresh -> post.
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

    # Link expansion, identical to /aura-ask's -- one mechanism, two triggers,
    # per CLAUDE.md. Placed strictly AFTER the emptiness check above, which is
    # what keeps it out of the gate's business: a message that found no fact of
    # its own still posts nothing, so a link can widen an answer Aura was
    # already going to give but can never authorize one it wasn't. Nothing here
    # touches the budget either -- the escalation slot was spent upstream, and
    # this adds facts to one prompt, not a second call.
    synthesis_facts = await expand_with_linked_facts(db, guild_id=guild.id, facts=relevant_facts)

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
    if settings.proactive_answer_format is AnswerFormat.V2:
        # Ships dark (PROACTIVE_ANSWER_FORMAT defaults to legacy); see
        # _respond_in_v2 and this module's docstring.
        return await _respond_in_v2(
            message,
            db=db,
            settings=settings,
            synthesis_facts=synthesis_facts,
            locale=locale,
            proactive_model=proactive_model,
            freshness=freshness,
        )
    try:
        result = await _within_deadline(
            synthesize_answer(
                synthesis_facts,
                message.content,
                locale,
                model=proactive_model,
                question_channel_name=channel_display_name(channel, channel.id),
                question_asked_at=message.created_at,
                fact_channel_names=fact_channel_names(
                    guild, {fact.channel_id for fact in synthesis_facts}
                ),
                extra_body=_proactive_route(settings, proactive_model),
                timeout_seconds=settings.proactive_request_timeout_seconds,
            ),
            settings,
        )
    except TimeoutError:
        _log_deadline_passed(channel, settings)
        return ProactiveResponseOutcome(answers_question=None, posted=False)

    if result is None:
        return ProactiveResponseOutcome(answers_question=None, posted=False)

    # The hard code-gate. answers_question is an additional requirement on top
    # of the numeric gates, never a substitute -- and a claim to answer with no
    # cited fact is treated as not-confident, since a confident answer that
    # draws from nothing is a contradiction.
    if not result.answers_question or not result.used_fact_ids:
        return ProactiveResponseOutcome(answers_question=result.answers_question, posted=False)

    # Against synthesis_facts, not relevant_facts, for the reason /aura-ask
    # states at its own copy of this line: a fact cited only because a link
    # made it available must still reach the grounding check below and the
    # source list in the embed.
    cited_facts = [fact for fact in synthesis_facts if fact.id in result.used_fact_ids]

    # The independent grounding check. It runs before the channel re-check
    # below rather than after, so the ordering stays the one the module
    # docstring states -- everything about whether the answer is FIT to post is
    # settled first, and the freshest-setting read stays the last word before
    # the send. It also means a moderator who disables the channel mid-check is
    # still obeyed either way.
    #
    # A rejection and a failure are both silence, and both are logged rather
    # than posted; see the module docstring for why Trigger 2 differs from
    # /aura-ask here. The escalation slot stays spent, the same documented
    # direction every other refusal on this path takes.
    grounding = await verify_answer_grounded(
        answer=result.answer,
        cited_facts=cited_facts,
        settings=settings,
        timeout_seconds=PROACTIVE_GROUNDING_TIMEOUT_SECONDS,
    )
    if grounding in (GroundingOutcome.UNGROUNDED, GroundingOutcome.CHECK_FAILED):
        logger.warning(
            "Proactive answer withheld in channel %s: grounding check returned %s",
            getattr(channel, "id", "<unknown>"),
            grounding.value,
        )
        return ProactiveResponseOutcome(answers_question=result.answers_question, posted=False)

    # Freshest-setting check: re-read the channel switch right before posting,
    # not the value the pipeline saw seconds ago, so a moderator who toggles the
    # channel off mid-synthesis is obeyed. The slot stays spent -- that is the
    # documented direction for a budget whose job is bounding cost.
    if not await is_channel_enabled(db, channel_id=channel.id):
        return ProactiveResponseOutcome(answers_question=result.answers_question, posted=False)

    # P5c: the conversation must not have moved on while the answer was being
    # written and checked -- the last word before the send, after the channel
    # re-check (see the module docstring).
    if _stale(freshness, settings, channel):
        return ProactiveResponseOutcome(answers_question=result.answers_question, posted=False)

    # Still the full candidate list, not the cited_facts computed above:
    # _build_proactive_embed does its own used_fact_ids filtering, and passing
    # the pre-filtered list would be the same output through a different path.
    # It must be synthesis_facts rather than relevant_facts, though -- a source
    # link for a fact the model cited through a link has to survive into the
    # embed, and filtering against the narrower list would silently drop it.
    embed = label_legacy_embed(
        _build_proactive_embed(result, synthesis_facts, locale),
        locale=locale,
        ai_label=settings.ai_label_enabled,
        privacy_line=settings.privacy_info_enabled,
    )
    try:
        await channel.send(embed=embed)
    except Exception:
        # A post can fail for reasons entirely outside Aura's control: the
        # channel was deleted, send permissions were revoked, Discord returned
        # an error. Fail closed -- no post recorded -- and let the caller record
        # the outcome. CancelledError is a BaseException and still propagates.
        logger.exception("Proactive post failed in channel %s", getattr(channel, "id", "<unknown>"))
        return ProactiveResponseOutcome(answers_question=result.answers_question, posted=False)

    return ProactiveResponseOutcome(answers_question=result.answers_question, posted=True)
