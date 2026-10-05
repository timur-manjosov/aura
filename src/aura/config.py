"""Application configuration loaded from environment variables and `.env`."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_EXAMPLE_HINT = "Copy .env.example to .env and fill in the required values."

# The shortest INTERNAL_API_SECRET accepted. 32 characters of a random token is
# well past brute-forcing over a network, and a floor is what stops "changeme"
# from being the one credential that decides who is on Pro.
MIN_INTERNAL_API_SECRET_LENGTH = 32


def require_strong_internal_api_secret(value: str) -> str:
    """Refuse a short secret, and one that cannot travel in an HTTP header intact.

    Parameters
    ----------
    value
        The raw shared secret.

    Returns
    -------
    str
        `value`, unchanged, when it is acceptable.

    Raises
    ------
    ValueError
        If it is shorter than MIN_INTERNAL_API_SECRET_LENGTH, or contains
        anything but printable ASCII without spaces. The message never contains
        the value.

    Notes
    -----
    One rule for both places a secret is accepted -- the INTERNAL_API_SECRET
    setting, and aura.billing.internal_api itself, which refuses to build a
    listener around a weak secret whoever calls it -- so the two cannot drift
    apart.

    A space, a control character or a non-ASCII character in a bearer
    credential is not a strong secret, it is one that some layer between the
    two services will eventually normalise or strip -- producing a 401 on every
    sync that looks like an outage rather than a typo.
    """
    if len(value) < MIN_INTERNAL_API_SECRET_LENGTH:
        raise ValueError(
            f"INTERNAL_API_SECRET must be at least {MIN_INTERNAL_API_SECRET_LENGTH} "
            'characters (generate one with `python -c "import secrets; '
            'print(secrets.token_urlsafe(48))"`).'
        )
    if not all(
        character.isascii() and character.isprintable() and not character.isspace()
        for character in value
    ):
        raise ValueError(
            "INTERNAL_API_SECRET may contain only printable ASCII characters without spaces."
        )
    return value


# Discord snowflakes are unsigned 64-bit, but every guild ID this project stores
# goes into a SQLite INTEGER, which is signed 64-bit -- the binding raises past
# this value, so a configured ID beyond it is refused at startup instead.
MAX_SQLITE_INTEGER = 2**63 - 1


class ModelComponent(StrEnum):
    """The distinct LLM-calling components, each resolving its own model.

    CLAUDE.md's "LLM Usage & Model Selection" is explicit that fact
    extraction, answer synthesis, digest formatting and proactive relief are
    genuinely different tasks with different requirements, and that no single
    model may be hardcoded as "the" model for the whole project. This enum is
    the closed set of components that go through resolve_model (see
    Settings.resolve_model); adding one here, rather than reading a raw string,
    is what keeps that resolution exhaustive and typo-proof.
    """

    SYNTHESIS = "synthesis"
    PROACTIVE = "proactive"
    EXTRACTION = "extraction"
    SUPERSESSION = "supersession"
    VARIANT = "variant"
    VARIANT_AUDIT = "variant_audit"
    GROUNDING_CHECK = "grounding_check"
    ANSWER_V2 = "answer_v2"
    ANSWER_V2_CHECK = "answer_v2_check"
    EXTRACTION_VERIFY = "extraction_verify"


class CrossGuildBudgetMode(StrEnum):
    """How aura.db.cross_guild_budget reacts once the combined cross-guild
    daily estimate clears the operator's budget (Phase 4a-2).

    WARN (the default): log loudly at WARNING level every time the combined
    estimate is over budget, but refuse nothing -- see
    cross_guild_daily_budget_usd's own comment for why this, not HARD, is the
    safer default for a single self-funded operator.

    HARD: refuse new calls at all five ledgers (proactive escalation,
    extraction, supersession judgment, variant generation, backfill
    distillation) once today's combined estimate is already at or above
    budget, until the next UTC day resets it. Each refusal is handled exactly
    like that ledger's own existing DAILY_CAP_REACHED case -- dropped, paused,
    or skipped per that call site's own established behavior -- so choosing
    HARD changes nothing structurally, only whether the operator-wide ceiling
    can ever actually bind.
    """

    WARN = "warn"
    HARD = "hard"


class BillingMode(StrEnum):
    """Whether this deployment gates Pro features on a subscription (Phase 4c).

    DISABLED (the default): every guild gets every feature, exactly as before
    Phase 4c. Subscription state is still recorded if the internal billing API
    is running -- a subscription that already exists stays visible -- it
    simply decides nothing, and the web backend refuses any new checkout,
    since it would buy nothing. See Settings.billing_mode for why this, not
    ENFORCED, is the default.

    ENFORCED: the Pro-only triggers (proactive relief, automatic extraction,
    the periodic digest, onboarding and backfill) run only for guilds whose
    subscription is in good standing or which the operator lists as
    complimentary. The Free features -- /aura-ask and manual fact management
    -- are never gated, in either mode.
    """

    DISABLED = "disabled"
    ENFORCED = "enforced"


class AnswerFormat(StrEnum):
    """Which answer format one answering trigger uses (P4).

    LEGACY (the default): the free-text synthesis of aura.synthesis, checked by
    aura.grounding and sent as the plain answer embed. Every prompt and every
    message is exactly what it was before this setting existed.

    V2: the structured answer contract of aura.answer_contract, checked point by
    point by aura.answer_check and rendered as an answer card by
    aura.answer_card. It ships dark: a deployment selects it per trigger, with
    ANSWER_FORMAT for /aura-ask and PROACTIVE_ANSWER_FORMAT for proactive relief,
    and neither is switched until its checker has passed acceptance.
    """

    LEGACY = "legacy"
    V2 = "v2"


class CardStyle(StrEnum):
    """How a v2 answer card is drawn in Discord (P4).

    EMBED (the default): a classic embed -- accent colour, the question in the
    author line, the answer in the description, the sources as a field. An
    embed never notifies anyone it mentions and renders on every client.

    CONTAINER: a Components V2 container with the same parts, set in subtext
    and separated by a divider. It cannot carry an embed or plain content, and
    is always sent with mentions disabled, since a text display can ping.
    """

    EMBED = "embed"
    CONTAINER = "container"


class MessageLook(StrEnum):
    """Which look one message family has (P5).

    CLASSIC (the default): the message exactly as before P5.

    CARD: the family's card from aura.cards -- the design system of the v2
    answer card, drawn in ANSWER_CARD_STYLE (a classic embed or a Components V2
    container). One setting per family (DIGEST_LOOK, ONBOARDING_LOOK, PLAN_LOOK,
    NOTICE_LOOK), so each can be switched, and switched back, on its own.
    """

    CLASSIC = "classic"
    CARD = "card"


class ConfigurationError(Exception):
    """Raised when application configuration is missing or invalid.

    Kept distinct from pydantic's ValidationError so callers (main.py) can
    catch one application-specific exception and print an actionable
    message, instead of parsing pydantic's internal error structure.
    """


class Settings(BaseSettings):
    """Typed, validated application settings.

    Values are sourced from real process environment variables first, then
    from a `.env` file (see `.env.example`); environment variables take
    precedence. Field names are matched to environment variables
    case-insensitively (e.g. `discord_token` <-> `DISCORD_TOKEN`).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Phase 4c: a refused value is never echoed into the error. Pydantic
        # otherwise includes `input_value` -- for a model-level validator,
        # every input at once -- so a traceback around settings loading would
        # carry DISCORD_TOKEN, LLM_API_KEY and INTERNAL_API_SECRET with it.
        hide_input_in_errors=True,
    )

    # No default (would make a blank/missing token indistinguishable from a
    # deliberate empty value); validate_default=True forces the validator
    # below to run even when the variable is absent entirely.
    #
    # Every credential in this class is a SecretStr (Phase 4c audit, F-10): its
    # repr and str are masked, so printing the settings object -- a debug log
    # line, a traceback renderer that shows locals -- cannot leak one. The plain
    # value is read with .get_secret_value() only where it is actually used.
    discord_token: SecretStr = Field(default="", validate_default=True)
    # LLM_PROVIDER, LLM_API_KEY, and SYNTHESIS_MODEL are all optional and unset
    # by default, on purpose: the bot as a whole -- every command except
    # /aura-ask -- must start and run completely normally with none of these
    # present, so a deployment that never wants LLM features simply omits them.
    # The defaults stay None even though .env.example now ships a measured model
    # choice: a missing key with a hardcoded model here would look "configured"
    # in code while failing on every call. is_llm_configured() below is the one
    # place that decides whether enough is actually here to make a call.
    llm_provider: str | None = None
    llm_api_key: SecretStr | None = None
    synthesis_model: str | None = None
    # Proactive relief's own model (CLAUDE.md's second trigger), resolved
    # through resolve_model like every other component. Its own config value on
    # purpose -- CLAUDE.md forbids assuming one model fits every task -- but NOT
    # assumed to differ from synthesis_model by default: left unset, it falls
    # back to synthesis_model (see resolve_model), so a deployment that
    # configures a single model still has a working second trigger. The
    # bake-off that chose the shipped value, and the evidence behind it, is at
    # the proactive synthesis call site (aura.proactive.responder); it landed on
    # the same model as synthesis, which is why the fallback above is a
    # convenience rather than the decision itself.
    proactive_model: str | None = None
    # Automatic fact extraction's own model (Phase 3a-2's distillation call),
    # resolved through the same seam as the two above. Falls back to
    # synthesis_model when unset, for the same reason proactive_model does: a
    # deployment that configures one model should still have a working third
    # call site rather than a silently dead one.
    #
    # SHIPPED VALUE: claude-haiku-4.5, and this is an ASSUMPTION CARRIED OVER
    # FROM PHASE 2, not a bake-off of its own -- stated plainly here because the
    # two other models in this file were chosen by measurement and a reader
    # would otherwise reasonably assume this one was too. Phase 2's bake-off
    # (reports/model-bakeoff.txt) measured a similarly-shaped task -- strict
    # JSON out, a judgement about whether text genuinely supports a claim,
    # across nine locales -- and found Haiku at the judgement ceiling (12/12,
    # and 6/6 on the two cases that specifically separate a calibrated model
    # from an optimistic one) with the two cheaper candidates losing on
    # calibration rather than on format or language. Distillation asks for the
    # same trait in the same shape: decide whether a message really asserts
    # something checkable, and refuse when it only looks like it does.
    #
    # Where the transfer is weakest, since a carried assumption should say
    # where it might break: distillation is a *generative* task (write a new
    # distilled sentence) as well as a judgement one, at far higher volume,
    # over raw chat rather than over already-distilled facts. Nothing in Phase
    # 2's evidence speaks to generation quality or to bulk-volume cost. Revisit
    # with a real bake-off if live usage shows distillation quality problems a
    # better model would plausibly fix -- see reports/phase-3a-2.txt, which
    # measures this model's actual behaviour on the cases this phase cares
    # about but does not compare it against alternatives.
    extraction_model: str | None = None
    # The supersession-judgment call's own model (Phase 3a-3): given an existing
    # active fact and a freshly distilled candidate that scored above
    # EXTRACTION_DEDUP_SIMILARITY_THRESHOLD against it, decide what the
    # relationship actually is -- supersession, complementary, contradiction, or
    # an embedding false positive. Resolved through the same seam as the three
    # above, and falling back to synthesis_model for the same reason.
    #
    # SHIPPED VALUE: claude-haiku-4.5, and unlike extraction_model above this
    # one WAS chosen by a bake-off of its own -- 120 real calls over 32
    # hand-written fact pairs across three candidates, written up in
    # reports/supersession-model-bakeoff.txt.
    #
    # THE DECIDING FINDING, in short, because it is not the number a reader
    # would expect: raw accuracy did not decide this. Sonnet 4.5, Haiku 4.5 and
    # Gemini 3.1 Flash Lite scored 92% / 92% / 95% -- a three-way near-tie, with
    # the cheapest-but-one nominally ahead. What separated them was the
    # DIRECTION of their mistakes. An over-confident judgement actively misleads
    # the moderator who reads it toward acting on a pair that is not actually
    # settled; an over-cautious one costs an extra manual look at something they
    # were already going to review. Haiku was the only candidate with zero
    # mistakes in the dangerous direction (0 dangerous / 3 conservative, against
    # Sonnet's 1/2 and Gemini's 2/0), and it was alone in getting the single
    # most important case right: two facts stating different numbers for the
    # same rule with NO transition language between them, which Sonnet and
    # Gemini both proposed as a confident supersession and Haiku correctly
    # escalated as a contradiction. That is precisely the failure this call
    # exists to avoid, so the cheaper model won on merit rather than on price --
    # the same trait ("fails toward the safe answer under ambiguity, reliably")
    # reports/model-bakeoff.txt found when choosing PROACTIVE_MODEL, observed
    # again here on a structurally different task.
    #
    # Cost is deliberately NOT the primary axis here, unlike extraction_model:
    # the dedup threshold already narrows this call to a small, advisory-only
    # slice of extraction's volume, and supersession_daily_cap below bounds the
    # worst case regardless.
    supersession_model: str | None = None
    database_path: str = "data/aura.db"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Phase 1d's own test data showed related content scoring around 0.98
    # cosine similarity and unrelated content around 0.08 against this
    # project's embedding model -- 0.4 sits comfortably above the
    # "unrelated" end with real margin to spare, while still being loose
    # enough that a reasonably-phrased question matches its facts. This is
    # specifically the direct-query bar. Proactive relief reuses
    # find_similar_facts with its own threshold, PROACTIVE_SIMILARITY_THRESHOLD
    # below -- since the 2026-08-15 operational decision a LOOSER one (0.20),
    # backstopped by the model's own answers_question judgement.
    #
    # Since the hybrid retrieval of 2026-10-02 this is no longer the only way a
    # fact reaches /aura-ask's synthesis: a fact below it still qualifies when
    # the question's own words cover it (the three ASK_LEXICAL_* settings
    # below; see aura.retrieval.hybrid). A fact at or above it qualifies
    # exactly as before.
    similarity_threshold: float = 0.4

    # --- /aura-ask word matching (hybrid retrieval) -------------------------
    # For a single keyword, an inflected form, a compound or a casual phrasing
    # the embedding model scores near its noise floor (~0.20 against ANY fact),
    # so "Mentoriate" never reached two facts about "Mentoriat". The question's
    # own words close that gap, locally and for free: a fact also qualifies
    # when its lexical coverage (aura.retrieval.lexical: the IDF-weighted share
    # of the question's content words the fact contains, inflection, compounds
    # and long-word typos included) reaches the first number below while its
    # similarity reaches the second. Qualifying facts are then ranked by
    # similarity + the third number x coverage.
    #
    # All three come from the quality diagnosis of 2026-10-02 (Section 4),
    # measured on 77 hand-written questions over a real guild's facts plus 24
    # unrelated ones: 74 instead of 47 questions found their facts, the same 6
    # unrelated questions selected anything as before, precision 0.90 instead
    # of 0.83. Used by /aura-ask only; proactive relief and extraction dedup
    # never read them.
    #
    # COVERAGE: half of the question's weighted subject must be in the fact.
    # One rare word of a two-word question is enough; one common word next to a
    # rare one that no fact contains ("Owner vom Server") is not. Above 0 by
    # construction -- 0 would admit every fact above the floor.
    ask_lexical_coverage_threshold: float = Field(default=0.5, gt=0.0, le=1.0, allow_inf_nan=False)
    # FLOOR: a fact found by its words must still not be unrelated in meaning.
    # Without it, "Wie werde ich Mentor?" matched a maintenance fact through
    # "werde"/"werden" at a similarity of -0.03. 0.05 removed that and lost
    # nothing measured; 0.10 already cost one question one of its two facts,
    # 0.15 a question its only match.
    ask_lexical_similarity_floor: float = Field(default=0.05, ge=-1.0, le=1.0, allow_inf_nan=False)
    # RANKING WEIGHT: how much coverage adds when qualifying facts compete for
    # the SYNTHESIS_FACT_LIMIT places. 0.5 puts a fact containing the asked
    # word ahead of one that merely shares the sentence shape ("Wann findet ...
    # statt?"), which the embedding alone ranked first. 0 ranks by similarity.
    ask_lexical_ranking_weight: float = Field(default=0.5, ge=0.0, le=10.0, allow_inf_nan=False)

    # --- Proactive relief (CLAUDE.md's second trigger) ---------------------
    # Every number below is a PLACEHOLDER pending recalibration against real
    # proactive_signals data in Phase 2b. They were chosen by measurement
    # rather than by feel, but the measurement was a hand-written eval set of
    # ~50 sentences and 5 facts, which is small enough that the live table is
    # the authority the moment it has data. None of them is final.

    # Stage 1: minimum contrastive question-likeness score (question-exemplar
    # similarity minus statement-exemplar similarity, so the useful range is
    # roughly [-0.2, +0.3], NOT [0, 1]).
    #
    # Measured on a 27-question / 27-statement held-out set across all nine
    # locales: accuracy peaks at -0.03 (94.4%, recall 0.93), and full recall
    # needs -0.07 or below (specificity 0.67). Phase 2a-3 chose -0.08 --
    # looser than the accuracy optimum, deliberately -- because the two
    # errors are not symmetric here. A false negative at Stage 1 is permanent
    # silence: the message is never considered again, and Aura fails at the
    # one job it has. A false positive costs one Stage 2 evaluation, which is
    # local CPU work and free, and Stage 2's own bar is what actually keeps
    # noise out.
    #
    # For context on why this is contrastive at all: the one-sided score it
    # replaces measured 66% on Phase 2a-1's own held-out set, below a naive
    # "contains a question mark" baseline. The contrastive score above beats
    # that baseline (94% vs 78% on the set measured here).
    #
    # RECALIBRATED for Phase 2b-3, -0.08 -> -0.15, against the real 580-message
    # synthetic corpus (reports/phase-2b-2.txt Section 5): at -0.15, recall on
    # genuine information requests is 0.982 (up from 0.903 at -0.08) -- roughly
    # 30 fewer real questions silently and permanently dropped before Stage 2
    # ever sees them, at negligible extra cost, since a Stage 1 pass only buys
    # one free local Stage 2 check. Not pushed further to -0.22, where accuracy
    # actually peaks on this corpus: that buys only 0.018 more recall (0.982 ->
    # 1.000) for a much larger drop in specificity (0.150 -> 0.033) -- far more
    # non-question traffic reaching Stage 2 for essentially no recall benefit.
    # This is the numeric half of the Phase 2b-3 product decision documented in
    # CLAUDE.md's "Proactive Relief: Visibly Active by Design" section: Aura is
    # now tuned to let real questions through rather than drop them silently.
    #
    # OPERATIONAL DECISION (2026-08-15, made permanent 2026-08-26): -0.15 -> -0.22,
    # the peak-accuracy point already named (but not shipped) in
    # reports/phase-2b-2.txt Section 5a/5b/5c: recall 0.982 -> 1.000 (373/380 ->
    # 380/380), accuracy unchanged (0.868) and F1 essentially unchanged (0.928 ->
    # 0.929), for a specificity drop from 0.150 to 0.033 -- more non-question
    # traffic reaching Stage 2. Stage 1 only gates one free, local Stage 2 check,
    # so the wider specificity loss costs nothing regardless of server size,
    # while a Stage 1 false negative is still permanent silence. Timur has
    # decided this value is the standing production default rather than a
    # placeholder to revert once real community traffic arrives -- see
    # reports/operational-values-decision.txt.
    proactive_question_threshold: float = Field(default=-0.22, gt=-2.0, le=2.0, allow_inf_nan=False)

    # Stage 2: minimum cosine similarity between the message and the best
    # matching fact. Separate from similarity_threshold above, and stricter,
    # because the failure modes differ in kind: a weak direct answer is shown
    # only to the person who asked for it, while a weak proactive answer
    # interrupts everyone in the channel unasked.
    #
    # RECALIBRATED for Phase 2a-3 from the 0.75 placeholder to 0.45, and still
    # a placeholder pending Phase 2b's recalibration against real production
    # data. Phase 2a-2 shipped 0.75 while nothing posted, but its own attack
    # pass measured that it keeps the gate essentially shut: a question and the
    # fact that answers it are not paraphrases of each other, and this
    # embedding model scores that asymmetry far lower than Phase 1d's note
    # about "related content around 0.98" suggests (that was paraphrase-to-
    # paraphrase). Eight genuine repeat questions measured against the facts
    # answering them scored 0.53-0.78 (median 0.63); only one of eight cleared
    # 0.75. Unrelated messages in the same measurement scored 0.19-0.26.
    #
    # 0.45 cleared all eight measured true matches (the lowest was 0.53, leaving
    # ~0.08 of margin below it -- the same "don't sit exactly on the lowest
    # real observation" reasoning proactive_question_threshold uses) while
    # staying well above the 0.19-0.26 unrelated band. It was deliberately at
    # the low end of the 0.45-0.5 band that data pointed to: Stage 2 is no
    # longer the last line of defence now that synthesis posts, because the
    # LLM's own answers_question self-assessment and the confidence gap below
    # both have to agree before anything is sent, so a somewhat permissive
    # similarity bar here is backstopped rather than load-bearing on its own.
    #
    # RECALIBRATED for Phase 2b-3, 0.45 -> 0.30, together with
    # proactive_confidence_gap (0.15 -> 0.05) below -- these two move as a
    # pair, per reports/phase-2b-2.txt Section 7's similarity x gap sweep.
    # At (0.30, 0.05) recall on genuinely answerable questions is 0.69 at
    # specificity 0.37 -- roughly three times the real answer rate the old
    # (0.45, 0.15) pair produced (16 of 80 answerable repeats reached
    # synthesis end-to-end; Section 12), while still correctly holding back
    # over a third of cases that should not escalate at all. This is a real
    # cost lever now, not a free one: a materially larger share of eligible
    # messages reaches paid Stage 3 synthesis than before. That trade is the
    # explicit Phase 2b-3 product decision (see CLAUDE.md) -- Timur accepted
    # the cost increase in exchange for Aura being genuinely, visibly active,
    # bounded by proactive_daily_cap below, at a server count small enough
    # that the worst case is still cheap in absolute terms.
    #
    # UNCHANGED in value by Phase 2b-4, but now load-bearing in two places
    # instead of one, and the second is a bug fix rather than an extension.
    # This is Trigger 2's ONE fact-relevance bar: the gate escalates on it, and
    # aura.proactive.responder now selects the facts it sends to synthesis on
    # it too. Until Phase 2b-4 the responder filtered on similarity_threshold
    # (0.40) instead, so a message whose best fact scored between 0.30 and 0.40
    # was granted an escalation slot and then found nothing to answer from --
    # 45 of 580 corpus cases. Since proactive_confidence_gap retired above,
    # this is also the only Stage 2 number left, so it no longer "moves as a
    # pair" with anything.
    #
    # OPERATIONAL DECISION (2026-08-15, made permanent 2026-08-26): 0.30 -> 0.20,
    # read off reports/phase-2b-4.txt Section 7's post-retirement sweep, gap=0.00
    # column (the operative column now that PROACTIVE_CONFIDENCE_GAP no longer
    # gates anything): recall 0.85 -> 0.99, specificity 0.07 -> 0.02, the lowest
    # value in that sweep table (no data below 0.20 to justify going further).
    # Unlike Stage 1 above, this bar directly gates paid Stage 3 synthesis
    # calls, so it is a real, ongoing cost lever -- accepted for the reason
    # CLAUDE.md's "Proactive Relief: Visibly Active by Design" documents: Timur
    # wants Aura visibly, actively helpful and has accepted the resulting cost
    # increase as standing production policy, bounded by the unchanged
    # PROACTIVE_DAILY_CAP. See reports/operational-values-decision.txt.
    proactive_similarity_threshold: float = Field(
        default=0.20, ge=-1.0, le=1.0, allow_inf_nan=False
    )

    # RETIRED in Phase 2b-4. This value no longer gates anything: it is read,
    # validated, and then used by nothing. Read aura.proactive.gate's module
    # docstring for the reasoning; the short version is that it was asked to
    # separate "two facts compete because one is stale" from "two facts compete
    # because both are relevant and complementary", those two produce the same
    # number, and the distinction is a judgement about meaning that Stage 3
    # makes instead.
    #
    # It is retained as a field rather than deleted outright, and that is a
    # deliberate compatibility decision rather than an oversight. Settings runs
    # under pydantic-settings' extra="forbid", which tolerates undeclared
    # process environment variables but REJECTS an undeclared key in a .env
    # file -- verified, not assumed. Every deployment that copied .env.example
    # since Phase 2a-2 has PROACTIVE_CONFIDENCE_GAP in its .env, so deleting
    # the field here would turn a routine `git pull && docker compose up` into
    # a container that will not start, with a pydantic traceback as its only
    # explanation. Silently refusing to boot is a far worse outcome than one
    # inert setting, so the field stays until a phase that is willing to own a
    # migration note removes it.
    #
    # Nothing reads it, so nothing can regress if it is mis-set. It is also NOT
    # passed to ProactiveGateConfig any more -- an unused field on the gate's
    # own config would invite exactly the "wait, does this still do something?"
    # question this comment exists to answer.
    proactive_confidence_gap: float = Field(default=0.05, ge=0.0, le=2.0, allow_inf_nan=False)

    # Per-channel cooldown, in seconds, on becoming eligible for synthesis.
    # 15 minutes caps an active channel at four unsolicited messages an hour
    # even in the worst case, which is well under the rate at which a bot
    # starts reading as noise -- and CLAUDE.md asks for proactive relief to be
    # "deliberately conservative, to avoid unwanted interruptions."
    # Upper-bounded, not merely non-negative. The cutoff this feeds is
    # computed as `now - timedelta(seconds=...)`, which raises rather than
    # saturating, so an absurd value here would fail on every message instead
    # of at startup. The bound duplicates MAX_COOLDOWN_SECONDS in
    # aura.db.proactive_state, which is the layer that does the arithmetic and
    # enforces it again; a test asserts the two agree, since config.py must
    # not depend on the data layer to state its own limits.
    proactive_cooldown_seconds: float = Field(
        default=900.0, ge=0.0, le=30 * 24 * 60 * 60.0, allow_inf_nan=False
    )

    # Per-guild, per-UTC-day ceiling on how many messages may become eligible
    # for paid synthesis. The inner safety net; the OpenRouter account's own
    # spending cap is the outer one. Counts eligibility rather than answers on
    # purpose, so a message that reaches synthesis and fails still spends its
    # slot -- otherwise a reliably-failing model would grant unlimited retries
    # to anyone who could trigger it. 0 is valid and disables proactive
    # escalation entirely.
    # Upper bound mirrors MAX_DAILY_CAP in aura.db.proactive_state: the value
    # is bound into SQL, and sqlite3 refuses an int that does not fit a signed
    # 64-bit integer.
    #
    # RECALIBRATED for Phase 2b-3, 20 -> 60. Not chosen from a sweep like the
    # three thresholds above -- it is a deliberate ceiling raise to match them:
    # at the loosened Stage 1/2 settings, a genuinely active server could
    # plausibly exceed the old cap of 20 on a busy day, which would silently
    # reintroduce the exact "quiet by accident" problem this phase exists to
    # fix, just relocated from Stage 1/2 to the daily cap. 60/day at Haiku's
    # measured ~$0.001-0.003 per Stage 3 call (see reports/model-bakeoff.txt)
    # bounds worst-case cost at roughly $2-5 per guild per month even at full
    # utilization every single day -- negligible at the handful of test and
    # community servers Aura realistically runs on right now, and explicitly
    # accepted by Timur as the cost of visible activity (see CLAUDE.md). This
    # is a value to revisit before any large multi-guild rollout: the math
    # here is per-guild, and Phase 4a-2's cross_guild_daily_budget_usd below
    # is the layer that now bounds the SUM once many guilds share one
    # operator key (see CLAUDE.md's Open Items) -- this field's own worst
    # case is unchanged either way.
    proactive_daily_cap: int = Field(default=60, ge=0, le=1_000_000)

    # Phase 2b-1: how long Aura waits, after a message becomes eligible for
    # paid synthesis, before actually calling the LLM -- giving a human the
    # chance to answer first. A PLACEHOLDER in the 60-120s range pending real
    # tuning, same treatment as every other threshold above: chosen to be long
    # enough that an active channel's regulars have a realistic chance to
    # reply, short enough that a genuinely unanswered question does not sit
    # unaddressed for many minutes.
    #
    # Deliberately unbounded above beyond a generous sanity ceiling rather than
    # tied to proactive_cooldown_seconds: nothing in this phase requires
    # grace < cooldown, but the wake-time freshness recheck (see
    # aura.db.proactive_state.is_still_freshest_escalation) exists specifically
    # to stay safe if an operator sets grace_period_seconds so long that a
    # second message in the same channel clears cooldown and escalates before
    # the first one's grace period even ends.
    #
    # OPERATIONAL DECISION (2026-07-31, made permanent 2026-08-26): 90 -> 13.
    # The 60-120s placeholder above was sized for the collision this grace
    # period exists to avoid -- Aura and a human both answering the same
    # question -- and 90s of silence before an eligible message gets its
    # proactive answer reads as slow. 13s keeps a wait long enough to be a real
    # grace period rather than none at all, while keeping the pipeline
    # responsive. Timur has accepted the resulting collision risk once real,
    # concurrent multi-member channel activity arrives, in exchange for Aura
    # feeling responsive now -- this is a standing production value, not one
    # to revert later. See reports/operational-values-decision.txt.
    proactive_grace_period_seconds: float = Field(
        default=13.0, ge=0.0, le=24 * 60 * 60.0, allow_inf_nan=False
    )

    # --- Automatic fact extraction (CLAUDE.md's Phase 3a, first filter only) --
    # Minimum contrastive fact-worthiness score (fact-worthy-exemplar similarity
    # minus not-fact-worthy-exemplar similarity; see
    # aura.extraction.fact_worthiness) for a message to be worth extraction's
    # attention at all. Same scale and same reasoning shape as
    # proactive_question_threshold above: a difference of two cosine
    # similarities, so the useful range sits well inside [-2, 2] rather than
    # spanning it.
    #
    # Not yet read by any live code path: Phase 3a-1 ships the filter
    # calibrated but unwired (see aura.extraction and
    # reports/phase-3a-1.txt), the same "value before wiring" order Settings
    # has followed for every threshold since Phase 2a-1. Phase 3a-1b
    # (reports/phase-3a-1b.txt) re-calibrated this value against a larger
    # corpus without wiring anything -- still a PLACEHOLDER pending real
    # usage data, the same as every synthetic-corpus-derived threshold in
    # this project regardless of corpus size.
    #
    # CALIBRATED for Phase 3a-1b against a 2,127-message synthetic corpus (9
    # locales, 25 fact-worthy cases per locale, ~10.6% fact-worthy / 89.4%
    # ordinary+hard-negative chat by construction; see reports/phase-3a-1b.txt
    # and scripts/extraction_corpus/). This supersedes Phase 3a-1's 443-message
    # corpus (5 fact-worthy/locale), which its own report flagged as too small
    # to trust the exact threshold position or any single locale's number.
    #
    # -0.02 is the F1-maximising point of the full precision/recall sweep
    # (P=0.609, R=0.707, specificity=0.946, F1=0.654) and stays the optimum
    # whether or not the 37 label-audit-disputed cases are excluded (F1=0.665
    # excluding them, P=0.605, R=0.739) -- both views were swept independently
    # and happened to agree, the same cross-check 3a-1 ran. The optimum moved
    # by 0.01 from 3a-1's -0.03 -- a real shift, reported rather than held for
    # continuity, though within the noise either corpus's own resampling would
    # produce.
    #
    # Deliberately NOT pushed looser to chase recall the way
    # proactive_question_threshold was: that threshold only gates one more
    # free local check (Stage 2), so a false negative there is cheap to
    # tolerate and a missed real question is a permanent silence. Here a
    # message that clears this bar is headed for Phase 3a-2's paid,
    # per-message LLM extraction call -- CLAUDE.md's own LLM Usage section
    # names automatic extraction as running "on every incoming message across
    # every connected server," the highest-volume, most cost-sensitive call
    # site in the whole project. A false negative here just means one
    # real fact is not captured automatically this one time (manual entry via
    # the "Add as Aura Fact" context menu still exists); a false positive
    # spends a paid call on ordinary chat, at extraction's volume. That
    # asymmetry is the opposite of Stage 1's, so this threshold sits at the
    # precision-favouring optimum rather than being loosened past it.
    #
    # Honest limitations, not smoothed over: the two hard-negative categories
    # (hedged speculation -- "I think the event might be Saturday"; rule-shaped
    # jokes, quotes and hypotheticals) still score measurably closer to real
    # facts than ordinary chat does at this threshold (hedged_speculation
    # false-positive rate 7.5%, adversarial_noise 12.9%, vs. 2.6% against
    # ordinary chat) -- improved from 3a-1's combined 14.2%, but not resolved,
    # and this filter was never meant to resolve that ambiguity alone, the same
    # way PROACTIVE_SIMILARITY_THRESHOLD was never meant to resolve a
    # stale-vs-complementary conflict alone (see aura.proactive.gate); a future
    # Phase 3a-2 extraction call is where that judgement belongs.
    #
    # Per-locale performance at n=25 positive/locale ranges from F1=0.784 (tr)
    # down to F1=0.432 (en-US) -- real spread, but NOT the same spread 3a-1
    # reported at n=5 (there: ja best at F1=1.000, pl worst at F1=0.250). At
    # 5x the sample, ja fell to mid-pack (F1=0.627) and pl rose to
    # second-best (F1=0.755): 3a-1's own caveat that its per-locale spread was
    # "dominated by small-sample noise rather than proof of a real per-locale
    # gap" is borne out directly by watching the ranking scramble, not just
    # asserted. en-US's weak showing here is new and specific: 11/40 (27.5%)
    # of its hedged-speculation cases score above this threshold, the worst
    # hedge leakage of any locale -- see reports/phase-3a-1b.txt.
    #
    # OPERATIONAL DECISION (2026-07-31, made permanent 2026-08-26): -0.02 above
    # remains the calibrated, precision-favouring value the rest of this
    # comment block justifies for extraction running at real, full-traffic
    # volume across a community server (reports/phase-3a-2.txt Section 8's cost
    # math: up to ~$16/guild/month worst case at full daily-cap utilization).
    # -0.04 is used instead, taken directly from the same phase-3a-1b.txt sweep
    # (Section 6) rather than any new calibration: P=0.553, R=0.787,
    # specificity=0.925, F1=0.650 -- a real recall gain over -0.02's R=0.707,
    # at essentially the same F1, and deliberately short of -0.05 (P=0.495,
    # already below half) and -0.06 (P=0.474, F1=0.608, worse than -0.02's),
    # where the sweep tips into flagging more noise than signal. Timur has
    # accepted the resulting full-volume cost as standing production policy,
    # not a testing-phase concession to revert later.
    #
    # RECONSIDERED (2026-08-15) for further loosening and KEPT UNCHANGED: this
    # is already the recall-favouring value chosen just above, and the same
    # phase-3a-1b.txt Section 6 sweep this comment already cites shows the next
    # step, -0.05, crossing into the range that comment itself already flags as
    # degenerate (precision 0.495, just under half, F1 0.619 below -0.04's own
    # 0.650). No further loosening is proposed here. See
    # reports/operational-values-decision.txt.
    extraction_fact_worthiness_threshold: float = Field(
        default=-0.04, gt=-2.0, le=2.0, allow_inf_nan=False
    )

    # How long candidate messages accumulate in one channel before being sent
    # to the distillation model as a single batch (see aura.extraction.pipeline).
    #
    # Batching at all is the point: nobody is waiting on an automatically
    # extracted fact -- unlike Trigger 1, where a user watches a deferred
    # interaction, and unlike Trigger 2, where a channel is mid-conversation --
    # so extraction can trade latency it does not need for a call count it does.
    # Ten messages batched into one call is one call instead of ten, over
    # roughly the same tokens, at the project's most cost-sensitive call site.
    #
    # 300s (5 minutes) is the low end of the 5-10 minute range the phase brief
    # proposed, and low deliberately: the batch window is also the window in
    # which an edit or a deletion can still withdraw a message before anything
    # is distilled from it (see aura.extraction.pipeline), and a longer window
    # holds raw message text in extraction_queue for longer. Both argue for the
    # short end; only call-count efficiency argues for the long end, and it
    # keeps almost all of its benefit at five minutes on any channel busy
    # enough for batching to matter at all.
    extraction_batch_window_seconds: float = Field(
        default=300.0, ge=0.0, le=24 * 60 * 60.0, allow_inf_nan=False
    )

    # Hard ceiling on how many messages go into one distillation call.
    #
    # A bound on the worst case, not a target: the window above says "wait five
    # minutes", and a channel that receives four hundred fact-worthy-looking
    # messages in those five minutes would otherwise produce one enormous
    # prompt. With per-message truncation in the prompt builder (see
    # aura.extraction.distiller), 20 messages is a worst case of roughly 20k
    # characters -- a few cents at the shipped model's pricing -- rather than an
    # unbounded one. Anything over the limit simply waits for the next sweep,
    # where it is already past its window and flushes immediately; it is not
    # dropped.
    extraction_batch_max_messages: int = Field(default=20, ge=1, le=1000)

    # Per-guild, per-UTC-day ceiling on DISTILLATION CALLS, mirroring
    # proactive_daily_cap exactly (same ledger shape, same atomic acquisition,
    # same durability across restarts -- see aura.db.extraction_state).
    #
    # Counts calls rather than extracted facts, for the same reason the
    # proactive cap counts eligibility rather than answers: a reliably-failing
    # model must not earn unlimited retries. 0 is valid and disables automatic
    # extraction entirely while leaving the rest of the pipeline configured.
    #
    # 50 is chosen against measured pricing rather than by feel, and against
    # what a call actually costs at this batch size: at the shipped model's
    # $1/$5 per Mtok, a full 20-message batch measures at roughly 6k input and
    # under 1k output tokens, about $0.011 -- so 50 calls a day is a worst case
    # near $16 per guild per month IF every single call were a maximum-size
    # batch every day, and realistically a small fraction of that, since a real
    # batch is a handful of messages rather than twenty. It is deliberately a
    # tighter bound in call terms than proactive_daily_cap's 60 despite
    # extraction being the higher-volume trigger: batching means one call here
    # covers many messages, so 50 calls a day is a great deal more coverage
    # than 60 escalations a day is. This cap's own per-guild worst case is
    # unchanged by Phase 4a-2's cross-guild budget below, which bounds the SUM
    # across every guild sharing one operator key rather than replacing this
    # number (see CLAUDE.md's Open Items).
    extraction_daily_cap: int = Field(default=50, ge=0, le=1_000_000)

    # Similarity at or above which a freshly distilled candidate is flagged as
    # possibly restating an existing active fact (see aura.extraction.pipeline).
    # Since Phase 3a-3 this flag also gates a paid judgement call
    # (aura.extraction.supersession), so "advisory only" no longer means "no
    # calibration needed" -- a false positive now spends a real, if small and
    # capped, judgement slot. reports/extraction-dedup-threshold-calibration.txt
    # is that calibration: 105 hand-written pairs across all nine locales
    # (25+ each of duplicate/paraphrase, genuine supersession, genuine
    # contradiction -- all three "should mark" -- plus 15 thematically-similar-
    # but-different-subject pairs and 15 genuinely unrelated pairs, both
    # "should not mark"), scored through the real, shipped fastembed model.
    #
    # THE HONEST HEADLINE FINDING: no single threshold cleanly separates a
    # weakly-worded genuine restatement from a strongly-worded false positive,
    # because their score distributions substantially overlap (should-mark:
    # 0.260-0.991; thematically-similar-but-unrelated: 0.497-0.924, nearly the
    # same median). This is the same shape as the retired
    # PROACTIVE_CONFIDENCE_GAP finding, one call site earlier: a status-change
    # supersession that keeps almost none of the predecessor's wording (a
    # channel closed, a role handed over) can score LOWER than an unrelated
    # pair that merely shares a sentence template. Neither raw F1 sweep is
    # usable as a result because of it: the full-corpus optimum (0.25) is
    # dominated by the trivially-separable unrelated pairs and marks 100% of
    # the thematically-similar ones; the sweep restricted to should-mark vs.
    # that one hard category degenerates further, to "mark everything", since
    # should-mark outnumbers it 5:1 and F1 rewards recall almost
    # unconditionally at that ratio. The report picks a value off the
    # resulting precision/recall Pareto frontier by hand instead, the same
    # "the optimum is a data point, not an instruction" stance every threshold
    # in this file already takes.
    #
    # 0.60, down from the unmeasured 0.70 placeholder, because the cost
    # asymmetry here is the OPPOSITE of Stage 1's fact-worthiness filter: a
    # false positive no longer buys a full extraction call, only a ~$0.001
    # judgement call bounded by SUPERSESSION_DAILY_CAP and -- per
    # reports/phase-3a-3.txt's own re-verification -- one the shipped judge
    # resolves correctly as "independent" on exactly this report's hardest
    # false-positive shapes, while a false negative silently drops the one
    # thing this call exists to catch, with no compensating signal at all. At
    # 0.70 the corpus's genuine supersessions were caught only 36% of the time
    # (9/25) -- a status change or a name/role handover routinely scores below
    # a strict paraphrase bar -- against 76% (19/25) at 0.60, while duplicates
    # and contradictions stay 90%+ caught at both. The cost: marking the one
    # hard false-positive category (thematically similar, different subject)
    # rises from 67% to 87%, including one of the two named Phase 3a-3 attack
    # cases (independent-upload-limit-different-channel, score 0.698, sits
    # right at the old bar) -- accepted rather than overlooked, on the same
    # reasoning: that exact case is the one this project already measured the
    # judge getting right. Both attack cases stay held back well above 0.70,
    # so raising this value instead would cost nothing on them specifically --
    # it is the supersession recall above that the higher bar was actually
    # giving up.
    #
    # OPERATIONAL DECISION (2026-08-15, made permanent 2026-08-26): 0.60 -> 0.53,
    # the "RECALL-LEANING ALTERNATIVE" reports/extraction-dedup-threshold-
    # calibration.txt Section 4 names (+0.527 on its Pareto frontier, folded
    # here into config.py's existing 2-decimal style). Unusually clean trade
    # for this corpus: the hard-negative sweep in that report shows IDENTICAL
    # false-positive behaviour on the independent_related category at 0.60 and
    # 0.527 (fp=13, tn=2, specificity 0.133 at both), while should-mark recall
    # (duplicate + supersession + contradiction combined) rises from 0.867 to
    # 0.933 (65/75 -> 70/75) --  no measured cost, in this corpus, for the extra
    # recall. Both of the report's two named attack cases behave identically to
    # the 0.60 setting (0.698 marked, 0.509 held back at both). This raises
    # real call volume: more candidates clear this bar and reach the paid
    # supersession-judgement call, bounded by SUPERSESSION_DAILY_CAP
    # (unchanged). Timur has accepted that volume increase as standing
    # production policy, not a value to revert later. See
    # reports/operational-values-decision.txt.
    extraction_dedup_similarity_threshold: float = Field(
        default=0.53, ge=-1.0, le=1.0, allow_inf_nan=False
    )

    # Per-guild, per-UTC-day ceiling on SUPERSESSION-JUDGMENT CALLS (Phase
    # 3a-3), the third daily cap in this file and a structural twin of the two
    # above it -- same append-only ledger, same guarded INSERT, same "claimed
    # before the call it authorizes, never refunded" rule (see
    # aura.db.supersession_state).
    #
    # It gets its own independent number rather than sharing
    # extraction_daily_cap, even though it can only fire downstream of a
    # distillation call that already spent one of those slots. Every paid call
    # site in this project carries its own cost safety net, and two call sites
    # sharing one budget would mean neither has a bound of its own: a burst of
    # dedup-flagged candidates would eat the extraction budget that produces
    # them, silently turning a judgment ceiling into an extraction outage.
    #
    # 50 is chosen against the same measured pricing as the two caps above. One
    # judgment call is small and fixed in size -- two sentences in, a category
    # and one sentence of reasoning out -- roughly 700 input and 80 output
    # tokens, about $0.001 at claude-haiku-4.5's $1/$5 per Mtok. 50 a day is
    # therefore a worst case near $1.50 per guild per month, and only if every
    # slot were spent every day, which the dedup threshold makes unlikely: this
    # call fires only for a candidate that cleared
    # EXTRACTION_DEDUP_SIMILARITY_THRESHOLD against an existing active fact
    # (deliberately not restated as a number here -- this comment previously
    # hardcoded "0.70" and silently went stale when that field was recalibrated
    # to 0.60 in this same phase, a drift caught and fixed on the VPS but never
    # fixed here; see reports/deployment-2026-07-31.txt's addendum and
    # reports/testing-threshold-note-2.txt), a small minority of what extraction
    # produces. When the cap
    # does bind, nothing breaks and nothing is lost -- the candidate is still
    # staged and still reviewed, it simply carries Phase 3a-2's plain similarity
    # hint instead of a judgment. 0 is valid and disables the judgment call
    # entirely while leaving the rest of extraction working.
    #
    # This cap's own per-guild worst case is unchanged by Phase 4a-2's
    # cross-guild budget below, which bounds the SUM across every guild
    # sharing one operator key rather than replacing this number (see
    # CLAUDE.md's Open Items).
    supersession_daily_cap: int = Field(default=50, ge=0, le=1_000_000)

    # --- Backfill over existing history (Phase 3b) --------------------------
    # Backfill runs the SAME already-hardened chain live extraction runs (first
    # filter, distillation, dedup hint, supersession proposal) over a channel's
    # EXISTING messages, on a moderator's explicit request. Nothing about how a
    # fact is recognised is configured here -- every threshold and model above
    # applies unchanged. What is configured here is the mechanics: how much it
    # may spend per day, and how fast it may ask Discord for history.

    # Per-guild, per-UTC-day ceiling on BACKFILL DISTILLATION CALLS -- the fifth
    # independent spend ledger in this file (see aura.db.backfill_state), and
    # deliberately NOT a share of extraction_daily_cap above.
    #
    # WHY IT MUST BE SEPARATE, since it is the same call against the same model:
    # a shared budget would let one moderator's backfill of a two-year channel
    # consume the entire day's allowance within minutes of being started, and
    # every message written in that guild that day would go unextracted. From
    # the outside that is indistinguishable from extraction having broken. It is
    # the same argument supersession_daily_cap already makes one call site
    # earlier -- two call sites sharing one number leaves neither with a bound of
    # its own -- with the addition that here the two call sites have genuinely
    # different shapes: bulk work over a fixed backlog, against a trickle over
    # live traffic.
    #
    # 30, and deliberately BELOW extraction's 50 rather than at or above it,
    # against the same measured pricing the three caps above use: a full
    # EXTRACTION_BATCH_MAX_MESSAGES batch costs roughly $0.011, so 30 calls is a
    # worst case near $0.33 per guild per day, and the run stops there and
    # resumes tomorrow rather than failing. Sizing it under extraction's cap is
    # the point of the number as much as the ceiling is: on a day when both are
    # saturated, the always-on mechanism keeps the larger share, because live
    # extraction has no second chance at a message and backfill has nothing but
    # second chances -- its input is Discord's own history, which is not going
    # anywhere.
    #
    # In coverage terms 30 is not tight: 30 batches of up to 20 fact-worthy
    # messages is up to 600 fact-worthy messages a day, and at the ~10%
    # fact-worthy rate reports/phase-3a-1b.txt measured by construction, roughly
    # 6,000 raw messages of history per guild per day. A small server's entire
    # backlog fits in one day; a large one takes a few, which is exactly the
    # multi-day run /aura-backfill status and pause exist to make manageable.
    #
    # 0 is valid and means "no backfill may spend anything", which stops every
    # run in its tracks without touching live extraction or losing a cursor.
    # This cap's own per-guild worst case is unchanged by Phase 4a-2's
    # cross-guild budget below, which bounds the SUM across every guild
    # sharing one operator key rather than replacing this number (see
    # CLAUDE.md's Open Items).
    backfill_daily_cap: int = Field(default=30, ge=0, le=1_000_000)

    # How long the backfill worker waits between two consecutive history page
    # requests for the same run.
    #
    # discord.py already handles Discord's rate limits underneath this -- it
    # reads the bucket headers, sleeps out a 429's Retry-After, and retries (see
    # aura.backfill.history for the layer Aura adds on top for the cases it
    # surfaces as exceptions instead). This value is not that. It is the
    # difference between a client that stops when it is told to and one that
    # never had to be told: one page is 100 messages, so a one-second pause caps
    # a backfill at 100 messages/second per run, orders of magnitude under any
    # bucket Discord enforces, while costing a 3,000-message channel about
    # thirty seconds it is in no hurry to save. Nobody is waiting on a backfill;
    # this is latency it does not need traded for a request rate it does.
    #
    # 0 is valid and means "as fast as discord.py allows", which is a reasonable
    # choice for a one-off run against a small private test server and a poor
    # one anywhere else.
    backfill_page_pause_seconds: float = Field(default=1.0, ge=0.0, le=60.0, allow_inf_nan=False)

    # How often the backfill worker wakes to look for runs when it has nothing
    # to do. NOT how fast a run progresses -- an active run advances batch after
    # batch without sleeping this long, pausing only backfill_page_pause_seconds
    # between pages (see aura.backfill.worker).
    #
    # The same indirection, for the same reason, that separates
    # digest_check_interval_seconds from a guild's digest interval: there is no
    # timer counting down to the next batch that a restart could reset, only a
    # stored cursor a tick compares against Discord's history. 30 seconds is how
    # long after /aura-backfill start (or after a daily cap resets at midnight)
    # a run takes to visibly begin moving, which is well inside "a moderator
    # ran a command and can see it working".
    #
    # Bounded at both ends like every other interval in this file: zero would
    # spin the worker task against the database in a tight loop, and a value in
    # days would make a paused-by-cap run resume most of a day after midnight.
    backfill_check_interval_seconds: float = Field(
        default=30.0, ge=1.0, le=24 * 60 * 60.0, allow_inf_nan=False
    )

    # --- Multi-representation indexing: variant generation (Part 1) --------
    # The paraphrase generator (see aura.variants_service): given one already-
    # active fact's canonical sentence, write several differently-worded
    # sentences that mean exactly the same thing, for later use as extra
    # embedding vectors over the same fact (Part 2, not this sub-phase).
    # Resolved through the same seam as every component above, falling back to
    # SYNTHESIS_MODEL when unset for the same reason EXTRACTION_MODEL does.
    #
    # SHIPPED VALUE: claude-haiku-4.5, and -- like EXTRACTION_MODEL -- this is
    # a CARRIED-OVER ASSUMPTION, not a bake-off of its own, stated plainly for
    # the same reason. What makes that acceptable here specifically: this call
    # is pure rewording with no asymmetric-cost judgement attached to it, the
    # way supersession's "which reading is safe to guess" has. Correctness is
    # enforced downstream by VARIANT_AUDIT_MODEL, not by this model's own
    # judgement, so a first choice that transfers a measured trait (strong
    # structured output, multilingual competence) is a reasonable starting
    # point rather than a gap. Revisit with a real bake-off if the audit below
    # shows this model systematically dropping qualifiers or generalising
    # scope -- see reports/variant-indexing-part1.txt.
    variant_model: str | None = None

    # The independent fidelity check every generated variant must pass before
    # it is stored (see aura.variants_service): a SEPARATE model from a
    # DIFFERENT vendor than VARIANT_MODEL, verifying a variant preserves the
    # canonical fact's meaning exactly -- no dropped exception or qualifier, no
    # scope over-generalisation (a channel-specific rule reworded as if it
    # applied everywhere). Same principle as the label audit in
    # reports/phase-3a-1b.txt: a model does not reliably catch its own
    # meaning-drift, so a second, differently-trained model checks it instead.
    #
    # DELIBERATELY has NO fallback to synthesis_model, unlike every other model
    # field in this file. Falling back would defeat the one property this
    # field exists to guarantee: an unconfigured deployment would silently
    # have its "independent" audit collapse onto the same model family as the
    # generator (both resolving to synthesis_model), auditing its own output
    # under the appearance of independence. Left unset, no variant is ever
    # stored rather than one stored on a silently-compromised check -- the
    # same "no grey areas" reasoning CLAUDE.md states as this project's
    # non-negotiable principle, applied to a config default instead of a code
    # path. See resolve_model's VARIANT_AUDIT arm for where this is enforced.
    variant_audit_model: str | None = None

    # How many paraphrased variants to attempt generating per newly active
    # fact. Not a promise: the independent fidelity audit above can and does
    # reject some, and this sub-phase deliberately does not regenerate to make
    # up the shortfall (see reports/variant-indexing-part1.txt) -- a smaller
    # stored set is an accepted, documented outcome, not a bug.
    #
    # 6 sits in the middle of the phase brief's 5-8 range: enough that a later
    # similarity search taking the maximum score over all of a fact's variants
    # gets real coverage benefit over a single canonical embedding, without
    # turning one fact into an unbounded generation-plus-audit prompt.
    variant_count: int = Field(default=6, ge=1, le=20)

    # Per-guild, per-UTC-day ceiling on variant-generation EPISODES (one
    # fact's generation call plus its audit call, always spent together -- see
    # aura.variants_service), the fourth independent spend ledger in this
    # project, built as the same durable, race-safe twin as the three above.
    #
    # Sized differently from the other three on purpose. Facts are created at
    # HUMAN speed, not message speed: every fact requires either a moderator
    # clicking "Add as Aura Fact" or a moderator confirming a candidate
    # through /aura-pending, one at a time (see aura.commands.pending's
    # module docstring, "deliberately one candidate at a time"). There is no
    # path that creates facts anywhere near the volume extraction or
    # proactive relief see, so the natural ceiling on how often this can fire
    # is already a human's click rate, not a message flood. A cap is still
    # added -- every paid call site in this project carries its own
    # independent cost safety net, per the pattern set by the three ledgers
    # above -- but it is set generously rather than swept from a corpus, since
    # there is no realistic scenario at today's usage where it binds. This
    # cap's own per-guild worst case is unchanged by Phase 4a-2's cross-guild
    # budget below, which bounds the SUM across every guild sharing one
    # operator key rather than replacing this number (see CLAUDE.md's Open
    # Items).
    variant_daily_cap: int = Field(default=200, ge=0, le=1_000_000)

    # --- /aura-ask cost bounds ------------------------------------------------
    # The sixth ledger (aura.db.ask_state). Until it existed /aura-ask was the
    # one paid call site with no daily bound: anyone could ask, and the only
    # throttle was a 30-second per-user cooldown. One slot is one paid answer
    # (a synthesis call plus its grounding check), claimed only when a question
    # matched at least one fact and synthesis is about to run -- a question that
    # matches nothing costs nothing and claims nothing.
    #
    # WHICH CAP APPLIES is decided by the plan gate at the moment of the call
    # (aura.billing.PlanGate.allows_pro), the same single seam every Pro trigger
    # asks. So BILLING_MODE=disabled and every complimentary guild get the Pro
    # cap, and a guild whose plan changes mid-day is measured against its new
    # cap at once while the answers it already spent today keep counting.
    #
    # WHEN A CAP IS REACHED the question is not refused: the asker gets, visible
    # only to them, a note that today's AI answers are used up and up to three
    # of the facts retrieval already found, with their sources and dates -- no
    # model call, no slot, no cost (see aura.commands.ask).
    #
    # 0 is valid for every one of the three and means "never use the model for
    # this plan": every matched question gets that free answer instead.
    #
    # Sized from the measured cost (reports/quality-diagnosis-2026-10-02.md,
    # Section 6, which is private): about $0.002 for a typical paid answer and
    # at most about $0.011 with the output and fact bounds below. Free at 10 a
    # day is at most ~$0.11 per guild per day; the per-member share of 5 keeps
    # one member from spending a whole server's allowance alone. Pro has the
    # guild cap only -- a paying server decides for itself who asks.
    ask_daily_cap_free: int = Field(default=10, ge=0, le=1_000_000)
    ask_daily_cap_pro: int = Field(default=25, ge=0, le=1_000_000)
    ask_user_daily_cap_free: int = Field(default=5, ge=0, le=1_000_000)

    # The output ceiling (max_tokens) on every synthesis call -- the shared
    # function behind /aura-ask AND proactive relief, so it bounds both. Before
    # it existed nothing did: the model's own limit is tens of thousands of
    # tokens, and a prompt-injected "write a very long answer" ran until the
    # 30-second timeout. A typical answer measured about 100 output tokens, so
    # 700 is a wide margin for an answer in any of the nine locales (Japanese
    # and Korean spend more tokens per sentence) while capping the worst case.
    # An answer cut off by this limit is treated exactly like any unparsable
    # response: not sent (see aura.synthesis). The lower bound keeps a
    # misconfiguration from silencing every answer.
    ask_synthesis_max_output_tokens: int = Field(default=700, ge=256, le=8192)

    # --- The answer format (P4) ----------------------------------------------
    # Which format each answering trigger uses: ANSWER_FORMAT for /aura-ask,
    # PROACTIVE_ANSWER_FORMAT for proactive relief. Two settings, not one,
    # because the proactive pipeline is calibrated on the legacy format and
    # switches only after a calibration of its own -- /aura-ask can move first.
    # LEGACY is the default for both, and with it every prompt and every
    # message is byte for byte what it was before these settings existed. V2
    # ships dark (see AnswerFormat) and needs a checker model: see
    # _the_new_format_needs_its_checker below.
    answer_format: AnswerFormat = AnswerFormat.LEGACY
    proactive_answer_format: AnswerFormat = AnswerFormat.LEGACY

    # How a v2 answer card is drawn (see CardStyle). Read only on the v2 path.
    answer_card_style: CardStyle = CardStyle.EMBED

    # The model that writes v2 answers for /aura-ask, resolved through
    # resolve_model (ModelComponent.ANSWER_V2). Its own value so the v2 answer
    # can move to a different model than the legacy one without touching
    # either trigger; unset, it falls back to SYNTHESIS_MODEL, so switching the
    # format alone never switches the model. Proactive relief's v2 path keeps
    # PROACTIVE_MODEL. Per-plan models (a stronger one for a higher tier) are
    # not wired here; resolve_model is the seam they will hook into.
    answer_v2_model: str | None = None

    # The model that checks v2 answers point by point (aura.answer_check), for
    # both triggers. Unset, it falls back to GROUNDING_CHECK_MODEL -- never to
    # a synthesis model, for the independence reason grounding_check_model
    # documents below.
    answer_v2_check_model: str | None = None

    # The output ceiling (max_tokens) of one v2 answer call, for both triggers.
    # Larger than ASK_SYNTHESIS_MAX_OUTPUT_TOKENS because the contract writes a
    # short analysis (a note per fact, the relations) before the answer. A
    # reply cut off at it is unusable, exactly like a legacy one.
    answer_v2_max_output_tokens: int = Field(default=1000, ge=256, le=8192)

    # The output ceiling (max_tokens) of one v2 check. A verdict cut off at it
    # fails closed, like any unparsable one.
    answer_v2_check_max_output_tokens: int = Field(default=600, ge=128, le=4096)

    # OpenRouter request options for the two v2 calls (aura.llm_request_options),
    # so production can send a model the same route it was measured on: the
    # providers to pin (comma-separated, in order, no fallback), the reasoning
    # level ("" = the model's default, "off", "low", "medium", "high"), and
    # whether to use only providers that neither retain nor train on the data.
    # All unset by default: then the calls carry no extra fields at all. Ignored
    # for a model not routed through OpenRouter. The ANSWER_V2_* route describes
    # ANSWER_V2_MODEL and goes only with /aura-ask's answer, never with proactive
    # relief's PROACTIVE_MODEL (providers pinned for one model may not serve
    # another); the check route and the data policy go with every v2 check.
    answer_v2_providers: str = ""
    answer_v2_reasoning: Literal["", "off", "low", "medium", "high"] = ""
    answer_v2_check_providers: str = ""
    answer_v2_check_reasoning: Literal["", "off", "low", "medium", "high"] = ""
    answer_v2_deny_data_collection: bool = False

    # --- The background functions: output ceilings and routes (P5) -----------
    # The output ceiling (max_tokens) of one fact-extraction call, live and
    # backfill alike (aura.extraction.distiller). Until P5 the call carried
    # none, so a misbehaving model or provider could run until the timeout.
    # A full batch is EXTRACTION_BATCH_MAX_MESSAGES messages, each of which can
    # yield a sentence of up to 500 characters plus its JSON fields -- about
    # 190 tokens -- so a legitimate reply to a 20-message batch stays under
    # 4,000; the P5 evaluation measured the replies actually written (see the
    # private P5 report). A reply cut off at this ceiling is treated exactly
    # like an unparsable one: nothing from it is staged and the batch takes the
    # existing failure path.
    extraction_max_output_tokens: int = Field(default=4096, ge=1024, le=16384)

    # The output ceiling of one supersession judgement
    # (aura.extraction.supersession): four short fields and one sentence of at
    # most 600 characters. A cut-off judgement is "not judged", the existing
    # failure path; the candidate keeps its plain similarity hint.
    supersession_max_output_tokens: int = Field(default=1024, ge=256, le=8192)

    # The output ceilings of the two variant calls (aura.variants_service),
    # which ship inactive (VARIANT_AUDIT_MODEL unset): a list of at most
    # VARIANT_COUNT short sentences, and one verdict per variant. A cut-off
    # reply stores no variants, the existing failure path.
    variant_max_output_tokens: int = Field(default=1024, ge=256, le=8192)
    variant_audit_max_output_tokens: int = Field(default=1024, ge=256, le=8192)

    # OpenRouter request options for fact extraction, the supersession judge
    # and proactive relief (aura.llm_request_options), in the same shape as the
    # ANSWER_V2_* ones above: the providers to pin (comma-separated, in order,
    # no fallback), the reasoning level ("" = the model's default, "off",
    # "low", "medium", "high"), and whether to use only providers that neither
    # retain nor train on the data. Each line describes its own function's
    # model and goes with that function's calls only. All unset by default:
    # then the calls carry no extra fields at all, exactly as before P5.
    # EXTRACTION_* goes with live extraction, backfill and the extraction
    # verification below; PROACTIVE_* with proactive relief's answer in either
    # format (its v2 check keeps the ANSWER_V2_CHECK_* route).
    extraction_providers: str = ""
    extraction_reasoning: Literal["", "off", "low", "medium", "high"] = ""
    extraction_deny_data_collection: bool = False
    supersession_providers: str = ""
    supersession_reasoning: Literal["", "off", "low", "medium", "high"] = ""
    supersession_deny_data_collection: bool = False
    proactive_providers: str = ""
    proactive_reasoning: Literal["", "off", "low", "medium", "high"] = ""
    proactive_deny_data_collection: bool = False

    # The output ceiling of proactive relief's answer in the v2 format. Unset
    # (the default), it is ANSWER_V2_MAX_OUTPUT_TOKENS, exactly as before P5.
    # Its own value because a model that reasons before it answers spends its
    # reasoning tokens against this ceiling: the P5 evaluation measured
    # DeepSeek V4.1 Flash with reasoning on at up to about 2,700 tokens for one
    # proactive answer, so it needs a higher ceiling than /aura-ask's
    # non-reasoning answer, which stays bounded at its own value. A reply cut
    # off at it is unusable, and proactive relief stays silent.
    proactive_max_output_tokens: int | None = Field(default=None, ge=256, le=16384)

    # The deadline of proactive relief's answer call, in either format (P5c).
    # UNSET (the default), the call runs exactly as before: it passes the 30
    # seconds every answer call has always passed to the HTTP client -- which,
    # measured in P5c, is a limit per read, not on the whole call: OpenRouter
    # keeps a slow non-streaming request alive, so a DeepSeek call with that
    # timeout still answered after 43 seconds in P5. SET, the value is passed
    # to the client and also enforced as a hard deadline around the call
    # (asyncio.wait_for, as the answer checks already do): an answer not ready
    # in time is silence, never a late post. Measured for DeepSeek V4.1 Flash
    # with reasoning on (P5, 1,026 proactive calls at a harness concurrency of
    # six): p50 9.9 s, p95 25.1 s, p99 36.5 s, max 43.4 s -- 60 covers all of
    # them. /aura-ask never reads this; its user is waiting, and it keeps its own
    # 30 seconds. Whatever this says, a proactive answer is posted only while the
    # conversation has not moved on (aura.proactive.grace.AnswerWatch).
    proactive_request_timeout_seconds: float | None = Field(
        default=None, ge=5.0, le=300.0, allow_inf_nan=False
    )

    # The model of the extraction verification (aura.extraction.verifier): a
    # second call that reads every distilled candidate against the batch it
    # came from and drops any the messages do not support -- a joke stored as
    # a rule, a dropped "nur", an unresolved "morgen", a value a later message
    # corrects. UNSET (the default) means no verification: extraction runs
    # exactly as before P5. Deliberately no fallback, for the reason
    # grounding_check_model gives: a check that silently fell back to the
    # extraction model would check a model's output with the same model.
    # Its route is EXTRACTION_VERIFY_PROVIDERS / EXTRACTION_VERIFY_REASONING,
    # and EXTRACTION_DENY_DATA_COLLECTION applies to it too.
    extraction_verify_model: str | None = None
    extraction_verify_providers: str = ""
    extraction_verify_reasoning: Literal["", "off", "low", "medium", "high"] = ""
    extraction_verify_max_output_tokens: int = Field(default=2048, ge=512, le=8192)

    # A batch whose DISTILLATION CALL (P5c) or VERIFICATION CALL (P5) failed
    # for a reason outside the batch (a timeout, a provider or network error,
    # a refused key; aura.llm_failures decides -- never the model judging the
    # batch, never an unusable reply and never a request the provider refuses,
    # none of which is retried) is held and tried again instead of cleared
    # (aura.extraction.verify_retry): up to EXTRACTION_VERIFY_MAX_ATTEMPTS
    # attempts in total, both calls counting against the same attempts, the
    # first pause EXTRACTION_VERIFY_RETRY_DELAY_SECONDS, doubling after each
    # failure. The defaults give 4 attempts over about 70 minutes (10, 20, 40
    # minutes), which outlasts a typical provider outage while bounding both
    # the time a batch's raw text is held and the slots one batch can spend
    # (one per attempt, 4 of EXTRACTION_DAILY_CAP's 50). After the last attempt
    # the batch is given up with an ERROR log line. The names predate P5c and
    # are kept so no deployed .env breaks; they apply whether or not
    # EXTRACTION_VERIFY_MODEL is set.
    extraction_verify_max_attempts: int = Field(default=4, ge=1, le=10)
    extraction_verify_retry_delay_seconds: float = Field(default=600.0, ge=1.0, le=86400.0)

    # --- Message looks (P5) ---------------------------------------------------
    # The card look of each message family (aura.cards), switched separately;
    # see MessageLook. CLASSIC, the default, sends every message byte for byte
    # as before P5. NOTICE_LOOK covers the command replies that have a card
    # (setting confirmations, the fact confirmations, the Pro-only refusal);
    # PLAN_LOOK covers /aura-plan itself. A card is drawn in ANSWER_CARD_STYLE.
    digest_look: MessageLook = MessageLook.CLASSIC
    onboarding_look: MessageLook = MessageLook.CLASSIC
    plan_look: MessageLook = MessageLook.CLASSIC
    notice_look: MessageLook = MessageLook.CLASSIC

    # --- Cross-guild operator budget (Phase 4a-2) ---------------------------
    # CLAUDE.md's Open Items section named this gap before any code existed for
    # it, and every one of the five daily-cap comments above already points
    # here ("revisit alongside the other N before any multi-guild rollout"):
    # PROACTIVE_DAILY_CAP, EXTRACTION_DAILY_CAP, SUPERSESSION_DAILY_CAP,
    # VARIANT_DAILY_CAP and BACKFILL_DAILY_CAP each bound one guild's worst
    # case, which says nothing about the SUM across every guild sharing one
    # operator-funded key. See aura.db.cross_guild_budget for the mechanism
    # this pair of settings drives, and reports/phase-4a-multitenancy-audit.txt
    # Section 3 for the gap as it stood before this phase closed it.
    #
    # Timur chose a flat, per-guild subscription price over metered billing
    # (Option A, that audit's Section 4), which is why this is ONE combined
    # ceiling across all five ledgers rather than five more per-ledger caps: a
    # flat-fee operator's real question is "is total spend still inside what
    # subscription revenue covers", not "which ledger, on which guild, spent
    # what" -- the five existing per-guild caps already answer that second
    # question, independently, the way CLAUDE.md's "every paid call site
    # carries its own safety net" principle asks them to.
    cross_guild_budget_mode: CrossGuildBudgetMode = CrossGuildBudgetMode.WARN

    # A rough, conservative dollar ceiling on TODAY's combined estimated spend
    # across all five ledgers and every guild sharing this deployment's key --
    # not a real-time cost meter (see aura.db.cross_guild_budget for the fixed,
    # documented worst-case per-call figures this reuses from each cap's own
    # comment rather than measuring real tokens or dollars).
    #
    # 15.00 is sized against the same worst-case-at-full-cap-utilization math
    # every per-guild cap above already uses, summed per guild and then scaled
    # to the "handful of test and community servers" CLAUDE.md's Proactive
    # Relief section names as Aura's realistic near-term footprint: roughly
    # $0.18/day proactive (60 * $0.003) + $0.55/day extraction (50 * $0.011) +
    # $0.05/day supersession (50 * $0.001) + $0.40/day variant (200 * $0.002)
    # + $0.33/day backfill (30 * $0.011) = ~$1.51/guild/day if every ledger's
    # cap were fully saturated every single day, which the individual caps'
    # own comments already call an extreme rather than a realistic case. (The
    # sixth ledger, /aura-ask, joined later and adds at most $0.10/day per Pro
    # guild, 25 * $0.004 -- not enough to move this default.) $15
    # covers roughly ten such guilds at that extreme simultaneously -- a
    # deliberately generous multiple of "a handful" rather than a tight fit,
    # since this ceiling's job is catching a genuine runaway (a bug, or far
    # more paying guilds than expected), not policing normal variance.
    #
    # MUST be revisited before any rollout past a handful of guilds: this
    # default does not scale itself, and CLAUDE.md's Open Items note on this
    # exact gap already said so before this field existed.
    cross_guild_daily_budget_usd: float = Field(default=15.0, ge=0.0, allow_inf_nan=False)

    # The Discord user ID allowed to run /aura-operator-budget (see
    # aura.commands.operator) -- the one command that shows the cross-guild
    # totals above are actually measuring, across every guild this process
    # serves. Deliberately a single configured ID rather than Discord's own
    # application-owner/team concept: this project runs as a single
    # self-hosted process for a single operator, not a published multi-team
    # app, and comparing against a plain configured ID needs no extra API call
    # and no team-vs-solo-owner branching to get subtly wrong. Left unset by
    # default, which means the command refuses everyone -- a deployment that
    # never sets this loses only a diagnostic view, never budget enforcement
    # itself, which runs unconditionally through cross_guild_budget_mode
    # above regardless of whether anyone can see the numbers.
    operator_discord_user_id: int | None = None

    # --- The independent grounding check on every answer Aura sends ---------
    # The last step before /aura-ask replies or the proactive responder posts:
    # a second model reads the finished answer against the facts it cited and
    # decides whether they actually support it (see aura.grounding). It never
    # rewrites anything -- it votes yes or no, and a no is silence.
    #
    # DELIBERATELY has NO fallback to synthesis_model, the second field in this
    # file to make that choice and for the same reason variant_audit_model does:
    # falling back would let an unconfigured deployment silently check the
    # synthesis model's output with the synthesis model, which is not a check.
    # An independent check that quietly stopped being independent is worse than
    # no check, because it reports the same "verified" either way.
    #
    # SHIPPED VALUE: openrouter/openai/gpt-4o-mini, a DIFFERENT VENDOR from the
    # Anthropic model this project ships for SYNTHESIS_MODEL and PROACTIVE_MODEL.
    # Like extraction_model and variant_model, this is a CARRIED-OVER CHOICE
    # rather than a bake-off of its own, and -- as with those two -- that is
    # stated plainly because two other models in this file WERE chosen by
    # measurement and a reader would otherwise reasonably assume this one was.
    # What it is carried from: the fidelity audit in aura.variants_service uses
    # this exact model for a structurally identical job (an independent,
    # differently-trained model checking whether one piece of generated text
    # says only what a source text says), and reports/phase-3a-1b.txt's label
    # audit used it as this project's standing independent reviewer before that.
    # The transfer is closer here than extraction_model's was: same task shape,
    # same "no dropped or added qualifiers" question, same strict JSON, and the
    # answer under check is short.
    #
    # Where the transfer is weakest, since a carried assumption should say where
    # it might break: the variant audit compares two SENTENCES in the same
    # language, while this compares a whole multi-sentence answer -- written in
    # the asker's locale -- against several facts that may be written in a
    # different one. reports/grounding-check.txt measures exactly that on real
    # calls, including a German case, rather than assuming it transfers.
    #
    # WHEN UNSET, THE CHECK DOES NOT RUN AND ANSWERS ARE SENT AS BEFORE. This is
    # the one place where this feature is deliberately NOT fail-closed, and the
    # reasoning is the same one that keeps proactive_confidence_gap in this file
    # long after it stopped doing anything: Aura runs live on a VPS, and a field
    # whose absence silences the bot entirely would turn a routine `git pull` into
    # an outage whose only explanation is that no answer ever arrives. An absent
    # model is an operator decision not to run the check; a FAILING check is a
    # different thing entirely and does fail closed (see aura.grounding). To keep
    # that from being a silent grey area, every send without the check logs a
    # warning naming this variable.
    grounding_check_model: str | None = None

    # The output ceiling (max_tokens) on every grounding check, on both send
    # paths. The verdict is four booleans and one sentence; a check measured
    # about 75 output tokens, and a rejection that describes all three findings
    # stays well under 300. A response cut off by this limit fails closed like
    # any unparsable one -- the answer is not sent (see aura.grounding). The
    # lower bound keeps a misconfiguration from failing every check.
    grounding_max_output_tokens: int = Field(default=300, ge=128, le=4096)

    # --- The periodic digest (CLAUDE.md's fourth trigger) -------------------
    # How often the background scheduler wakes to ask which guilds are due for a
    # digest. NOT how often a digest is posted -- that is per-guild state a
    # moderator picks with /aura-digest (see aura.digest.intervals), and this
    # value only bounds how late a due digest can be.
    #
    # Hourly is chosen against what being wrong in either direction costs, since
    # nothing here is calibrated against data and nothing needs to be. Too slow
    # and a digest configured for "daily" could arrive up to an hour off, which
    # nobody notices in a weekly or daily summary. Too fast and the deployment
    # pays a handful of indexed reads per guild for nothing -- there is no LLM
    # call and no API call on a tick where nothing is due, so the floor is not a
    # cost problem but a pointlessness one. An hour sits comfortably between:
    # 1/24 of the shortest offered cadence, and 24 wake-ups a day.
    #
    # Bounded at both ends for the same reason the proactive cooldown is: the
    # value is used in arithmetic and in a sleep, and a deployment that sets it
    # to zero would spin the scheduler task in a tight loop against the
    # database, while one that sets it to a month would make the shortest digest
    # interval meaningless.
    digest_check_interval_seconds: float = Field(
        default=3600.0, ge=1.0, le=24 * 60 * 60.0, allow_inf_nan=False
    )

    # --- Onboarding (CLAUDE.md's third trigger) ------------------------------
    # The TOTAL number of facts one member's onboarding message may list,
    # spent in priority order across all three sections (rules, then status
    # changes, then everything else -- see aura.onboarding.builder). A single
    # total rather than a per-section cap, because the product decision this
    # sub-phase makes is about the size of the WHOLE message a brand-new
    # member is handed, not about how much room any one heading gets: three
    # sections independently capped at the digest's per-field value of 10
    # would let a message reach 30 items, which defeats onboarding's own
    # purpose of orienting someone with zero context rather than overwhelming
    # them on arrival.
    #
    # 15 is chosen as comfortably more than a typical small server's rule set
    # (rarely more than a handful) plus enough current-status items to be
    # useful, while staying well short of "wall of text". A server that
    # outgrows it is exactly the case reports/phase-3d.txt names as the point
    # where plain category filtering stops being enough and real relevance
    # judgment (an LLM call) would be justified -- deliberately out of scope
    # for this sub-phase, and revisit this default if that line is crossed.
    onboarding_fact_limit: int = Field(default=15, ge=1, le=100)

    # Per-guild, per-UTC-day ceiling on onboarding MESSAGES SENT, mirroring the
    # shape of proactive_daily_cap and extraction_daily_cap (aura.db.
    # onboarding_state) but bounding something different: this is not a spend
    # control (there is no LLM call in the base case, see aura.onboarding.
    # builder) but a channel-flood control, the direct answer to what happens
    # on a mass-join event -- a raid, a bot pile-on, a partnership's invite
    # spike -- where every arriving member would otherwise get their own
    # embed in the same channel within seconds of each other.
    #
    # 20 is generous for organic growth (a small server rarely gains 20
    # genuine members in a single day) while still bounding an unbounded pile
    # of joins to a fixed, small number of public posts. Members who join
    # after the cap is reached get no onboarding message for that guild that
    # day; there is no catch-up mechanism (see aura.onboarding.listener) since
    # onboarding has no periodic sweep to retry from, unlike the digest.
    onboarding_daily_cap: int = Field(default=20, ge=0, le=1_000_000)

    # --- Plans and billing (Phase 4c) ---------------------------------------
    # Whether the Pro-only triggers are gated on a subscription at all. See
    # BillingMode for what each value does, and aura.billing for the rules.
    #
    # DISABLED by default, and the reason is the live VPS rather than billing
    # itself: Aura already runs for real guilds, and a routine
    # `git pull && docker compose up` must not quietly move every one of them
    # to Free the moment this code lands. Turning enforcement on is a decision
    # with consequences for real servers, so it is something an operator sets
    # on purpose, never something they inherit. The default also fails in the
    # recoverable direction: forgetting to enable enforcement gives features
    # away for a while; inheriting it by accident takes them away from guilds
    # that did nothing wrong.
    billing_mode: BillingMode = BillingMode.DISABLED

    # How long Pro survives past a subscription's paid-through date while the
    # renewal has not yet been CONFIRMED to Aura. This is the bounded grace for
    # "subscription status cannot be established right now": it extends trust
    # in the last known good state by a fixed amount past what was actually
    # paid for, and never further.
    #
    # 72 hours is Stripe's own delivery horizon, not a round number. At the end
    # of a period Stripe creates the renewal invoice, holds it as a draft for
    # about an hour, then charges it, and the confirmation reaches Aura as a
    # webhook after that. If the web backend or this bot is down at that
    # moment, Stripe retries the delivery for up to three days in live mode.
    # Shorter than that would move a paying guild to Free during an outage
    # Stripe itself would still heal; longer buys nothing, because an event
    # Stripe has stopped retrying is recovered by the web backend's periodic
    # reconciliation (aura_web.billing_sync), not by waiting longer here.
    #
    # Not applied to a subscription that is already set to end: "cancels at
    # the end of this period" is a known end date, so Pro ends exactly there.
    billing_renewal_grace_hours: float = Field(
        default=72.0, ge=0.0, le=30 * 24.0, allow_inf_nan=False
    )

    # How long Pro survives a failed renewal payment (subscription past_due)
    # before the guild moves to Free.
    #
    # Cards fail for harmless reasons -- an expired card, a bank's fraud
    # heuristic, a temporary limit -- and Stripe keeps retrying the charge on
    # its own schedule (its recommended default is 8 attempts within 2 weeks)
    # while emailing the payer. Seven days covers a weekend plus a working week
    # for a server admin who does not read billing mail every day, which is the
    # realistic person on the other end of this, while keeping unpaid Pro
    # bounded to a quarter of a monthly period for each payment that lapses.
    #
    # Anchored at the START of the OLDEST period still unpaid, never at "the
    # first failure Aura heard about" and never at the current period's start:
    # a retry that fails again cannot extend it, a webhook that arrives late
    # cannot restart it, and neither can Stripe rolling a still-unpaid
    # subscription into its next period (which it does every cycle while the
    # subscription stays past_due) or writing the unpaid invoice off. Only a
    # paid period resets it (aura.billing.entitlement.next_unpaid_since). A
    # cancellation date inside the grace still ends Pro on that date. Stripe's
    # own final decision -- canceled or unpaid after its last retry -- ends Pro
    # immediately regardless of this number.
    billing_payment_grace_days: float = Field(default=7.0, ge=0.0, le=60.0, allow_inf_nan=False)

    # Comma-separated guild IDs that are on Pro without any subscription: the
    # operator's own test servers, or community servers they choose to support.
    # This is the rollout valve for BILLING_MODE=enforced -- without it,
    # switching enforcement on would move the operator's own servers to Free
    # together with everyone else's. Configuration rather than a command
    # because granting Pro for free is a decision about the operator's money,
    # and this project's operator decisions live in configuration (see
    # OPERATOR_DISCORD_USER_ID above).
    billing_complimentary_guild_ids: str = ""

    # Where a moderator can subscribe or manage billing (the web dashboard).
    # Shown in /aura-plan and in every "this is a Pro feature" refusal.
    # Optional: a refusal without a link is still a clear refusal.
    billing_dashboard_url: str | None = None

    # The shared secret the web backend presents to this process's internal
    # billing API (aura.billing.internal_api) -- the ONLY path by which
    # subscription state changes while the bot is running. Unset means the API
    # is not started at all. Required when BILLING_MODE=enforced: an enforced
    # deployment whose subscription state can never change would move every
    # paying guild to Free the day its first period ends.
    internal_api_secret: SecretStr | None = None

    # Where the internal billing API listens. 127.0.0.1 by default, so a process
    # that was never configured for billing never listens on a reachable
    # interface; the compose file sets 0.0.0.0 inside the container, where only
    # the dedicated internal network can reach it and no host port is published.
    internal_api_host: str = "127.0.0.1"
    internal_api_port: int = Field(default=8081, ge=1, le=65535)

    log_level: str = "INFO"

    @field_validator("discord_token")
    @classmethod
    def _require_non_blank_token(cls, value: SecretStr) -> SecretStr:
        stripped = value.get_secret_value().strip()
        if not stripped:
            raise ValueError(f"DISCORD_TOKEN is missing or blank. {ENV_EXAMPLE_HINT}")
        return SecretStr(stripped)

    @field_validator("operator_discord_user_id", mode="before")
    @classmethod
    def _blank_operator_id_means_unset(cls, value: object) -> object:
        """Treat a blank OPERATOR_DISCORD_USER_ID the same as an absent one.

        Every other optional field in this file is a `str | None`, where a
        blank env value parses harmlessly to `""` (falsy, read the same as
        unset by every caller). This is the first optional `int | None`
        field, and pydantic does NOT extend that same courtesy to numbers: an
        empty string fails int coercion outright, so `OPERATOR_DISCORD_USER_ID=`
        left blank in .env -- exactly what .env.example ships, and exactly what
        an operator clearing a previously-set ID by blanking rather than
        deleting the line would produce -- would otherwise crash the whole
        process at startup on a field that gates one diagnostic command. That
        is precisely the "routine `git pull` becomes an outage" failure mode
        CLAUDE.md's grounding_check_model comment already names for a
        different field; this validator is the same fix, applied here before
        pydantic's own int parsing ever sees the value.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("internal_api_secret", "billing_dashboard_url", mode="before")
    @classmethod
    def _blank_optional_billing_string_means_unset(cls, value: object) -> object:
        """Treat `INTERNAL_API_SECRET=` and `BILLING_DASHBOARD_URL=` left blank as unset.

        .env.example ships both lines blank. Without this, a blank secret
        would reach the length check below and crash startup for a deployment
        that never asked for billing at all -- the same "routine git pull
        becomes an outage" shape _blank_operator_id_means_unset exists for.
        """
        if isinstance(value, SecretStr):
            stripped_secret = value.get_secret_value().strip()
            return SecretStr(stripped_secret) if stripped_secret else None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("internal_api_secret")
    @classmethod
    def _internal_api_secret_is_strong_and_header_safe(
        cls, value: SecretStr | None
    ) -> SecretStr | None:
        """Apply require_strong_internal_api_secret to a configured secret."""
        if value is None:
            return None
        require_strong_internal_api_secret(value.get_secret_value())
        return value

    @field_validator("billing_dashboard_url")
    @classmethod
    def _dashboard_url_is_absolute(cls, value: str | None) -> str | None:
        """Reject a dashboard URL Discord would render as plain, unclickable text."""
        if value is None:
            return None
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"BILLING_DASHBOARD_URL must be an absolute http(s) URL, got {value!r}."
            )
        return value

    @field_validator("billing_complimentary_guild_ids")
    @classmethod
    def _complimentary_ids_are_guild_ids(cls, value: str) -> str:
        """Validate the comma-separated list once, at startup, and normalise it.

        A typo here is not harmless: an ID that silently fails to parse is a
        server the operator believes is on Pro and is not. So every entry must
        be a plain positive decimal ID that fits the database's integer type,
        and anything else refuses to start with the offending entry named.
        Empty entries (a trailing comma) are ignored, since they carry no
        intent to be wrong about.
        """
        normalised: list[str] = []
        for raw_entry in value.split(","):
            entry = raw_entry.strip()
            if not entry:
                continue
            if (
                not (entry.isascii() and entry.isdigit())
                or int(entry) <= 0
                or int(entry) > MAX_SQLITE_INTEGER
            ):
                raise ValueError(
                    f"BILLING_COMPLIMENTARY_GUILD_IDS contains {entry!r}, which is not a Discord guild ID."
                )
            normalised.append(str(int(entry)))
        return ",".join(normalised)

    @field_validator("internal_api_host")
    @classmethod
    def _internal_api_host_is_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("INTERNAL_API_HOST must not be blank.")
        return stripped

    @model_validator(mode="after")
    def _enforced_billing_needs_the_internal_api(self) -> Settings:
        """Refuse BILLING_MODE=enforced without a way for subscription state to change.

        Enforcement reads subscription state that only the internal billing
        API can write. Enforcing with the API switched off would start a bot
        that looks healthy while every guild -- paying or not -- is on Free and
        can never leave it, which is the "paying guild wrongly locked out"
        failure this sub-phase exists to prevent. Failing at startup with the
        reason named is the honest version of that.
        """
        if self.billing_mode is BillingMode.ENFORCED and self.internal_api_secret is None:
            raise ValueError(
                "BILLING_MODE=enforced requires INTERNAL_API_SECRET, otherwise no "
                "subscription could ever reach this process."
            )
        return self

    @model_validator(mode="after")
    def _the_new_format_needs_its_checker(self) -> Settings:
        """Refuse the v2 answer format on a trigger when no model can check it.

        The legacy format sends an answer without a check when
        GROUNDING_CHECK_MODEL is unset (logged at every send), so that a missing
        setting can never silence a running bot. The v2 format is chosen
        deliberately by an operator, and it must not go live unchecked: so
        selecting it without ANSWER_V2_CHECK_MODEL or GROUNDING_CHECK_MODEL
        fails at startup with the reason named, instead of sending unverified
        answers.
        """
        selected = [
            name
            for name, value in (
                ("ANSWER_FORMAT", self.answer_format),
                ("PROACTIVE_ANSWER_FORMAT", self.proactive_answer_format),
            )
            if value is AnswerFormat.V2
        ]
        if selected and self.resolve_model(ModelComponent.ANSWER_V2_CHECK) is None:
            raise ValueError(
                f"{' and '.join(selected)}=v2 requires ANSWER_V2_CHECK_MODEL or "
                "GROUNDING_CHECK_MODEL: a v2 answer is never sent unchecked."
            )
        return self

    @property
    def complimentary_guild_ids(self) -> frozenset[int]:
        """Return the operator's complimentary Pro guilds.

        Returns
        -------
        frozenset[int]
            Guild IDs parsed from the validated comma-separated setting; empty when
            none are configured. Validation has already happened on the field, so
            the parse cannot fail here.
        """
        if not self.billing_complimentary_guild_ids:
            return frozenset()
        return frozenset(int(entry) for entry in self.billing_complimentary_guild_ids.split(","))

    def resolve_model(self, component: ModelComponent) -> str | None:
        """Resolve the model a given LLM-calling component should use.

        Parameters
        ----------
        component
            Which of the distinct LLM-calling tasks is asking.

        Returns
        -------
        str or None
            The configured model string, or None when this component has none and
            no fallback applies.

        Notes
        -----
        The single seam every component resolves its model through -- there is one
        convention in the codebase, not two. Today it reads the component's
        configured value from the environment, but it is the one place a future
        subscription-tier lookup would hook in, so no call site changes when that
        arrives (the same "new provider -> zero code changes" principle from
        CLAUDE.md's Scalability section, applied per task).

        PROACTIVE, EXTRACTION, SUPERSESSION and VARIANT all fall back to the
        synthesis model when their own is unset: each has its own config value
        (CLAUDE.md forbids assuming one model fits every task) but none is assumed to
        differ from synthesis by default, so a deployment that configures a single
        model still has every call site working rather than some that silently never
        run.

        VARIANT_AUDIT and GROUNDING_CHECK are the two deliberate exceptions: neither
        has a fallback at all, because falling back to `synthesis_model` would
        silently collapse an "independent" check onto the very model it is supposed
        to be independent of -- for VARIANT_AUDIT the generator it audits, for
        GROUNDING_CHECK the synthesis model whose finished answer it checks. See each
        field's own comment for the full reasoning.

        ANSWER_V2 falls back to the synthesis model, so selecting the v2 format
        never changes the model by itself. ANSWER_V2_CHECK falls back to the
        grounding check's model and never to a synthesis model, for the same
        independence reason.

        EXTRACTION_VERIFY has no fallback at all: unset, extraction is not
        verified, exactly as before it existed.
        """
        match component:
            case ModelComponent.SYNTHESIS:
                return self.synthesis_model
            case ModelComponent.PROACTIVE:
                return self.proactive_model or self.synthesis_model
            case ModelComponent.EXTRACTION:
                return self.extraction_model or self.synthesis_model
            case ModelComponent.SUPERSESSION:
                return self.supersession_model or self.synthesis_model
            case ModelComponent.VARIANT:
                return self.variant_model or self.synthesis_model
            case ModelComponent.VARIANT_AUDIT:
                return self.variant_audit_model
            case ModelComponent.GROUNDING_CHECK:
                return self.grounding_check_model
            case ModelComponent.ANSWER_V2:
                return self.answer_v2_model or self.synthesis_model
            case ModelComponent.ANSWER_V2_CHECK:
                return self.answer_v2_check_model or self.grounding_check_model
            case ModelComponent.EXTRACTION_VERIFY:
                return self.extraction_verify_model

    def is_llm_configured(self, component: ModelComponent) -> bool:
        """Report whether enough is present to actually call the LLM for a component.

        Parameters
        ----------
        component
            Which of the distinct LLM-calling tasks is asking.

        Returns
        -------
        bool
            True only when BOTH an API key and a resolved model string are present.
            A model with no key, or a key with no model, each count as "not
            configured".

        Notes
        -----
        The one place this gets decided, so it is never re-implemented or
        second-guessed at each call site (see /aura-ask and the proactive responder).
        """
        return bool(self.llm_api_key and self.resolve_model(component))


def load_settings() -> Settings:
    """Load and validate settings from the environment.

    Returns
    -------
    Settings
        A fully validated configuration.

    Raises
    ------
    ConfigurationError
        On any validation failure, carrying a single plain-text message rather
        than pydantic's structured error.

    Notes
    -----
    The entry point production code (main.py) should use. It translates
    pydantic's ValidationError into one readable message so a misconfigured
    deployment fails immediately with a stated cause instead of a traceback
    surfacing three layers down inside discord.py.

    The raw values -- DISCORD_TOKEN, LLM_API_KEY, INTERNAL_API_SECRET -- are
    never included in that message; `include_input=False` keeps them out of the
    loop that builds it.
    """
    try:
        return Settings()
    except ValidationError as exc:
        # include_input=False: the raw values -- DISCORD_TOKEN, LLM_API_KEY,
        # INTERNAL_API_SECRET -- are not even handed to this loop.
        messages = [
            str(error.get("ctx", {}).get("error", error["msg"]))
            for error in exc.errors(include_input=False)
        ]
        raise ConfigurationError(" ".join(messages)) from exc
