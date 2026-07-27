"""Application configuration loaded from environment variables and `.env`."""
from __future__ import annotations

from enum import StrEnum

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_EXAMPLE_HINT = "Copy .env.example to .env and fill in the required values."


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
    )

    # No default (would make a blank/missing token indistinguishable from a
    # deliberate empty value); validate_default=True forces the validator
    # below to run even when the variable is absent entirely.
    discord_token: str = Field(default="", validate_default=True)
    # LLM_PROVIDER, LLM_API_KEY, and SYNTHESIS_MODEL are all optional and unset
    # by default, on purpose: the bot as a whole -- every command except
    # /aura-ask -- must start and run completely normally with none of these
    # present, so a deployment that never wants LLM features simply omits them.
    # The defaults stay None even though .env.example now ships a measured model
    # choice: a missing key with a hardcoded model here would look "configured"
    # in code while failing on every call. is_llm_configured() below is the one
    # place that decides whether enough is actually here to make a call.
    llm_provider: str | None = None
    llm_api_key: str | None = None
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
    database_path: str = "data/aura.db"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Phase 1d's own test data showed related content scoring around 0.98
    # cosine similarity and unrelated content around 0.08 against this
    # project's embedding model -- 0.4 sits comfortably above the
    # "unrelated" end with real margin to spare, while still being loose
    # enough that a reasonably-phrased question matches its facts. This is
    # specifically the direct-query bar: Phase 2's proactive relief reuses
    # find_similar_facts too but needs its own, much stricter threshold,
    # since a wrong direct answer is only shown to the person who asked,
    # while a wrong proactive interruption is unsolicited for everyone.
    similarity_threshold: float = 0.4

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
    proactive_question_threshold: float = Field(
        default=-0.15, gt=-2.0, le=2.0, allow_inf_nan=False
    )

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
    proactive_similarity_threshold: float = Field(
        default=0.30, ge=-1.0, le=1.0, allow_inf_nan=False
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
    proactive_confidence_gap: float = Field(
        default=0.05, ge=0.0, le=2.0, allow_inf_nan=False
    )

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
    # here is per-guild and changes once many guilds share one operator key
    # (see CLAUDE.md's Open Items on a cross-guild shared budget), even though
    # it doesn't change yet.
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
    proactive_grace_period_seconds: float = Field(
        default=90.0, ge=0.0, le=24 * 60 * 60.0, allow_inf_nan=False
    )

    log_level: str = "INFO"

    @field_validator("discord_token")
    @classmethod
    def _require_non_blank_token(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"DISCORD_TOKEN is missing or blank. {ENV_EXAMPLE_HINT}")
        return stripped

    def resolve_model(self, component: ModelComponent) -> str | None:
        """Resolve the model a given LLM-calling component should use.

        The single seam every component resolves its model through -- there is
        one convention in the codebase, not two. Today it just reads the
        component's configured value from the environment, but it is the one
        place a future subscription-tier (Free/Pro) lookup would hook in, so no
        call site changes when that arrives (the same "new provider -> zero
        code changes" principle from CLAUDE.md's Scalability section, applied
        per task).

        PROACTIVE falls back to the synthesis model when its own is unset: its
        model is its own config value (CLAUDE.md forbids assuming one model
        fits every task) but is deliberately NOT assumed to differ from
        synthesis by default, so a deployment that configures a single model
        still has a working second trigger.
        """
        match component:
            case ModelComponent.SYNTHESIS:
                return self.synthesis_model
            case ModelComponent.PROACTIVE:
                return self.proactive_model or self.synthesis_model

    def is_llm_configured(self, component: ModelComponent) -> bool:
        """Whether enough is present to actually call the LLM for component.

        The one place this gets decided, so it's never re-implemented or
        second-guessed at each call site (see /aura-ask and the proactive
        responder) -- a model string with no key, or a key with no model
        string, both count as "not configured", for whichever component's model
        resolve_model returns.
        """
        return bool(self.llm_api_key and self.resolve_model(component))


def load_settings() -> Settings:
    """Load and validate settings, raising ConfigurationError on failure.

    This is the entry point production code (main.py) should use. It
    translates pydantic's ValidationError into a single plain-text message
    so a misconfigured deployment fails immediately with a readable cause
    instead of a traceback surfacing three layers down inside discord.py.
    """
    try:
        return Settings()
    except ValidationError as exc:
        messages = [
            str(error.get("ctx", {}).get("error", error["msg"])) for error in exc.errors()
        ]
        raise ConfigurationError(" ".join(messages)) from exc
