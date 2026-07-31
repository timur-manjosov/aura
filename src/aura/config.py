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
    EXTRACTION = "extraction"
    SUPERSESSION = "supersession"


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
    # TESTING-PHASE OVERRIDE (2026-07-31): -0.02 above remains the calibrated,
    # precision-favouring value this whole comment block justifies, and that
    # reasoning is unchanged -- it assumed extraction running at real,
    # full-traffic volume across a community server (reports/phase-3a-2.txt
    # Section 8's cost math: up to ~$16/guild/month worst case at full daily-
    # cap utilization). Aura is currently running on a single-member test
    # server, where that cost model is close to moot and a precision-favouring
    # bar mostly just means fewer facts to look at while testing. -0.04 is
    # used here instead, taken directly from the same phase-3a-1b.txt sweep
    # (Section 6) rather than any new calibration: P=0.553, R=0.787,
    # specificity=0.925, F1=0.650 -- a real recall gain over -0.02's R=0.707,
    # at essentially the same F1, and deliberately short of -0.05 (P=0.495,
    # already below half) and -0.06 (P=0.474, F1=0.608, worse than -0.02's),
    # where the sweep tips into flagging more noise than signal. REVERT to
    # -0.02 before any real multi-member community rollout, when the
    # full-volume cost math above becomes load-bearing again.
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
    # than 60 escalations a day is. Revisit alongside proactive_daily_cap
    # before any multi-guild rollout -- CLAUDE.md's Open Items note about a
    # cross-guild shared budget applies to this cap identically.
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
    extraction_dedup_similarity_threshold: float = Field(
        default=0.60, ge=-1.0, le=1.0, allow_inf_nan=False
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
    # call fires only for a candidate that scored above 0.70 against an existing
    # active fact, a small minority of what extraction produces. When the cap
    # does bind, nothing breaks and nothing is lost -- the candidate is still
    # staged and still reviewed, it simply carries Phase 3a-2's plain similarity
    # hint instead of a judgment. 0 is valid and disables the judgment call
    # entirely while leaving the rest of extraction working.
    #
    # Revisit alongside the other two before any multi-guild rollout;
    # CLAUDE.md's Open Items note on a cross-guild shared budget applies here
    # identically.
    supersession_daily_cap: int = Field(default=50, ge=0, le=1_000_000)

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

        PROACTIVE, EXTRACTION and SUPERSESSION all fall back to the synthesis
        model when their own is unset: each has its own config value (CLAUDE.md
        forbids assuming one model fits every task) but none is assumed to
        differ from synthesis by default, so a deployment that configures a
        single model still has every call site working rather than some that
        silently never run.
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
