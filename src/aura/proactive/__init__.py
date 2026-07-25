"""Proactive relief (CLAUDE.md's second trigger): the full loop, message to reply.

As of Phase 2a-3 the pipeline is complete end to end. A message in a
proactive-enabled channel is scored, matched against the knowledge model, and
checked against a durable cooldown and daily cap (the free, local, race-safe
gate proven in 2a-1/2a-2); a message the gate finds eligible is handed to the
responder, which calls the LLM, reads its own answers_question self-assessment,
and posts a visibly-distinct public answer -- but only when every check,
including that self-assessment, agrees. This is the first point in Aura's
existence where money is spent and a message is posted unprompted, so the
posting policy is deliberately far more conservative than /aura-ask.
"""
from aura.proactive.gate import ProactiveGateConfig, evaluate_message
from aura.proactive.grace import GraceRegistry, GraceWaitOutcome
from aura.proactive.listener import handle_message, should_classify
from aura.proactive.question_detector import (
    QUESTION_EXEMPLARS,
    STATEMENT_EXEMPLARS,
    QuestionDetector,
)
from aura.proactive.responder import ProactiveResponseOutcome, respond_with_synthesis

__all__ = [
    "QUESTION_EXEMPLARS",
    "STATEMENT_EXEMPLARS",
    "GraceRegistry",
    "GraceWaitOutcome",
    "ProactiveGateConfig",
    "ProactiveResponseOutcome",
    "QuestionDetector",
    "evaluate_message",
    "handle_message",
    "respond_with_synthesis",
    "should_classify",
]
