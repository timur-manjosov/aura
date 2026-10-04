"""The arms of the P4 model bake-off: which model, routed how, at what price.

Prices are OpenRouter's, read from its public models and endpoints lists on the
day of the run (2026-10-04) -- the standing rule since reports/model-bakeoff.txt
found a carried price stale by 3.7x. Where an arm pins a provider, the price is
that provider's; otherwise it is the model's listed price (every endpoint of
those models charges the same). The metering in scripts/llm_metering.py books
the provider's own reported cost per call; these prices only bound the worst
case before a call leaves and estimate a run in advance.

Routing choices, and why:

* DeepSeek V4.1 Flash is served by about thirty providers, some quantized to
  four bits. Its arms pin DeepInfra (headquartered in the US, fp8, structured
  outputs) with OpenRouter's "data_collection": "deny", so every call goes to
  one known provider that neither retains nor trains on the data. Reasoning is
  switched off for the main arm, left at the model's default for the extra arm.
* GLM 5.3 Flash is pinned to Together (US) for the same reason; that endpoint
  refuses to switch reasoning off (measured in the pilot), so it reasons at "low".
* Qwen 3.8 Flash has a single provider (Alibaba; data centres in Singapore and
  China) -- recorded, not pinnable to anywhere else.
* Mistral Small is pinned to Mistral's own EU endpoint: the one candidate whose
  data stays in the EU.
* Anthropic, OpenAI and Google models keep OpenRouter's default routing among
  their first-party and cloud endpoints, as production does today.
* Reasoning models get a larger output ceiling (reasoning tokens count against
  it), and their reasoning is set to the lowest level the arm allows; the
  tokens are counted in cost and latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from llm_metering import ArmRouting, ModelPrice


@dataclass(frozen=True)
class Arm:
    """One model as configured for the bake-off.

    Attributes
    ----------
    key
        Short, stable name used in file names and tables.
    model
        The litellm model string.
    price
        Price per million tokens of the endpoint this arm uses.
    routing
        Request options the metering adds to every call of this arm.
    provider_note
        Where the data is processed, for the report.
    temperature_supported
        Whether the endpoint accepts a temperature (reasoning models do not).
    """

    key: str
    model: str
    price: ModelPrice
    routing: ArmRouting = field(default_factory=ArmRouting)
    provider_note: str = ""
    temperature_supported: bool = True


# The output ceiling for an arm whose model reasons before it answers.
REASONING_MAX_TOKENS: Final = 4000

ARMS: Final[dict[str, Arm]] = {
    arm.key: arm
    for arm in (
        Arm(
            "haiku",
            "openrouter/anthropic/claude-haiku-4.5",
            ModelPrice(1.00, 5.00),
            provider_note="Anthropic / Azure / Amazon Bedrock / Google Vertex (US); as production",
        ),
        Arm(
            "sonnet",
            "openrouter/anthropic/claude-sonnet-5.5",
            ModelPrice(2.00, 10.00),
            ArmRouting(max_tokens=REASONING_MAX_TOKENS),
            provider_note="Anthropic / Azure / Bedrock / Vertex (US)",
            temperature_supported=False,
        ),
        Arm(
            "gpt4omini",
            "openrouter/openai/gpt-4o-mini",
            ModelPrice(0.15, 0.60),
            provider_note="OpenAI / Azure (US); as production's grounding check",
        ),
        Arm(
            "deepseek",
            "openrouter/deepseek/deepseek-v4.1-flash",
            ModelPrice(0.14, 0.42),
            ArmRouting(
                provider_order=("DeepInfra",),
                data_collection_deny=True,
                reasoning={"enabled": False},
            ),
            provider_note="DeepInfra (US), fp8, pinned, data_collection=deny; reasoning off",
        ),
        Arm(
            "deepseek-think",
            "openrouter/deepseek/deepseek-v4.1-flash",
            ModelPrice(0.14, 0.42),
            ArmRouting(
                provider_order=("DeepInfra",),
                data_collection_deny=True,
                max_tokens=REASONING_MAX_TOKENS,
            ),
            provider_note="DeepInfra (US), fp8, pinned, data_collection=deny; default reasoning",
        ),
        Arm(
            "gpt6luna",
            "openrouter/openai/gpt-6-luna",
            ModelPrice(0.10, 0.50),
            ArmRouting(reasoning={"effort": "low"}, max_tokens=REASONING_MAX_TOKENS),
            provider_note="OpenAI / Azure (US); reasoning effort low",
            temperature_supported=False,
        ),
        Arm(
            "qwen",
            "openrouter/qwen/qwen3.8-flash",
            ModelPrice(0.15, 0.47),
            ArmRouting(reasoning={"enabled": False}),
            provider_note="Alibaba (data centres SG, CN), the only provider; reasoning off",
        ),
        Arm(
            "glm",
            "openrouter/z-ai/glm-5.3-flash",
            ModelPrice(0.15, 0.50),
            ArmRouting(
                provider_order=("Together",),
                data_collection_deny=True,
                reasoning={"effort": "low"},
                max_tokens=REASONING_MAX_TOKENS,
            ),
            provider_note=(
                "Together (US), pinned, data_collection=deny; reasoning effort low (the "
                "endpoint refuses to switch reasoning off)"
            ),
        ),
        Arm(
            "mistral",
            "openrouter/mistralai/mistral-small-2603",
            ModelPrice(0.165, 0.66),
            ArmRouting(provider_order=("Mistral",), data_collection_deny=True),
            provider_note="Mistral (FR), pinned, data_collection=deny",
        ),
        Arm(
            "flashlite",
            "openrouter/google/gemini-3.1-flash-lite",
            ModelPrice(0.25, 1.50),
            ArmRouting(reasoning={"effort": "low"}, max_tokens=REASONING_MAX_TOKENS),
            provider_note="Google AI Studio / Vertex (US); reasoning effort low",
        ),
        Arm(
            "gemini38",
            "openrouter/google/gemini-3.8-flash",
            ModelPrice(0.75, 3.75),
            ArmRouting(reasoning={"effort": "low"}, max_tokens=REASONING_MAX_TOKENS),
            provider_note="Google AI Studio / Vertex (US); reasoning effort low",
        ),
    )
}


def prices() -> dict[str, ModelPrice]:
    """Return the price of every arm's model, for the metering (one price per model string).

    Returns
    -------
    dict[str, ModelPrice]
        By litellm model string. Two arms on one model (DeepSeek with and
        without reasoning) share the pinned provider's price.
    """
    table: dict[str, ModelPrice] = {}
    for arm in ARMS.values():
        table.setdefault(arm.model, arm.price)
    return table
