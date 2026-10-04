"""OpenRouter request options for the v2 answer calls: provider pinning, data policy, reasoning.

A model measured in the P4 bake-off was measured as a model AND a route: which
provider served it, whether that provider may keep or train on the data, and how
much the model reasoned before answering. Production has to send the same
options to get the same behaviour -- DeepSeek V4.1 Flash, for one, is served by
some thirty providers (several quantized to four bits) and reasons by default,
which changes both its answers and its latency.

This module turns the operator's settings into the `extra_body` litellm passes to
OpenRouter, for the two calls of the v2 answer format only (the contract
synthesis and its check). It returns nothing at all when no option is set, so a
call without options is exactly a call as before, and nothing for a model that
is not routed through OpenRouter, whose API these fields belong to.

Pure; imports nothing from aura.
"""

from __future__ import annotations

from typing import Final, Literal

ReasoningSetting = Literal["", "off", "low", "medium", "high"]

# The prefix litellm uses for models it routes through OpenRouter.
OPENROUTER_PREFIX: Final = "openrouter/"


def parse_provider_list(raw: str) -> tuple[str, ...]:
    """Return the provider names of a comma-separated setting, blanks dropped.

    Parameters
    ----------
    raw
        For example "DeepInfra, Together".

    Returns
    -------
    tuple[str, ...]
        The names in order, whitespace trimmed; empty for an empty setting.
    """
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def openrouter_extra_body(
    model: str,
    *,
    providers: tuple[str, ...],
    deny_data_collection: bool,
    reasoning: ReasoningSetting,
) -> dict[str, object] | None:
    """Return the OpenRouter request fields for one call, or None when there are none.

    Parameters
    ----------
    model
        The litellm model string of the call.
    providers
        Providers to pin, in order, with no fallback to others; empty for
        OpenRouter's own routing.
    deny_data_collection
        Use only providers that neither retain nor train on the data.
    reasoning
        "" for the model's default, "off" to switch reasoning off, or an
        effort level.

    Returns
    -------
    dict[str, object] or None
        The fields for litellm's `extra_body`; None when no option is set or
        the model is not routed through OpenRouter.
    """
    if not model.startswith(OPENROUTER_PREFIX):
        return None
    body: dict[str, object] = {}
    provider: dict[str, object] = {}
    if providers:
        provider["order"] = list(providers)
        provider["allow_fallbacks"] = False
    if deny_data_collection:
        provider["data_collection"] = "deny"
    if provider:
        body["provider"] = provider
    if reasoning == "off":
        body["reasoning"] = {"enabled": False}
    elif reasoning:
        body["reasoning"] = {"effort": reasoning}
    return body or None
