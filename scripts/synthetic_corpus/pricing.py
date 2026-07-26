"""Live OpenRouter pricing, fetched rather than remembered.

The model bake-off (reports/model-bakeoff.txt) found that one of the three
prices this project had written down had already moved by 3.7x since it was
noted, and that stale figure was the one the recommendation rested on. So this
module reads the catalog at run time and refuses to guess: a model whose price
cannot be fetched cannot be budgeted, and a run that cannot be budgeted does
not start.

No API key is needed -- OpenRouter's model catalog is public -- which also
means the pre-run cost estimate can be printed before any paid call happens.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from synthetic_corpus.budget import ModelPrice

_CATALOG_URL = "https://openrouter.ai/api/v1/models"
_FETCH_TIMEOUT_SECONDS = 30

# litellm addresses OpenRouter as "openrouter/<vendor>/<model>", while the
# catalog lists "<vendor>/<model>". One prefix, stripped in one place.
_LITELLM_PREFIX = "openrouter/"


class PricingUnavailableError(RuntimeError):
    """Raised when a model's price cannot be established from the live catalog."""


def catalog_id(model: str) -> str:
    """Strip litellm's provider prefix to get OpenRouter's own model id."""
    return model[len(_LITELLM_PREFIX):] if model.startswith(_LITELLM_PREFIX) else model


def fetch_model_prices(models: list[str]) -> dict[str, ModelPrice]:
    """Return live per-million-token prices for every model in `models`.

    Raises PricingUnavailableError if the catalog cannot be read or if any
    requested model is absent from it. Failing on an unknown model is the
    point: an unknown model is one whose spend cannot be capped, and silently
    treating it as free would disable the only ceiling that catches a
    mispriced call.
    """
    try:
        with urllib.request.urlopen(_CATALOG_URL, timeout=_FETCH_TIMEOUT_SECONDS) as response:
            catalog = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise PricingUnavailableError(
            f"could not read OpenRouter's model catalog ({exc}); refusing to run "
            "with an unbudgeted spend ceiling"
        ) from exc

    by_id = {entry.get("id"): entry for entry in catalog.get("data", [])}
    prices: dict[str, ModelPrice] = {}

    for model in models:
        entry = by_id.get(catalog_id(model))
        if entry is None:
            raise PricingUnavailableError(
                f"model {model!r} is not in OpenRouter's catalog; it may have been "
                "renamed or withdrawn"
            )
        pricing = entry.get("pricing") or {}
        try:
            input_price = float(pricing["prompt"]) * 1_000_000
            output_price = float(pricing["completion"]) * 1_000_000
        except (KeyError, TypeError, ValueError) as exc:
            raise PricingUnavailableError(
                f"model {model!r} has no usable prompt/completion pricing in the catalog"
            ) from exc
        prices[model] = ModelPrice(
            model=model,
            usd_per_million_input=input_price,
            usd_per_million_output=output_price,
        )

    return prices
