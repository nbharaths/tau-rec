"""Token/cost accounting.

LiteLLMAgent and UserSimulator call `record_usage()` after every
litellm.acompletion call so we can measure how many tokens a typical trial
consumes, then extrapolate to the full experiment.
"""
from __future__ import annotations
from typing import NamedTuple
import litellm

# Module-level accumulators keyed by role ("agent" / "sim").
#
# `*_cached` and `*_cache_write` are subsets of `*_input`, not additions to it:
# providers report prompt_tokens inclusive of both. They are tracked separately
# because the three bill at different rates, and this benchmark moves cache
# tokens by the million — the agent's tool loop resends the whole conversation
# on every call, so the shared prefix is written once and read back over and
# over. Anthropic charges a 1.25x premium to write, which is only worth paying
# because the prefix is then read 10-25 times per trial at a tenth of list.
_COUNTERS: dict[str, int] = {
    "agent_input": 0,
    "agent_cached": 0,
    "agent_cache_write": 0,
    "agent_output": 0,
    "sim_input": 0,
    "sim_cached": 0,
    "sim_cache_write": 0,
    "sim_output": 0,
}


class Rates(NamedTuple):
    """Per-token prices. `cache_write` equals `input` where there is no premium."""
    input: float
    output: float
    cache_read: float
    cache_write: float


def reset() -> None:
    for k in _COUNTERS:
        _COUNTERS[k] = 0


def snapshot() -> dict[str, int]:
    return dict(_COUNTERS)


def record_usage(kind: str, response) -> None:
    """Extract prompt/completion tokens from a litellm response and add to counters."""
    usage = getattr(response, "usage", None)
    if not usage:
        return
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    written = (getattr(details, "cache_creation_tokens", 0) or 0) if details else 0
    # Anthropic reports fresh/write/read as three disjoint counts and litellm
    # sums them into prompt_tokens, so the subsets must not exceed the total.
    cached = min(cached, prompt)
    written = min(written, prompt - cached)
    _COUNTERS[f"{kind}_input"] += prompt
    _COUNTERS[f"{kind}_cached"] += cached
    _COUNTERS[f"{kind}_cache_write"] += written
    _COUNTERS[f"{kind}_output"] += completion


# Models litellm has no pricing for, so `--dry-run` can still report a total.
# Rates are per token, taken from the provider's own pricing endpoint.
#
# Only the paid tiers are listed. Several providers offer the same weights far
# cheaper on a tier that retains prompts for training — for Muse Spark that is
# `meta/muse-spark-1.x-contributor`, at roughly a tenth of the price. Do not
# benchmark on those: a run sends every task and its answer to the provider, so
# a retaining tier would leak the task set into a future training corpus and
# quietly destroy the benchmark it was measuring.
_PRICE_OVERRIDES: dict[str, Rates] = {
    "meta/muse-spark-1.3": Rates(1.25e-6, 4.25e-6, 1.5e-7, 1.25e-6),
    "meta/muse-spark-1.2": Rates(1.25e-6, 4.25e-6, 1.5e-7, 1.25e-6),
}


def _lookup_prices(model: str) -> Rates | None:
    """Return per-token rates, or None if the model's pricing is unknown.

    Both cache rates fall back to the full input rate when the provider
    publishes no separate figure, so an unknown rate overstates rather than
    flatters — a silent discount would be the dangerous direction to guess in.
    """
    for key, prices in _PRICE_OVERRIDES.items():
        if model == key or model.endswith("/" + key):
            return prices
    info = litellm.model_cost.get(model)
    if not info:
        # Try bare model name (strip provider prefix)
        info = litellm.model_cost.get(model.split("/")[-1])
    if not info:
        return None
    in_rate = info.get("input_cost_per_token", 0.0) or 0.0
    return Rates(
        in_rate,
        info.get("output_cost_per_token", 0.0) or 0.0,
        info.get("cache_read_input_token_cost") or in_rate,
        info.get("cache_creation_input_token_cost") or in_rate,
    )


def _split_cost(in_toks: float, cached_toks: float, written_toks: float,
                out_toks: float, prices: Rates | None) -> float | None:
    """Price tokens, billing each subset of the input at its own rate."""
    if prices is None:
        return None
    cached = min(max(cached_toks, 0.0), in_toks)
    written = min(max(written_toks, 0.0), in_toks - cached)
    fresh = in_toks - cached - written
    return (
        fresh * prices.input
        + cached * prices.cache_read
        + written * prices.cache_write
        + out_toks * prices.output
    )


def price_tokens(agent_model: str, sim_model: str, tokens: dict[str, int]) -> dict:
    """Price an observed token count. Values are None when rates are unknown."""
    agent = _split_cost(
        tokens.get("agent_input", 0), tokens.get("agent_cached", 0),
        tokens.get("agent_cache_write", 0), tokens.get("agent_output", 0),
        _lookup_prices(agent_model),
    )
    sim = _split_cost(
        tokens.get("sim_input", 0), tokens.get("sim_cached", 0),
        tokens.get("sim_cache_write", 0), tokens.get("sim_output", 0),
        _lookup_prices(sim_model),
    )
    return {
        "agent": agent,
        "simulator": sim,
        "total": None if agent is None or sim is None else agent + sim,
    }


def estimate_cost(
    agent_model: str,
    sim_model: str,
    sampled_trials: int,
    total_trials: int,
) -> dict:
    """Scale observed sample counters up to the full experiment and price it.

    Returns a dict with per-trial averages, total token estimates, and USD costs.
    """
    c = snapshot()
    if sampled_trials <= 0:
        raise ValueError("sampled_trials must be > 0")

    per_trial = {k: v / sampled_trials for k, v in c.items()}
    est = {k: v * total_trials for k, v in per_trial.items()}

    agent_prices = _lookup_prices(agent_model)
    sim_prices = _lookup_prices(sim_model)

    agent_cost = _split_cost(est["agent_input"], est["agent_cached"],
                             est["agent_cache_write"], est["agent_output"], agent_prices)
    sim_cost = _split_cost(est["sim_input"], est["sim_cached"],
                           est["sim_cache_write"], est["sim_output"], sim_prices)

    total_cost: float | None
    if agent_cost is not None and sim_cost is not None:
        total_cost = agent_cost + sim_cost
    else:
        total_cost = None

    return {
        "sampled_trials": sampled_trials,
        "total_trials": total_trials,
        "per_trial": per_trial,
        "estimated_total_tokens": est,
        "prices": {
            "agent": agent_prices,
            "sim": sim_prices,
        },
        "estimated_cost_usd": {
            "agent": agent_cost,
            "sim": sim_cost,
            "total": total_cost,
        },
    }
