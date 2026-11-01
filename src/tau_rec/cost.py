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

# Provider/gateway-reported charges are tracked separately from prices inferred
# from tokens. The latter remain useful for projections, but an OpenRouter bill
# can differ materially when a model's live routing price is newer than
# LiteLLM's bundled price table.
_PROVIDER_COSTS: dict[str, float] = {"agent": 0.0, "sim": 0.0}
_PROVIDER_REPORTED_CALLS: dict[str, int] = {"agent": 0, "sim": 0}
_SUCCESSFUL_RESPONSE_CALLS: dict[str, int] = {"agent": 0, "sim": 0}
_MAX_PROVIDER_COST_USD: float | None = None
_PROVIDER_COST_OFFSET_USD = 0.0


class ProviderBudgetError(RuntimeError):
    """Base class for fail-closed provider-cost enforcement."""


class ProviderBudgetExceededError(ProviderBudgetError):
    pass


class ProviderCostUnavailableError(ProviderBudgetError):
    pass


class Rates(NamedTuple):
    """Per-token prices. `cache_write` equals `input` where there is no premium."""
    input: float
    output: float
    cache_read: float
    cache_write: float


def reset() -> None:
    global _MAX_PROVIDER_COST_USD, _PROVIDER_COST_OFFSET_USD
    for k in _COUNTERS:
        _COUNTERS[k] = 0
    for values in (
        _PROVIDER_COSTS, _PROVIDER_REPORTED_CALLS, _SUCCESSFUL_RESPONSE_CALLS,
    ):
        for key in values:
            values[key] = 0
    _MAX_PROVIDER_COST_USD = None
    _PROVIDER_COST_OFFSET_USD = 0.0


def configure_provider_budget(max_usd: float, *, already_spent_usd: float = 0.0) -> None:
    """Require complete provider-cost reporting and stop at the given ceiling."""
    global _MAX_PROVIDER_COST_USD, _PROVIDER_COST_OFFSET_USD
    if max_usd <= 0:
        raise ValueError("provider budget must be positive")
    if already_spent_usd < 0 or already_spent_usd >= max_usd:
        raise ValueError("already-spent provider cost must be below the budget")
    _MAX_PROVIDER_COST_USD = float(max_usd)
    _PROVIDER_COST_OFFSET_USD = float(already_spent_usd)


def provider_budget_status() -> dict:
    current = sum(_PROVIDER_COSTS.values())
    total = _PROVIDER_COST_OFFSET_USD + current
    return {
        "max_usd": _MAX_PROVIDER_COST_USD,
        "already_spent_usd": _PROVIDER_COST_OFFSET_USD,
        "current_process_usd": current,
        "total_usd": total,
        "remaining_usd": (
            None if _MAX_PROVIDER_COST_USD is None
            else max(0.0, _MAX_PROVIDER_COST_USD - total)
        ),
        "exceeded": (
            False if _MAX_PROVIDER_COST_USD is None
            else total > _MAX_PROVIDER_COST_USD
        ),
    }


def check_provider_budget() -> None:
    """Fail before a new request once recorded spend reaches the ceiling."""
    if _MAX_PROVIDER_COST_USD is None:
        return
    total = _PROVIDER_COST_OFFSET_USD + sum(_PROVIDER_COSTS.values())
    if total >= _MAX_PROVIDER_COST_USD:
        raise ProviderBudgetExceededError(
            f"provider-cost ceiling reached: ${total:.4f} / "
            f"${_MAX_PROVIDER_COST_USD:.4f}"
        )


def snapshot() -> dict[str, int]:
    return dict(_COUNTERS)


def provider_cost_snapshot() -> dict:
    """Return gateway/provider charges and their successful-call coverage."""
    roles = {}
    for internal, public in (("agent", "agent"), ("sim", "simulator")):
        responses = _SUCCESSFUL_RESPONSE_CALLS[internal]
        reported = _PROVIDER_REPORTED_CALLS[internal]
        roles[public] = {
            "usd": _PROVIDER_COSTS[internal],
            "reported_calls": reported,
            "successful_response_calls": responses,
            "coverage": reported / responses if responses else None,
        }
    response_calls = sum(_SUCCESSFUL_RESPONSE_CALLS.values())
    reported_calls = sum(_PROVIDER_REPORTED_CALLS.values())
    return {
        **roles,
        "total_usd": sum(_PROVIDER_COSTS.values()),
        "reported_calls": reported_calls,
        "successful_response_calls": response_calls,
        "coverage": reported_calls / response_calls if response_calls else None,
    }


def response_usage(response) -> dict[str, int]:
    """Normalize one provider response's token accounting."""
    usage = getattr(response, "usage", None)
    if not usage:
        return {"input": 0, "cached": 0, "cache_write": 0, "output": 0}
    # Chat Completions names these prompt/completion; Responses names them
    # input/output. Both totals include cached tokens.
    prompt = (
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", 0)
        or 0
    )
    completion = (
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", 0)
        or 0
    )
    details = (
        getattr(usage, "prompt_tokens_details", None)
        or getattr(usage, "input_tokens_details", None)
    )
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    written = (getattr(details, "cache_creation_tokens", 0) or 0) if details else 0
    cached = min(cached, prompt)
    written = min(written, prompt - cached)
    return {
        "input": int(prompt),
        "cached": int(cached),
        "cache_write": int(written),
        "output": int(completion),
    }


def call_cost(model: str, usage: dict[str, int]) -> float | None:
    """Price one normalized call using the same rates as aggregate accounting."""
    return _split_cost(
        usage.get("input", 0),
        usage.get("cached", 0),
        usage.get("cache_write", 0),
        usage.get("output", 0),
        _lookup_prices(model),
    )


def record_usage(kind: str, response) -> None:
    """Extract prompt/completion tokens from a litellm response and add to counters."""
    if kind not in _PROVIDER_COSTS:
        raise ValueError(f"unknown usage kind: {kind}")
    usage = response_usage(response)
    _COUNTERS[f"{kind}_input"] += usage["input"]
    _COUNTERS[f"{kind}_cached"] += usage["cached"]
    _COUNTERS[f"{kind}_cache_write"] += usage["cache_write"]
    _COUNTERS[f"{kind}_output"] += usage["output"]
    _SUCCESSFUL_RESPONSE_CALLS[kind] += 1
    hidden_value = getattr(response, "_hidden_params", {}) or {}
    hidden = hidden_value if isinstance(hidden_value, dict) else {}
    reported_cost = hidden.get("response_cost")
    if _MAX_PROVIDER_COST_USD is not None and not (
        isinstance(reported_cost, (int, float))
        and not isinstance(reported_cost, bool)
    ):
        raise ProviderCostUnavailableError(
            "provider omitted response_cost while a hard cost ceiling is active"
        )
    if isinstance(reported_cost, (int, float)) and not isinstance(
        reported_cost, bool
    ):
        _PROVIDER_COSTS[kind] += float(reported_cost)
        _PROVIDER_REPORTED_CALLS[kind] += 1
    if _MAX_PROVIDER_COST_USD is not None:
        total = _PROVIDER_COST_OFFSET_USD + sum(_PROVIDER_COSTS.values())
        if total > _MAX_PROVIDER_COST_USD:
            raise ProviderBudgetExceededError(
                f"provider-cost ceiling exceeded by the completed response: "
                f"${total:.4f} / ${_MAX_PROVIDER_COST_USD:.4f}"
            )


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
    # OpenAI's published direct-API GPT-6.1 Sol rates (2026-10-06).
    "gpt-6.1-sol": Rates(2.00e-6, 10.00e-6, 0.10e-6, 2.50e-6),
    # OpenAI's published GPT-5.6 Luna rates. New models can predate litellm's
    # bundled table, which must not turn a mandatory cost gate into "unknown".
    "gpt-5.6-luna": Rates(0.20e-6, 1.20e-6, 0.02e-6, 0.20e-6),
    # Open-weight model routes, priced from the provider/OpenRouter catalogs
    # on 2026-10-05. Provider-reported response_cost remains authoritative;
    # these rates keep dry-run and trace-local estimates useful when a model is
    # newer than LiteLLM's bundled table.
    "deepseek/deepseek-v4.1-flash": Rates(0.30e-6, 1.20e-6, 0.006e-6, 0.30e-6),
    "z-ai/glm-5.3-flash": Rates(0.15e-6, 0.50e-6, 0.03e-6, 0.15e-6),
    "qwen/qwen3.8-27b": Rates(0.425e-6, 2.55e-6, 0.085e-6, 0.53125e-6),
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


def price_tokens(
    agent_model: str,
    sim_model: str,
    tokens: dict[str, int],
    *,
    simulator_enabled: bool = True,
) -> dict:
    """Price an observed token count. Values are None when rates are unknown."""
    agent = _split_cost(
        tokens.get("agent_input", 0), tokens.get("agent_cached", 0),
        tokens.get("agent_cache_write", 0), tokens.get("agent_output", 0),
        _lookup_prices(agent_model),
    )
    sim = (
        _split_cost(
            tokens.get("sim_input", 0), tokens.get("sim_cached", 0),
            tokens.get("sim_cache_write", 0), tokens.get("sim_output", 0),
            _lookup_prices(sim_model),
        )
        if simulator_enabled else 0.0
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
    *,
    simulator_enabled: bool = True,
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
    sim_cost = (
        _split_cost(est["sim_input"], est["sim_cached"],
                    est["sim_cache_write"], est["sim_output"], sim_prices)
        if simulator_enabled else 0.0
    )

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
