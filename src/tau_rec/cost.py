"""Token/cost accounting.

LiteLLMAgent and UserSimulator call `record_usage()` after every
litellm.acompletion call so we can measure how many tokens a typical trial
consumes, then extrapolate to the full experiment.
"""
from __future__ import annotations
import litellm

# Module-level accumulators keyed by role ("agent" / "sim").
_COUNTERS: dict[str, int] = {
    "agent_input": 0,
    "agent_output": 0,
    "sim_input": 0,
    "sim_output": 0,
}


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
    _COUNTERS[f"{kind}_input"] += prompt
    _COUNTERS[f"{kind}_output"] += completion


def _lookup_prices(model: str) -> tuple[float, float] | None:
    """Return (input_cost_per_token, output_cost_per_token) or None if unknown."""
    info = litellm.model_cost.get(model)
    if not info:
        # Try bare model name (strip provider prefix)
        info = litellm.model_cost.get(model.split("/")[-1])
    if not info:
        return None
    return (
        info.get("input_cost_per_token", 0.0) or 0.0,
        info.get("output_cost_per_token", 0.0) or 0.0,
    )


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

    agent_in_per = c["agent_input"] / sampled_trials
    agent_out_per = c["agent_output"] / sampled_trials
    sim_in_per = c["sim_input"] / sampled_trials
    sim_out_per = c["sim_output"] / sampled_trials

    est_agent_in = agent_in_per * total_trials
    est_agent_out = agent_out_per * total_trials
    est_sim_in = sim_in_per * total_trials
    est_sim_out = sim_out_per * total_trials

    agent_prices = _lookup_prices(agent_model)
    sim_prices = _lookup_prices(sim_model)

    def _cost(in_toks: float, out_toks: float, prices: tuple[float, float] | None) -> float | None:
        if prices is None:
            return None
        in_rate, out_rate = prices
        return in_toks * in_rate + out_toks * out_rate

    agent_cost = _cost(est_agent_in, est_agent_out, agent_prices)
    sim_cost = _cost(est_sim_in, est_sim_out, sim_prices)

    total_cost: float | None
    if agent_cost is not None and sim_cost is not None:
        total_cost = agent_cost + sim_cost
    else:
        total_cost = None

    return {
        "sampled_trials": sampled_trials,
        "total_trials": total_trials,
        "per_trial": {
            "agent_input": agent_in_per,
            "agent_output": agent_out_per,
            "sim_input": sim_in_per,
            "sim_output": sim_out_per,
        },
        "estimated_total_tokens": {
            "agent_input": est_agent_in,
            "agent_output": est_agent_out,
            "sim_input": est_sim_in,
            "sim_output": est_sim_out,
        },
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
