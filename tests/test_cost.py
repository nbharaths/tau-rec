"""Cost accounting.

The benchmark generates cache hits by the million: the agent's tool loop
resends the whole conversation on every call, so the shared prefix is read back
hundreds of times in a single trial. Pricing those reads at the full input rate
overstates a run several-fold, and it overstates worst for exactly the models
that loop most — which is what makes them look unaffordable to evaluate.
"""
from types import SimpleNamespace

from tau_rec import cost


def _usage(prompt, completion, cached=None, provider_cost=None):
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=details,
        )
    )
    if provider_cost is not None:
        response._hidden_params = {"response_cost": provider_cost}
    return response


def test_cached_tokens_recorded_as_subset_of_input():
    cost.reset()
    cost.record_usage("agent", _usage(1000, 50, cached=800))
    c = cost.snapshot()
    # prompt_tokens already includes the cache hits; cached must not inflate input.
    assert c["agent_input"] == 1000
    assert c["agent_cached"] == 800
    assert c["agent_output"] == 50


def test_missing_cache_details_counts_as_zero_cached():
    cost.reset()
    cost.record_usage("agent", _usage(1000, 50))
    assert cost.snapshot()["agent_cached"] == 0


def test_provider_cost_is_separate_and_reports_coverage():
    cost.reset()
    cost.record_usage("agent", _usage(100, 10, provider_cost=0.0123))
    cost.record_usage("agent", _usage(50, 5))
    cost.record_usage("sim", _usage(20, 2, provider_cost=0.004))
    recorded = cost.provider_cost_snapshot()
    assert recorded["agent"] == {
        "usd": 0.0123,
        "reported_calls": 1,
        "successful_response_calls": 2,
        "coverage": 0.5,
    }
    assert recorded["simulator"]["usd"] == 0.004
    assert abs(recorded["total_usd"] - 0.0163) < 1e-12
    assert recorded["reported_calls"] == 2
    assert recorded["successful_response_calls"] == 3
    assert recorded["coverage"] == 2 / 3


def test_cost_reset_clears_provider_totals():
    cost.reset()
    cost.record_usage("agent", _usage(100, 10, provider_cost=1.0))
    cost.reset()
    assert cost.provider_cost_snapshot()["total_usd"] == 0.0
    assert cost.provider_cost_snapshot()["successful_response_calls"] == 0


def test_provider_budget_stops_after_recorded_cost_reaches_ceiling():
    cost.reset()
    cost.configure_provider_budget(1.0, already_spent_usd=0.2)
    cost.record_usage("agent", _usage(100, 10, provider_cost=0.5))
    assert abs(cost.provider_budget_status()["remaining_usd"] - 0.3) < 1e-12
    cost.record_usage("agent", _usage(100, 10, provider_cost=0.3))
    try:
        cost.check_provider_budget()
    except cost.ProviderBudgetExceededError:
        pass
    else:
        raise AssertionError("budget guard allowed another provider request")


def test_provider_budget_fails_closed_when_cost_is_missing():
    cost.reset()
    cost.configure_provider_budget(1.0)
    try:
        cost.record_usage("agent", _usage(100, 10))
    except cost.ProviderCostUnavailableError:
        pass
    else:
        raise AssertionError("hard ceiling accepted a response without provider cost")
    cost.reset()


def test_single_call_usage_and_cost_use_the_aggregate_accounting_rules():
    usage = cost.response_usage(_usage(1_000, 100, cached=800))
    assert usage == {"input": 1_000, "cached": 800, "cache_write": 0, "output": 100}
    expected = 200 * 0.20e-6 + 800 * 0.02e-6 + 100 * 1.20e-6
    assert cost.call_cost("gpt-5.6-luna", usage) == expected


def test_responses_api_usage_names_are_normalized():
    response = SimpleNamespace(usage=SimpleNamespace(
        input_tokens=1_000,
        output_tokens=100,
        input_tokens_details=SimpleNamespace(cached_tokens=800),
    ))
    usage = cost.response_usage(response)
    assert usage == {"input": 1_000, "cached": 800, "cache_write": 0, "output": 100}
    expected = 200 * 2.00e-6 + 800 * 0.10e-6 + 100 * 10.00e-6
    assert cost.call_cost("gpt-6.1-sol", usage) == expected


def test_cached_tokens_clamped_to_prompt_total():
    """A provider reporting more cached than prompt tokens must not go negative."""
    cost.reset()
    cost.record_usage("agent", _usage(100, 10, cached=500))
    assert cost.snapshot()["agent_cached"] == 100
    priced = cost.price_tokens("gpt-5-mini", "gpt-5-mini", cost.snapshot())
    assert priced["agent"] >= 0


def test_cache_reads_priced_at_the_discounted_rate():
    """Muse Spark: $1.25/M input, $0.15/M cache read — an 8.3x spread."""
    rates = cost._lookup_prices("openrouter/meta/muse-spark-1.3")
    assert (rates.input, rates.output, rates.cache_read) == (1.25e-6, 4.25e-6, 1.5e-7)

    all_fresh = cost.price_tokens(
        "openrouter/meta/muse-spark-1.3", "gpt-5-mini",
        {"agent_input": 1_000_000, "agent_cached": 0, "agent_output": 0,
         "sim_input": 0, "sim_cached": 0, "sim_output": 0},
    )
    all_cached = cost.price_tokens(
        "openrouter/meta/muse-spark-1.3", "gpt-5-mini",
        {"agent_input": 1_000_000, "agent_cached": 1_000_000, "agent_output": 0,
         "sim_input": 0, "sim_cached": 0, "sim_output": 0},
    )
    assert all_fresh["agent"] == 1.25
    assert all_cached["agent"] == 0.15


def test_luna_override_uses_published_rates():
    rates = cost._lookup_prices("openai/gpt-5.6-luna")
    assert rates == cost.Rates(0.20e-6, 1.20e-6, 0.02e-6, 0.20e-6)


def test_gpt61_sol_override_uses_published_rates():
    rates = cost._lookup_prices("gpt-6.1-sol")
    assert rates == cost.Rates(2.00e-6, 10.00e-6, 0.10e-6, 2.50e-6)


def test_open_weight_overrides_use_frozen_route_rates():
    expected = {
        "openrouter/deepseek/deepseek-v4.1-flash":
            cost.Rates(0.30e-6, 1.20e-6, 0.006e-6, 0.30e-6),
        "openrouter/z-ai/glm-5.3-flash":
            cost.Rates(0.15e-6, 0.50e-6, 0.03e-6, 0.15e-6),
        "openrouter/qwen/qwen3.8-27b":
            cost.Rates(0.425e-6, 2.55e-6, 0.085e-6, 0.53125e-6),
    }
    for model, rates in expected.items():
        assert cost._lookup_prices(model) == rates


def test_reproduces_a_real_openrouter_bill():
    """Ground truth: a 9-trial Muse Spark 1.3 calibration billed $3.388.

    Pricing every input token at the full rate predicted $12.60 for this same
    sample. The gap was entirely prompt caching.
    """
    priced = cost.price_tokens(
        "openrouter/meta/muse-spark-1.3", "gpt-5-mini",
        {"agent_input": 9_401_904, "agent_cached": 8_367_694, "agent_output": 198_891,
         "sim_input": 10_710, "sim_cached": 0, "sim_output": 15_624},
    )
    assert abs(priced["agent"] - 3.388) < 0.05


def test_unknown_cache_rate_falls_back_to_full_input_rate():
    """An unpublished cache rate must overstate, never flatter."""
    r = cost._lookup_prices("gpt-4o")
    assert r.cache_read <= r.input
    cost._PRICE_OVERRIDES.setdefault(
        "_test/no-cache-rate", cost.Rates(1e-6, 2e-6, 1e-6, 1e-6))
    assert cost._lookup_prices("_test/no-cache-rate").cache_read == 1e-6
    del cost._PRICE_OVERRIDES["_test/no-cache-rate"]


def test_unknown_model_prices_to_none():
    priced = cost.price_tokens("not-a-real-model-xyz", "gpt-5-mini", cost.snapshot())
    assert priced["agent"] is None
    assert priced["total"] is None


def test_symbolic_simulator_cost_is_zero_even_without_a_model_price():
    priced = cost.price_tokens(
        "gpt-5.6-luna",
        "symbolic-controller",
        {
            "agent_input": 100,
            "agent_cached": 0,
            "agent_cache_write": 0,
            "agent_output": 10,
            "sim_input": 0,
            "sim_cached": 0,
            "sim_cache_write": 0,
            "sim_output": 0,
        },
        simulator_enabled=False,
    )
    assert priced["simulator"] == 0.0
    assert priced["total"] == priced["agent"]


def test_cache_writes_billed_at_the_premium_rate():
    """Anthropic charges 1.25x to write a cache entry and 0.1x to read it.

    Ignoring the premium would make caching look strictly free, which it isn't:
    it only pays because the prefix is read back many times per trial.
    """
    r = cost._lookup_prices("claude-sonnet-4-6")
    assert (r.input, r.cache_write, r.cache_read) == (3e-6, 3.75e-6, 3e-7)

    def agent_cost(cached, written):
        return cost.price_tokens(
            "claude-sonnet-4-6", "gpt-5-mini",
            {"agent_input": 1_000_000, "agent_cached": cached,
             "agent_cache_write": written, "agent_output": 0,
             "sim_input": 0, "sim_cached": 0, "sim_cache_write": 0, "sim_output": 0},
        )["agent"]

    assert agent_cost(0, 0) == 3.00           # all fresh
    assert agent_cost(1_000_000, 0) == 0.30   # all read back
    assert agent_cost(0, 1_000_000) == 3.75   # all written: costs MORE than fresh


def test_write_premium_pays_off_on_a_single_reuse():
    """One write plus N reads against N+1 fresh sends.

    Break-even is N > 0.28, so a prefix read even once already beats sending it
    twice; a prefix never reused costs 25% more than not caching at all. The
    agent's tool loop reads it 10-25 times, so the premium is never in question
    here — but a one-shot call would be the wrong place to mark a breakpoint.
    """
    r = cost._lookup_prices("claude-sonnet-4-6")
    prefix = 1_432  # system prompt + tool definitions, the static prefix
    for reads, better in [(0, False), (1, True), (2, True), (20, True)]:
        cached = prefix * r.cache_write + prefix * reads * r.cache_read
        fresh = prefix * (reads + 1) * r.input
        assert (cached < fresh) is better, f"{reads} reads"
