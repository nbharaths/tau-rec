"""Cost accounting.

The benchmark generates cache hits by the million: the agent's tool loop
resends the whole conversation on every call, so the shared prefix is read back
hundreds of times in a single trial. Pricing those reads at the full input rate
overstates a run several-fold, and it overstates worst for exactly the models
that loop most — which is what makes them look unaffordable to evaluate.
"""
from types import SimpleNamespace

from tau_rec import cost


def _usage(prompt, completion, cached=None):
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=details,
        )
    )


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
