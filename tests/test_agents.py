import pytest

from tau_rec.agents.base import BaseAgent
from tau_rec.agents.litellm_agent import LiteLLMAgent, _rejects_temperature


@pytest.mark.parametrize("model", [
    # Anthropic deprecated temperature at Opus 4.7; these reject the request.
    "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "claude-fable-5", "claude-fable-5-1",
    # Same weights, other routes: OpenRouter spells the version with a dot.
    "anthropic/claude-opus-5", "openrouter/anthropic/claude-opus-4.8",
    "openrouter/anthropic/claude-fable-5.1",
    # Pre-existing cases: o-series and gpt-5 take no temperature either.
    "o1", "o3-mini", "gpt-5-mini", "gpt-5.6-sol",
])
def test_temperature_rejected(model):
    assert _rejects_temperature(model) is True


@pytest.mark.parametrize("model", [
    # 4.6 and earlier still accept temperature. claude-sonnet-4-6 is a
    # published leaderboard entry, so moving this boundary rescores the board.
    "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001", "claude-3-5-sonnet-20240620",
    "gpt-4o", "openrouter/deepseek/deepseek-v4-flash", "openrouter/qwen/qwen3-32b",
])
def test_temperature_accepted(model):
    assert _rejects_temperature(model) is False


def test_reasoning_effort_routed_by_param_support():
    """Models that reject `reasoning_effort` get OpenRouter's `reasoning` field.

    litellm raises rather than dropping the unsupported parameter, so sending it
    to such a model fails every trial in the run.
    """
    from tau_rec.agents.litellm_agent import _supports_reasoning_effort

    assert _supports_reasoning_effort("gpt-5.6-sol") is True
    assert _supports_reasoning_effort("openrouter/deepseek/deepseek-v4-flash") is True
    assert _supports_reasoning_effort("openrouter/meta/muse-spark-1.3") is False
    # A model litellm knows nothing about keeps the old behaviour: send it and
    # let the provider decide, rather than silently switching transport.
    assert _supports_reasoning_effort("not-a-real-model-xyz") is True
    # An unrecognised OpenRouter model routes through the native field instead,
    # because litellm would refuse to forward reasoning_effort for it.
    assert _supports_reasoning_effort("openrouter/vendor/not-a-real-model-xyz") is False

def test_base_agent_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        BaseAgent()

def test_litellm_agent_init():
    agent = LiteLLMAgent(
        model="gpt-4o",
        system_prompt="You are a movie assistant.",
        tool_definitions=[],
    )
    assert agent.model == "gpt-4o"

def test_litellm_agent_add_user_message():
    agent = LiteLLMAgent(
        model="gpt-4o",
        system_prompt="test",
        tool_definitions=[],
    )
    agent.add_user_message("hello")
    assert len(agent._message_history) == 1
    assert agent._message_history[0]["role"] == "user"
    assert agent._message_history[0]["content"] == "hello"

def test_litellm_agent_reset():
    agent = LiteLLMAgent(
        model="gpt-4o",
        system_prompt="test",
        tool_definitions=[],
    )
    agent.add_user_message("hello")
    agent.reset()
    assert len(agent._message_history) == 0


def test_anthropic_gets_cache_breakpoints_and_others_do_not():
    from tau_rec.agents.litellm_agent import _marks_cache_explicitly

    assert _marks_cache_explicitly("claude-sonnet-4-6") is True
    assert _marks_cache_explicitly("openrouter/anthropic/claude-opus-4.8") is True
    # These providers cache a repeated prefix on their own; marking is a no-op
    # at best and a schema error at worst.
    assert _marks_cache_explicitly("gpt-5.4") is False
    assert _marks_cache_explicitly("openrouter/meta/muse-spark-1.3") is False


def test_cache_breakpoints_land_on_system_and_newest_message():
    from tau_rec.agents.litellm_agent import _with_cache_breakpoints

    msgs = [
        {"role": "system", "content": "policy text"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "something scary"},
    ]
    out = _with_cache_breakpoints(msgs)
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert out[2]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # Anthropic allows four breakpoints; spending more than needed wastes them.
    marked = sum(
        1 for m in out if isinstance(m.get("content"), list)
        and any("cache_control" in b for b in m["content"])
    )
    assert marked == 2
    # The input must not be mutated - the agent reuses its own history.
    assert msgs[0]["content"] == "policy text"


def test_cache_breakpoint_skips_tool_call_only_turns():
    """An assistant turn holding only tool_calls has empty content.

    Marking it would send an empty text block, which the API rejects outright —
    so the breakpoint has to fall back to the last turn carrying real text.
    """
    from tau_rec.agents.litellm_agent import _with_cache_breakpoints

    msgs = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "find me a film"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
    ]
    out = _with_cache_breakpoints(msgs)
    assert out[2]["content"] == ""
    assert out[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
