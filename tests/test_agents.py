import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tau_rec.agents.base import BaseAgent
from tau_rec.agents.litellm_agent import (
    LiteLLMAgent,
    _rejects_temperature,
    _responses_tool_definitions,
    _uses_responses_api,
    request_parameter_manifest,
)


@pytest.mark.parametrize("model", [
    # Anthropic deprecated temperature at Opus 4.7; these reject the request.
    "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "claude-fable-5", "claude-fable-5-1",
    # Same weights, other routes: OpenRouter spells the version with a dot.
    "anthropic/claude-opus-5", "openrouter/anthropic/claude-opus-4.8",
    "openrouter/anthropic/claude-fable-5.1",
    # Pre-existing cases: o-series and gpt-5 take no temperature either.
    "o1", "o3-mini", "gpt-5-mini", "gpt-5.6-sol", "gpt-6.1-sol",
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


def test_request_parameter_manifest_names_omitted_defaults():
    regular = request_parameter_manifest("gpt-4o", None)
    assert regular["temperature"] == 0.0
    assert regular["top_p"] == "omitted (provider default)"
    assert regular["seed"] == "omitted (provider default)"
    reasoning = request_parameter_manifest("gpt-5.6-luna", "medium")
    assert str(reasoning["temperature"]).startswith("omitted")
    assert reasoning["reasoning_effort"] == "medium"
    deepseek = request_parameter_manifest(
        "openrouter/deepseek/deepseek-v4-flash", "high"
    )
    assert deepseek["thinking"] == {"type": "enabled"}


def test_direct_gpt6_uses_responses_and_converts_tools():
    assert _uses_responses_api("gpt-6.1-sol") is True
    assert _uses_responses_api("openai/gpt-6-astra") is True
    assert _uses_responses_api("openrouter/openai/gpt-6.1-sol") is False
    converted = _responses_tool_definitions([{
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Search",
            "parameters": {"type": "object", "properties": {}},
        },
    }])
    assert converted == [{
        "type": "function",
        "name": "search_catalog",
        "description": "Search",
        "parameters": {"type": "object", "properties": {}},
    }]
    assert request_parameter_manifest("gpt-6.1-sol", "high")["api"] == "responses"


def test_reasoning_effort_routed_by_param_support(monkeypatch):
    """Models that reject `reasoning_effort` get OpenRouter's `reasoning` field.

    litellm raises rather than dropping the unsupported parameter, so sending it
    to such a model fails every trial in the run.
    """
    import litellm
    from tau_rec.agents.litellm_agent import _supports_reasoning_effort

    # litellm's param map is data, refreshed from the network at import when
    # it can be, so the test fakes the map and checks the routing decisions
    # taken on it.
    table = {
        ("gpt-5.6-sol", None): ["temperature", "reasoning_effort"],
        ("deepseek/deepseek-v4-flash", "openrouter"): ["reasoning_effort"],
        ("meta/muse-spark-1.3", "openrouter"): ["temperature", "max_tokens"],
        # litellm answers the generic OpenRouter set for an id it has not seen.
        ("vendor/not-a-real-model-xyz", "openrouter"): ["temperature", "max_tokens"],
    }

    def fake(model, custom_llm_provider=None):
        if model == "not-a-real-model-xyz":
            return None
        if model == "raises-xyz":
            raise ValueError("no such model")
        return table[(model, custom_llm_provider)]

    monkeypatch.setattr(litellm, "get_supported_openai_params", fake)
    _supports_reasoning_effort.cache_clear()
    try:
        assert _supports_reasoning_effort("gpt-5.6-sol") is True
        assert _supports_reasoning_effort("openrouter/deepseek/deepseek-v4-flash") is True
        assert _supports_reasoning_effort("openrouter/meta/muse-spark-1.3") is False
        # A model litellm knows nothing about keeps the old behaviour: send it
        # and let the provider decide, rather than silently switching transport.
        assert _supports_reasoning_effort("not-a-real-model-xyz") is True
        assert _supports_reasoning_effort("raises-xyz") is True
        # An unrecognised OpenRouter model routes through the native field
        # instead, because litellm would refuse to forward reasoning_effort.
        assert _supports_reasoning_effort("openrouter/vendor/not-a-real-model-xyz") is False
    finally:
        _supports_reasoning_effort.cache_clear()


def test_qwen_reasoning_override_preserves_legacy_and_allows_explicit_effort():
    """The old Qwen3 row remains non-thinking; Qwen3.8 can opt into reasoning."""
    from tau_rec.agents.litellm_agent import _disables_qwen_thinking

    assert _disables_qwen_thinking("openrouter/qwen/qwen3-32b", None) is True
    assert _disables_qwen_thinking("openrouter/qwen/qwen3.8-27b", None) is True
    assert _disables_qwen_thinking("openrouter/qwen/qwen3.8-27b", "high") is False
    assert _disables_qwen_thinking("qwen/qwen3.8-27b", None) is False


def test_openrouter_json_decode_failure_is_short_retryable():
    from tau_rec.agents.litellm_agent import _retryable_provider_error

    retryable, short_delay = _retryable_provider_error(
        RuntimeError("OpenrouterException - Unable to get json response")
    )
    assert retryable is True
    assert short_delay is True


def test_provider_request_timeout_is_retryable():
    from tau_rec.agents.litellm_agent import _retryable_provider_error

    retryable, short_delay = _retryable_provider_error(
        RuntimeError("OpenrouterException - 408 Request Timeout")
    )
    assert retryable is True
    assert short_delay is False

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


async def test_litellm_agent_records_per_call_provenance(monkeypatch):
    import litellm

    response = SimpleNamespace(
        id="response-1",
        model="gpt-5.6-luna-2026-09-01",
        system_fingerprint="fp_test",
        _hidden_params={"custom_llm_provider": "openai", "model_id": "deployment-7"},
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=80,
                cache_creation_tokens=0,
            ),
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello", tool_calls=None))],
    )
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=response))
    agent = LiteLLMAgent("gpt-5.6-luna", "system", [])

    result = await agent.respond()

    assert result.message == "Hello"
    assert len(agent.call_log) == 1
    call = agent.call_log[0]
    assert call["response_model"] == "gpt-5.6-luna-2026-09-01"
    assert call["system_fingerprint"] == "fp_test"
    assert call["provider"] == "openai"
    assert call["tokens"] == {
        "input": 100, "cached": 80, "cache_write": 0, "output": 10,
    }
    assert call["latency_seconds"] >= 0


async def test_gpt6_responses_tool_loop_preserves_output_items(monkeypatch):
    import litellm

    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=10,
        input_tokens_details=SimpleNamespace(cached_tokens=80),
    )
    reasoning = SimpleNamespace(type="reasoning")
    function_call = SimpleNamespace(
        type="function_call",
        call_id="call-1",
        name="search_catalog",
        arguments='{"query":"space opera"}',
    )
    first = SimpleNamespace(
        id="resp-1", model="gpt-6.1-sol", output_text="",
        output=[reasoning, function_call], usage=usage,
        _hidden_params={"custom_llm_provider": "openai", "response_cost": 0.001},
    )
    message_item = SimpleNamespace(type="message")
    second = SimpleNamespace(
        id="resp-2", model="gpt-6.1-sol", output_text="I found one.",
        output=[message_item], usage=usage,
        _hidden_params={"custom_llm_provider": "openai", "response_cost": 0.001},
    )
    mocked = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(litellm, "aresponses", mocked)
    agent = LiteLLMAgent(
        "gpt-6.1-sol",
        "system",
        [{
            "type": "function",
            "function": {
                "name": "search_catalog",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        reasoning_effort="high",
    )
    agent.add_assistant_message("Hello")
    agent.add_user_message("Find a space opera")

    result = await agent.respond()
    assert result.tool_calls == [{
        "id": "call-1",
        "name": "search_catalog",
        "arguments": {"query": "space opera"},
    }]
    first_kwargs = mocked.await_args_list[0].kwargs
    assert first_kwargs["instructions"] == "system"
    assert first_kwargs["reasoning"] == {"effort": "high"}
    assert first_kwargs["store"] is False
    assert first_kwargs["include"] == ["reasoning.encrypted_content"]

    follow_up = await agent.respond([{
        "role": "tool", "tool_call_id": "call-1", "content": "[]",
    }])
    assert follow_up.message == "I found one."
    second_input = mocked.await_args_list[1].kwargs["input"]
    assert reasoning in second_input
    assert function_call in second_input
    assert second_input[-1] == {
        "type": "function_call_output", "call_id": "call-1", "output": "[]",
    }
    assert agent.call_log[-1]["request_parameters"]["api"] == "responses"


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


DEEPSEEK = "openrouter/deepseek/deepseek-v4.1-flash"
PIN = {
    "order": ["deepinfra/fp8"],
    "allow_fallbacks": False,
    "require_parameters": True,
    "data_collection": "deny",
    "zdr": True,
}


def _served(provider: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello", tool_calls=None))],
        usage=None,
        provider=provider,
    )


async def test_openrouter_pin_is_merged_into_extra_body(monkeypatch):
    import litellm

    completion = AsyncMock(return_value=_served("DeepInfra"))
    monkeypatch.setattr(litellm, "acompletion", completion)
    agent = LiteLLMAgent(
        model=DEEPSEEK, system_prompt="s", tool_definitions=[],
        reasoning_effort="high", provider_order=["deepinfra/fp8"],
    )

    await agent.respond()

    kwargs = completion.await_args.kwargs
    assert kwargs["extra_body"]["provider"] == PIN
    assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}
    assert "thinking" not in kwargs  # litellm rejects it for unmapped OpenRouter models


async def test_served_provider_and_pin_are_logged(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=_served("DeepInfra")))
    agent = LiteLLMAgent(
        model=DEEPSEEK, system_prompt="s", tool_definitions=[],
        reasoning_effort="high", provider_order=["deepinfra/fp8"],
    )

    await agent.respond()

    assert agent.call_log[0]["served_provider"] == "DeepInfra"
    assert agent.call_log[0]["provider_order"] == ["deepinfra/fp8"]


def test_unservable_pin_is_not_retried():
    from tau_rec.agents.litellm_agent import _retryable_provider_error

    error = RuntimeError(
        "No endpoints found for deepseek/deepseek-v4.1-flash. Every candidate "
        "endpoint was removed during routing: provider unavailable"
    )
    assert _retryable_provider_error(error) == (False, False)


def test_request_manifest_records_the_provider_pin():
    manifest = request_parameter_manifest(DEEPSEEK, "high", provider_order=["deepinfra/fp8"])
    assert manifest["provider_routing"] == PIN
    assert "provider_routing" not in request_parameter_manifest(DEEPSEEK, "high")


def test_pinned_litellm_entries_resolve_models_missing_from_the_bundled_map():
    """With the frozen litellm map, the frontier model still resolves and is priced."""
    import subprocess
    import sys

    script = (
        "import os; os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'\n"
        "import litellm\n"
        "from tau_rec.agents.litellm_agent import _ensure_pinned_models\n"
        "_ensure_pinned_models()\n"
        "print(litellm.get_llm_provider('gpt-6.1-sol')[1])\n"
        "print(litellm.model_cost['gpt-6.1-sol']['input_cost_per_token'])\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    provider, price = [line.strip() for line in out.stdout.splitlines() if line.strip()][-2:]
    assert provider == "openai"
    assert float(price) == 2e-06


# --- Claude 4.7+ request path ----------------------------------------------

import json as _json
from pathlib import Path as _Path

from tau_rec.harness import BOOKS_V4 as _BOOKS_V4

_SEARCH_TOOL = [{"type": "function", "function": {
    "name": "search_catalog", "description": "d",
    "parameters": {"type": "object", "properties": {}},
}}]


def _chat_response(content="Hello", tool_calls=None, thinking_blocks=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    if thinking_blocks is not None:
        message.thinking_blocks = thinking_blocks
    return SimpleNamespace(
        id="r", model="m", system_fingerprint=None, _hidden_params={}, usage=None,
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        provider="P",
    )


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
async def test_claude_5_effort_goes_in_output_config(monkeypatch, model):
    import litellm

    completion = AsyncMock(return_value=_chat_response())
    monkeypatch.setattr(litellm, "acompletion", completion)
    agent = LiteLLMAgent(model, "sys", _SEARCH_TOOL, reasoning_effort="high", harness=_BOOKS_V4)
    agent.add_user_message("hi")

    await agent.respond()

    kwargs = completion.await_args.kwargs
    assert kwargs["output_config"] == {"effort": "high"}
    for absent in ("reasoning_effort", "thinking", "temperature", "top_p", "top_k", "tool_choice"):
        assert absent not in kwargs
    assert "reasoning" not in kwargs.get("extra_body", {})


@pytest.mark.parametrize("model,expected", [
    ("claude-opus-5-5", True), ("claude-sonnet-5-5", True), ("claude-opus-4-7", True),
    ("claude-fable-5-1", True), ("openrouter/anthropic/claude-opus-4.8", True),
    ("claude-opus-4-6", False), ("claude-sonnet-4-6", False), ("gpt-6.1-sol", False),
    ("openrouter/z-ai/glm-5.3-flash", False),
])
def test_effort_via_output_config_versions(model, expected):
    from tau_rec.agents.litellm_agent import _effort_via_output_config

    assert _effort_via_output_config(model) is expected


async def test_claude_thinking_blocks_are_replayed_unchanged(monkeypatch):
    import litellm

    blocks = [{"type": "thinking", "thinking": "", "signature": "sig1"}]
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="search_catalog", arguments="{}"))
    completion = AsyncMock(side_effect=[
        _chat_response(content=None, tool_calls=[call], thinking_blocks=blocks),
        _chat_response(),
    ])
    monkeypatch.setattr(litellm, "acompletion", completion)
    agent = LiteLLMAgent("claude-opus-5-5", "sys", _SEARCH_TOOL, reasoning_effort="high", harness=_BOOKS_V4)
    agent.add_user_message("hi")

    await agent.respond()
    await agent.respond(tool_results=[{"role": "tool", "tool_call_id": "c1", "content": "ok"}])

    second = completion.await_args_list[1].kwargs["messages"]
    replayed = [m for m in second if m.get("role") == "assistant" and m.get("tool_calls")]
    assert replayed and replayed[0]["thinking_blocks"] == blocks


async def test_non_claude_request_kwargs_unchanged(monkeypatch):
    """The open models' requests must match the v4 runs byte for byte."""
    import litellm

    expected = _json.loads((_Path(__file__).parent / "fixtures" / "agent_request_kwargs_books_v4.json").read_text())
    orders = {
        "openrouter/deepseek/deepseek-v4.1-flash": ["deepinfra/fp8"],
        "openrouter/z-ai/glm-5.3-flash": ["near-ai/fp8"],
        "openrouter/qwen/qwen3.8-27b": ["near-ai/fp8"],
    }
    for model, order in orders.items():
        completion = AsyncMock(return_value=_chat_response())
        monkeypatch.setattr(litellm, "acompletion", completion)
        agent = LiteLLMAgent(model, "sys", _SEARCH_TOOL, reasoning_effort="high",
                             provider_order=order, harness=_BOOKS_V4)
        agent.add_user_message("hi")
        await agent.respond()
        assert _json.loads(_json.dumps(completion.await_args.kwargs)) == expected[model]


async def test_claude_refusal_is_logged_not_rerouted(monkeypatch):
    """litellm reports Anthropic's `refusal` stop reason as `content_filter`; it is logged, not rerouted."""
    import litellm
    from litellm.litellm_core_utils.core_helpers import map_finish_reason

    completion = AsyncMock(return_value=_chat_response(content="", finish_reason=map_finish_reason("refusal")))
    monkeypatch.setattr(litellm, "acompletion", completion)
    monkeypatch.setattr("tau_rec.agents.litellm_agent.asyncio.sleep", AsyncMock())
    agent = LiteLLMAgent("claude-opus-5-5", "sys", _SEARCH_TOOL, reasoning_effort="high", harness=_BOOKS_V4)
    agent.add_user_message("hi")

    await agent.respond()

    assert agent.call_log[-1]["finish_reason"] == "content_filter"
    assert {c.kwargs["model"] for c in completion.await_args_list} == {"claude-opus-5-5"}


async def test_claude_call_log_records_output_config_effort(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=_chat_response()))
    agent = LiteLLMAgent("claude-opus-5-5", "sys", _SEARCH_TOOL, reasoning_effort="high", harness=_BOOKS_V4)
    agent.add_user_message("hi")

    await agent.respond()

    assert agent.call_log[-1]["request_parameters"]["output_config"] == {"effort": "high"}


async def test_non_claude_call_log_rows_have_no_output_config(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=_chat_response()))
    agent = LiteLLMAgent("openrouter/deepseek/deepseek-v4.1-flash", "sys", _SEARCH_TOOL, reasoning_effort="high",
                         provider_order=["deepinfra/fp8"], harness=_BOOKS_V4)
    agent.add_user_message("hi")

    await agent.respond()

    assert "output_config" not in agent.call_log[-1]["request_parameters"]


def test_claude_models_are_pinned_for_the_local_map():
    """Runs freeze litellm's bundled map, which predates the Claude 5.5 models."""
    import litellm
    from tau_rec.agents.litellm_agent import _PINNED_MODELS_PATH

    bundled = _json.loads(
        _Path(litellm.__file__).with_name("model_prices_and_context_window_backup.json").read_text()
    )
    pins = _json.loads(_PINNED_MODELS_PATH.read_text())["models"]
    for model in ("claude-opus-5-5", "claude-sonnet-5-5"):
        entry = bundled.get(model) or pins.get(model)
        assert entry is not None, model
        assert entry["litellm_provider"] == "anthropic"
        assert entry["input_cost_per_token"] > 0 and entry["output_cost_per_token"] > 0
        assert entry["cache_read_input_token_cost"] > 0


def test_claude_manifest_names_output_config_effort():
    manifest = request_parameter_manifest("claude-opus-5-5", "high", harness=_BOOKS_V4)
    assert manifest["output_config"] == {"effort": "high"}
    assert manifest["thinking"] == "adaptive (always on, not sent)"
