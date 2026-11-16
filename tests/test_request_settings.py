"""The shared request-settings helper must reproduce the native requests exactly."""
from __future__ import annotations

import pytest

from tau_rec.agents.request_settings import (
    agent_request_settings,
    native_model_name,
    simulator_request_settings,
)
from tau_rec.harness import BOOKS_V4, LEGACY

from tests.test_request_fixtures import (
    BOOKS_MODEL,
    EFFORT_MODELS,
    MOVIES_MODELS,
    _load,
)

_NON_SETTINGS = ("model", "messages", "tools", "timeout")

# Reasoning-effort routing reads litellm's model map; pin the answers the
# fixtures were recorded under (tests/conftest.py).
pytestmark = pytest.mark.usefixtures("pinned_reasoning_effort_params")


def _settings_only(kwargs: dict) -> dict:
    """Native kwargs minus the request envelope and OpenRouter provider routing."""
    out = {k: v for k, v in kwargs.items() if k not in _NON_SETTINGS}
    if "extra_body" in out:
        extra = {k: v for k, v in out["extra_body"].items() if k != "provider"}
        if extra:
            out["extra_body"] = extra
        else:
            del out["extra_body"]
    return out


@pytest.mark.parametrize(
    ("inspect_name", "native"),
    [
        ("openai/gpt-5-mini", "gpt-5-mini"),
        ("openai-api/deepseek/deepseek-chat", "deepseek/deepseek-chat"),
        ("openrouter/qwen/qwen3-32b", "openrouter/qwen/qwen3-32b"),
        ("anthropic/claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"),
        ("mockllm/model", "mockllm/model"),
    ],
)
def test_native_model_name_table(inspect_name, native):
    assert native_model_name(inspect_name) == native


_AGENT_CASES = (
    [(m, None, f"{m}|{LEGACY.id}|default", LEGACY) for m in MOVIES_MODELS]
    + [(m, "high", f"{m}|{LEGACY.id}|effort-high", LEGACY) for m in EFFORT_MODELS]
)


@pytest.mark.parametrize(("model", "effort", "key", "harness"), _AGENT_CASES)
def test_agent_settings_match_native(model, effort, key, harness):
    expected = _settings_only(_load("agent_request_kwargs_movies.json")[key])
    assert agent_request_settings(model, effort, harness) == expected


def test_agent_settings_books_match_native():
    books = _load("agent_request_kwargs_books_v4.json")
    assert books, "fixture is empty"
    for key, kwargs in books.items():
        model = kwargs["model"]
        assert agent_request_settings(model, "high", BOOKS_V4) == _settings_only(kwargs), key


@pytest.mark.parametrize("model", MOVIES_MODELS)
def test_simulator_settings_match_native(model):
    expected = _settings_only(
        _load("simulator_request_kwargs.json")[f"{model}|{LEGACY.id}|simulator"]
    )
    assert simulator_request_settings(model) == expected


def test_simulator_settings_books_model_match_native():
    key = f"{BOOKS_MODEL}|{BOOKS_V4.id}|simulator"
    expected = _settings_only(_load("simulator_request_kwargs.json")[key])
    assert simulator_request_settings(BOOKS_MODEL) == expected
