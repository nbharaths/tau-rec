"""Session-wide test setup.

litellm loads its model map once, at import: from GitHub unless
``LITELLM_LOCAL_MODEL_COST_MAP`` is set, else the map bundled with the
package. Setting the flag here, before any test module imports litellm, makes
every session use the bundled map regardless of collection order or network
access. Tests whose answer depends on the map's contents pin it themselves
(``pinned_reasoning_effort_params`` below).
"""
from __future__ import annotations

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest  # noqa: E402

# litellm.get_supported_openai_params answers, keyed by (model, provider), for
# every model the request-settings fixtures send a reasoning effort to. These
# are the answers the fixtures were recorded under (litellm 1.83.0, remote
# model map): the bundled map has no entry for openrouter/deepseek-v4-flash
# and would report reasoning_effort unsupported.
_PINNED_PARAMS: dict[tuple[str, str | None], list[str]] = {
    ("deepseek/deepseek-v4-flash", "openrouter"): ["temperature", "max_tokens", "reasoning_effort"],
    ("deepseek/deepseek-chat", None): ["temperature", "max_tokens", "reasoning_effort"],
    ("qwen/qwen3-32b", "openrouter"): ["temperature", "max_tokens"],
    # Books-v4 fixture models: litellm's generic OpenRouter set.
    ("deepseek/deepseek-v4.1-flash", "openrouter"): ["temperature", "max_tokens"],
    ("qwen/qwen3.8-27b", "openrouter"): ["temperature", "max_tokens"],
    ("z-ai/glm-5.3-flash", "openrouter"): ["temperature", "max_tokens"],
}


@pytest.fixture
def pinned_reasoning_effort_params(monkeypatch):
    """Pin litellm's reasoning-effort support answers for the fixture models.

    ``_supports_reasoning_effort`` is cached per process, so the cache is
    cleared before (drop answers from the real map) and after (do not leak the
    pinned answers into other tests).
    """
    import litellm

    from tau_rec.agents import request_settings

    original = litellm.get_supported_openai_params

    def pinned(model, custom_llm_provider=None, *args, **kwargs):
        key = (model, custom_llm_provider)
        if key in _PINNED_PARAMS:
            return list(_PINNED_PARAMS[key])
        return original(model, custom_llm_provider, *args, **kwargs)

    monkeypatch.setattr(litellm, "get_supported_openai_params", pinned)
    request_settings._supports_reasoning_effort.cache_clear()
    yield
    request_settings._supports_reasoning_effort.cache_clear()
