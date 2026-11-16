"""Per-model request settings shared by the native agent, the native user
simulator and the Inspect adapter.

The rule logic lives here so every path sends the same settings for the same
model.  A change to anything below changes the requests a reproduction sends.
"""
from __future__ import annotations

import functools
import re

from tau_rec.harness import HarnessProfile

# Anthropic deprecated `temperature` at Claude Opus 4.7 — 4.6 and earlier still
# accept it, later models reject the request outright. Match on the version
# number rather than a list of names: the same model reaches us as
# `claude-opus-4-8` direct and `anthropic/claude-opus-4.8` via OpenRouter, and
# the 5 family omits the minor version entirely (`claude-opus-5`).
_CLAUDE_VERSION = re.compile(r"claude-[a-z]+-(\d+)(?:[-.](\d+))?")
_TEMPERATURE_DEPRECATED_FROM = (4, 7)


@functools.cache
def _supports_reasoning_effort(model: str) -> bool:
    """Whether litellm will forward `reasoning_effort` for this model.

    Some models reason unconditionally and have no `reasoning_effort` in their
    param map; litellm raises before the request leaves rather than dropping
    it, which would fail every trial in a run.
    """
    import litellm

    provider = "openrouter" if model.startswith("openrouter/") else None
    name = model.split("/", 1)[1] if provider else model
    try:
        params = litellm.get_supported_openai_params(model=name, custom_llm_provider=provider)
    except Exception:
        params = None
    if params is None:
        # litellm knows nothing about this model. Send the parameter and let the
        # provider decide, which is what this code did before there was a check.
        return True
    # Otherwise the param map is authoritative. An OpenRouter model litellm does
    # not recognise lands here too, reported as unsupported — which is the right
    # answer regardless, since litellm would refuse to forward the parameter.
    return "reasoning_effort" in params


def _claude_version(model: str) -> tuple[int, int] | None:
    match = _CLAUDE_VERSION.search(model.lower())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _rejects_temperature(model: str) -> bool:
    name = model.lower()
    if any(tag in name for tag in ("o1", "o3", "o4", "gpt-5", "gpt-6")):
        return True
    version = _claude_version(name)
    return version is not None and version >= _TEMPERATURE_DEPRECATED_FROM


# The same models dropped thinking budgets: effort is set through
# `output_config`, and a `budget_tokens` request is rejected. litellm 1.83.0
# still turns `reasoning_effort` into a budget for every Claude model except
# 4.6, so these get `output_config` directly. OpenRouter Claude models keep
# OpenRouter's own `reasoning` field.
def _effort_via_output_config(model: str) -> bool:
    version = _claude_version(model)
    return version is not None and version >= _TEMPERATURE_DEPRECATED_FROM


def _direct_output_config_effort(model: str) -> bool:
    return _effort_via_output_config(model) and not model.startswith("openrouter/")


def _disables_qwen_thinking(model: str, reasoning_effort: str | None) -> bool:
    """Preserve legacy non-thinking Qwen3 runs unless reasoning is explicit."""
    return (
        "openrouter" in model.lower()
        and "qwen3" in model.lower()
        and reasoning_effort is None
    )


def agent_request_settings(
    model: str, reasoning_effort: str | None, harness: HarnessProfile,
) -> dict:
    """Per-model settings the native agent adds to a chat request.

    Excludes ``model``, ``messages``, ``tools``, ``timeout`` and OpenRouter
    ``provider`` routing, which the caller owns.
    """
    settings: dict = {}
    # Some models (o-series, gpt-5, Claude 4.7+) don't support temperature
    if not _rejects_temperature(model):
        settings["temperature"] = 0.0
    if "deepseek" in model.lower():
        thinking_mode = "enabled" if reasoning_effort else "disabled"
        # litellm refuses a top-level `thinking` for OpenRouter models its
        # param map does not list, failing every call before it is sent.
        if not model.startswith("openrouter/"):
            settings["thinking"] = {"type": thinking_mode}
        settings["extra_body"] = {"thinking": {"type": thinking_mode}}
    # Legacy Qwen3 leaderboard runs used the non-thinking mode. Preserve
    # that behavior when no effort is requested, but allow newer Qwen3.x
    # models to participate in explicitly labelled reasoning experiments.
    if _disables_qwen_thinking(model, reasoning_effort):
        settings["extra_body"] = {"thinking": {"type": "disabled"}}
    if reasoning_effort:
        if _direct_output_config_effort(model):
            settings["output_config"] = {"effort": reasoning_effort}
        elif _supports_reasoning_effort(model):
            settings["reasoning_effort"] = reasoning_effort
        else:
            # OpenRouter's own `reasoning` field is the only way to steer
            # these. litellm forwards extra_body verbatim. A model may
            # accept a level it does not actually honour, and single
            # requests are far too noisy to tell the difference — reasoning
            # length varies severalfold on identical input. Compare token
            # totals across whole runs before labelling one with a level.
            settings.setdefault("extra_body", {})["reasoning"] = {"effort": reasoning_effort}
    if harness.agent_max_tokens:
        settings["max_tokens"] = harness.agent_max_tokens
    return settings


def simulator_request_settings(model: str) -> dict:
    """Per-model settings the native user simulator adds to a chat request."""
    settings: dict = {"temperature": 1.0}
    if "deepseek" in model.lower():
        # litellm refuses a top-level `thinking` for unmapped OpenRouter models.
        if not model.startswith("openrouter/"):
            settings["thinking"] = {"type": "disabled"}
        settings["extra_body"] = {"thinking": {"type": "disabled"}}
    return settings


def native_model_name(inspect_model: str) -> str:
    """The native litellm model name for an Inspect model name."""
    if inspect_model.startswith("openai-api/"):
        return inspect_model[len("openai-api/"):]
    if inspect_model.startswith("openai/"):
        return inspect_model[len("openai/"):]
    return inspect_model
