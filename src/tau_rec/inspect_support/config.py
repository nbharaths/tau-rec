"""Native request settings expressed as an Inspect ``GenerateConfig``.

The settings come from ``tau_rec.agents.request_settings`` (the helper the
native harness uses), so the adapter sends what native would send for the
same model. A field the user set explicitly always wins.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, get_args

from inspect_ai.model import GenerateConfig

# Native keys that are GenerateConfig fields of the same name and meaning.
_DIRECT_FIELDS = ("temperature", "reasoning_effort", "max_tokens")

# The values GenerateConfig.effort accepts (its annotation is Literal[...] | None).
_EFFORT_VALUES = frozenset(
    value
    for arg in get_args(GenerateConfig.model_fields["effort"].annotation)
    for value in get_args(arg)
)


def fill_unset(mirrored: dict, explicit: GenerateConfig) -> GenerateConfig:
    """Return ``explicit`` with only its unset (``None``) fields taken from ``mirrored``.

    ``mirrored`` is the dict ``agent_request_settings`` (or
    ``simulator_request_settings``) returns. Keys with no GenerateConfig field
    go to ``extra_body``, which the OpenAI-compatible providers merge into the
    request body as litellm does:

    - ``thinking`` becomes ``extra_body["thinking"]``;
    - ``output_config`` (Claude >= 4.7 effort) becomes ``GenerateConfig.effort``,
      which Inspect's Anthropic provider sends as ``output_config.effort``; a
      shape ``effort`` cannot hold falls back to ``extra_body["output_config"]``.

    ``extra_body`` is merged key by key with any explicit ``extra_body``, and
    explicit keys win. Neither argument is modified.
    """
    remaining = dict(mirrored)
    extra_body: dict[str, Any] = dict(remaining.pop("extra_body", None) or {})
    fields: dict[str, Any] = {}

    for key in _DIRECT_FIELDS:
        if key in remaining:
            fields[key] = remaining.pop(key)

    if "thinking" in remaining:
        # Native sends DeepSeek's toggle both top-level and in extra_body with
        # the same value; on the wire both land in the request body.
        extra_body.setdefault("thinking", remaining.pop("thinking"))

    if "output_config" in remaining:
        output_config = remaining.pop("output_config")
        effort = output_config.get("effort") if isinstance(output_config, dict) else None
        if set(output_config or {}) == {"effort"} and effort in _EFFORT_VALUES:
            fields["effort"] = effort
        else:
            extra_body.setdefault("output_config", output_config)

    if remaining:
        raise ValueError(
            f"native request setting(s) with no Inspect mapping: {sorted(remaining)}"
        )

    updates: dict[str, Any] = {
        key: value for key, value in fields.items() if getattr(explicit, key) is None
    }
    if extra_body:
        updates["extra_body"] = {**extra_body, **(explicit.extra_body or {})}
    return explicit.model_copy(update=deepcopy(updates), deep=True)
