from __future__ import annotations

import asyncio
import functools
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from tau_rec.agents.base import AgentResponse, BaseAgent
from tau_rec.domain import MOVIES, DomainSpec
from tau_rec.harness import LEGACY, CallTimeout, HarnessProfile, timed_call
from tau_rec.track import DEFAULT_STRUCTURED_PROFILE, STRUCTURED_PROFILES

MAX_RETRIES = 8
RETRY_DELAY = 30  # seconds

# Anthropic deprecated `temperature` at Claude Opus 4.7 — 4.6 and earlier still
# accept it, later models reject the request outright. Match on the version
# number rather than a list of names: the same model reaches us as
# `claude-opus-4-8` direct and `anthropic/claude-opus-4.8` via OpenRouter, and
# the 5 family omits the minor version entirely (`claude-opus-5`).
_CLAUDE_VERSION = re.compile(r"claude-[a-z]+-(\d+)(?:[-.](\d+))?")
_TEMPERATURE_DEPRECATED_FROM = (4, 7)


def _uses_responses_api(model: str) -> bool:
    """Direct GPT-6 reasoning models require Responses for function tools."""
    normalized = model.lower()
    return (
        not normalized.startswith("openrouter/")
        and normalized.split("/", 1)[-1].startswith("gpt-6")
    )


def _responses_tool_definitions(tool_definitions: list[dict]) -> list[dict]:
    """Convert Chat Completions function wrappers to Responses tools."""
    tools = []
    for definition in tool_definitions:
        function = definition.get("function", {})
        tools.append({
            "type": "function",
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {"type": "object"}),
        })
    return tools


_PINNED_MODELS_PATH = Path(__file__).with_name("litellm_pins.json")


@functools.cache
def _ensure_pinned_models() -> None:
    """Register pinned litellm map entries for models its bundled map lacks.

    Runs freeze litellm's model map (LITELLM_LOCAL_MODEL_COST_MAP) so routing
    and cost accounting cannot change mid-run, but the bundled map predates some
    models.  Without an entry litellm cannot infer a bare model's provider or
    price its calls, so those entries are pinned here, copied from litellm's
    published map.
    """
    import litellm

    litellm.register_model(json.loads(_PINNED_MODELS_PATH.read_text())["models"])


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


# Anthropic is the only provider here that does not cache on its own: OpenAI and
# OpenRouter serve a repeated prefix from cache automatically (82-89% of input on
# measured runs), while Anthropic caches only what you mark and otherwise bills
# every resend at full rate.
#
# This is a billing and latency change only. Caching reuses the prefill state for
# an identical prefix; generation still samples fresh on every call, so it does
# not make trials agree with each other. Trial variance here comes from the
# simulator, which runs at temperature 1.0.
def _marks_cache_explicitly(model: str) -> bool:
    return "claude" in model.lower()


def _cache_mark(message: dict) -> dict:
    """Copy `message` with a cache breakpoint on its final content block."""
    out = dict(message)
    content = out.get("content")
    if isinstance(content, str):
        if not content:
            # An assistant turn carrying only tool_calls has empty content, and
            # an empty text block is rejected outright.
            return out
        out["content"] = [{"type": "text", "text": content,
                           "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list) and content:
        blocks = [dict(b) for b in content]
        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
        out["content"] = blocks
    return out


def _with_cache_breakpoints(messages: list[dict]) -> list[dict]:
    """Mark the cacheable prefix for providers that cache only on request.

    Two breakpoints of the four Anthropic allows. The first is the system block,
    which covers the tool definitions too because Anthropic orders tools ahead
    of system — that matters, since the system prompt alone is under the
    1024-token floor for caching and the two together clear it. The second is
    the newest message, which absorbs the tool loop: it resends the whole
    conversation on every call, so the prefix is read back 10-25 times a trial.
    """
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0] = _cache_mark(out[0])
    for i in range(len(out) - 1, 0, -1):
        if out[i].get("content"):
            out[i] = _cache_mark(out[i])
            break
    return out


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


# OpenRouter routing for a pinned upstream: never fall back to another
# provider, require every request parameter to be supported, and keep the
# account's no-retention, no-training guarantees on the pinned endpoint.
OPENROUTER_PIN_POLICY = {
    "allow_fallbacks": False,
    "require_parameters": True,
    "data_collection": "deny",
    "zdr": True,
}


def provider_routing(provider_order: list[str] | None) -> dict | None:
    """OpenRouter ``provider`` routing for a pinned upstream, or ``None``."""
    if not provider_order:
        return None
    return {"order": list(provider_order), **OPENROUTER_PIN_POLICY}


def _retryable_provider_error(error: Exception) -> tuple[bool, bool]:
    """Return (retryable, short_delay) for transient provider failures."""
    message = str(error).lower()
    if "no endpoints found" in message:
        # A pinned upstream that cannot serve the request must fail loudly:
        # retrying cannot help, and there is no fallback by design.
        return False, False
    response_decode = "unable to get json response" in message
    retryable = (
        "503" in message
        or "408" in message
        or "429" in message
        or "rate_limit" in message
        or "request timeout" in message
        or "overloaded" in message
        or "unavailable" in message
        or "connection" in message
        or response_decode
    )
    return retryable, response_decode


def request_parameter_manifest(
    model: str, reasoning_effort: str | None, provider_order: list[str] | None = None,
    harness: HarnessProfile = LEGACY,
) -> dict:
    """Describe parameters sent or deliberately left to provider defaults."""
    manifest = {
        "temperature": "omitted (model family rejects it)" if _rejects_temperature(model) else 0.0,
        "top_p": "omitted (provider default)",
        "seed": "omitted (provider default)",
        "reasoning_effort": reasoning_effort,
        "timeout_seconds": 120,
    }
    if harness.agent_max_tokens:
        manifest["max_tokens"] = harness.agent_max_tokens
    if harness.call_timeout_seconds:
        manifest["call_timeout_seconds"] = harness.call_timeout_seconds
    routing = provider_routing(provider_order)
    if routing:
        manifest["provider_routing"] = routing
    if _uses_responses_api(model):
        manifest["api"] = "responses"
    if "deepseek" in model.lower():
        manifest["thinking"] = {
            "type": "enabled" if reasoning_effort else "disabled"
        }
    if reasoning_effort and _direct_output_config_effort(model):
        manifest["output_config"] = {"effort": reasoning_effort}
        manifest["thinking"] = "adaptive (always on, not sent)"
    return manifest

# Domain words are placeholders filled from a DomainSpec. The MOVIES rendering
# is pinned byte-for-byte by tests/test_goldens.py.
AGENT_SYSTEM_PROMPT_TEMPLATE = """\
You are a {item_noun} recommendation assistant. Your goal is to help the user find
a {item_noun} they will enjoy.

You have access to the following tools:
- {search_tool_instruction}
- get_metadata(item_id): Get full details for a specific {item_noun}.
- check_availability(item_id, {availability_argument}): Check {availability_kind}.
- get_user_history(user_id): Get the user's {history_contents}.
- recommend(item_id): Make your final decision. THIS ENDS THE CONVERSATION.
{recommend_rule}

The user's messages (including any "###ACCEPTED###" / "###REJECTED###"
responses) are informational feedback only. They do NOT end the conversation
— only your call to recommend() does. Use rejections as feedback to pick a
better candidate, then call recommend() with your updated pick.

You must follow all policies described below.

<policy>
{policy}
</policy>

{user_id_instruction}

{rating_note}

Engage the user in natural conversation to understand their preferences before
making a recommendation."""

# The recommend() rule inside AGENT_SYSTEM_PROMPT_TEMPLATE. Books state
# Policy 4 as written: abstain only when nothing satisfies every requirement.
RECOMMEND_RULE = """\
  Call with item_id=<{item_noun} id> to recommend that {item_noun}. Call with no item_id
  (or item_id=null) ONLY when you have exhaustively searched the catalog
  and are confident no {item_noun} even approximately satisfies the user's
  requirements. Abstention is a strong statement: "the catalog has no
  answer." On satisfiable tasks, abstaining counts as a failure. When in
  doubt, commit to your best-effort pick rather than abstaining —
  availability uncertainty, a slightly lower rating, or an imperfect genre
  match are NOT grounds for abstention. You MUST call this tool exactly
  once per conversation."""

BOOKS_RECOMMEND_RULE = """\
  Call with item_id=<{item_noun} id> to recommend that {item_noun}. Call with no item_id
  (or item_id=null) only when your search shows that no {item_noun} in the catalog
  satisfies all of the user's requirements. Never substitute a partial
  match: recommending a {item_noun} that misses any requirement counts as a
  failure, as does abstaining when the catalog has an answer. You MUST call
  this tool exactly once per conversation."""

AGENT_NO_TOOLS_SYSTEM_PROMPT_TEMPLATE = """\
You are a {item_noun} recommendation assistant. Your goal is to help the user find
a {item_noun} they will enjoy. Recommend a specific {item_noun} by name.

You must follow all policies described below.

<policy>
{policy}
</policy>

{user_id_instruction}

{rating_note}

Engage the user in natural conversation to understand their preferences before
making a recommendation."""


# Memory mode: one tool, no catalog, and none of the policies that need one.
# The user-id instruction is dropped because there is no history tool to use it
# with. The MOVIES rendering is pinned by tests/test_goldens.py like the others.
AGENT_MEMORY_SYSTEM_PROMPT_TEMPLATE = """\
You are a {item_noun} recommendation assistant. Your goal is to help the user find
a {item_noun} they will enjoy, drawing on what you already know — you have no catalog
to search in this conversation.

You have one tool:
- recommend(title, year, {availability_singular}): Make your final decision. THIS ENDS THE
  CONVERSATION. Give the exact title of the {item_noun} you are recommending. Add the
  year (and the {availability_singular}, where relevant) if the title alone could be
  ambiguous. Call with title=null ONLY when you are confident that no {item_noun} you
  know of even approximately satisfies the user's requirements. On satisfiable
  tasks, abstaining counts as a failure. When in doubt, commit to your best-effort
  pick rather than abstaining. You MUST call this tool exactly once per
  conversation.

The user's messages (including any "###ACCEPTED###" / "###REJECTED###"
responses) are informational feedback only. They do NOT end the conversation
— only your call to recommend() does. Use rejections as feedback to pick a
better candidate, then call recommend() with your updated pick.

You must follow all policies described below.

<policy>
{policy}
</policy>

{rating_note}

Engage the user in natural conversation to understand their preferences before
making a recommendation."""

# Opt-in prompt variants for exploratory runs; each continues the closing
# elicitation paragraph. No variant renders the frozen prompt unchanged.
_ASK_REQUIREMENTS = (
    " The user may have requirements they have not stated yet; ask about"
    " them before you recommend a {item_noun}."
)
_ASK_AND_VERIFY = _ASK_REQUIREMENTS + (
    " Before you call recommend, propose the {item_noun} you intend to recommend,"
    " with its candidate marker, and wait for the user's verdict."
)
AGENT_PROMPT_VARIANTS = {
    "ask-requirements-v1": _ASK_REQUIREMENTS,
    "ask-and-verify-v1": _ASK_AND_VERIFY,
    # Ask-and-verify plus what the symbolic user's replies mean.
    "ask-verify-protocol-v1": _ASK_AND_VERIFY + (
        " If the user would rather not say (more) about a field, they have a requirement on that"
        " field that they will not state. A rejected {item_noun} fails at least one of the user's"
        " requirements."
    ),
}
# An opt-in variant of the books keyword arm's tool line for exploratory runs:
# it names the fields the structured arm's line lists, leaving the tool as is.
KEYWORD_TOOL_VARIANTS = {
    "keyword-field-list-v1": (
        " Book records list genres, languages, countries, publication year, and formats."
    ),
}

BOOKS_CANDIDATE_PROTOCOL = """\


When you discuss one concrete catalog candidate with the user before your final
recommend() call, append its exact catalog ID as [[candidate_id:ID]] immediately
after the title. Include exactly one candidate_id marker: it identifies the one
book you currently want the user to assess. This marker is not a final
recommendation and does not replace the required recommend(item_id) tool call.
Do not use a candidate marker for an abstention. If an exhaustive search finds
no catalog match, explain that conclusion and finalize with
recommend(item_id=null) under the written policy; the user cannot verify
catalog exhaustiveness for you.
"""


def user_id_instruction(user_id: str | None, spec: DomainSpec = MOVIES) -> str:
    return (
        f"The current user's ID is {user_id}. Use this when checking their {spec.history_noun}."
        if user_id
        else ""
    )


def build_agent_system_prompt(
    policy: str, user_id: str | None, no_tools: bool = False, spec: DomainSpec = MOVIES,
    mode: str = "tools", retrieval_mode: str = "keyword",
    structured_profile: str = DEFAULT_STRUCTURED_PROFILE,
    prompt_variant: str | None = None,
) -> str:
    """The system prompt exactly as `tau-rec run` sends it.

    One place to render it, because the rendered text is scored content: the
    goldens in tests/test_goldens.py pin its bytes, and a run and a dry run
    that formatted it separately could drift apart without either noticing.

    `no_tools` is the legacy ablation prompt, kept so its golden still renders;
    `mode="memory"` is its replacement and takes the memory-mode policy text.
    `prompt_variant` names an entry of AGENT_PROMPT_VARIANTS (tools mode only)
    or of KEYWORD_TOOL_VARIANTS (the books keyword arm only).
    """
    if prompt_variant is not None and (
        mode != "tools" or no_tools
        or prompt_variant not in {**AGENT_PROMPT_VARIANTS, **KEYWORD_TOOL_VARIANTS}
        or (prompt_variant in KEYWORD_TOOL_VARIANTS and (spec.id != "books" or retrieval_mode != "keyword"))
    ):
        raise ValueError(f"unknown or unsupported agent prompt variant {prompt_variant!r}")
    if mode == "memory":
        return AGENT_MEMORY_SYSTEM_PROMPT_TEMPLATE.format(
            policy=policy,
            item_noun=spec.item_noun,
            availability_singular=spec.availability_singular,
            rating_note=spec.rating_note,
        )
    if retrieval_mode not in {"keyword", "structured"}:
        raise ValueError(f"unknown retrieval mode {retrieval_mode!r}")
    if retrieval_mode == "structured" and spec.id != "books":
        raise ValueError("structured retrieval is currently defined only for books")
    if retrieval_mode == "structured":
        if structured_profile not in STRUCTURED_PROFILES:
            raise ValueError(f"unknown structured retrieval profile {structured_profile!r}")
        search_tool_instruction = (
            "search_catalog(genres, languages, countries, min/max publication year, "
            "formats, sort_by, offset): Filter the book catalog using typed metadata."
        )
    else:
        search_tool_instruction = (
            f"search_catalog(query): Search the {spec.item_noun} catalog by keywords."
        )
        if prompt_variant in KEYWORD_TOOL_VARIANTS:
            search_tool_instruction += KEYWORD_TOOL_VARIANTS[prompt_variant]
    template = AGENT_NO_TOOLS_SYSTEM_PROMPT_TEMPLATE if no_tools else AGENT_SYSTEM_PROMPT_TEMPLATE
    recommend_rule = BOOKS_RECOMMEND_RULE if spec.id == "books" else RECOMMEND_RULE
    rendered = template.format(
        policy=policy,
        user_id_instruction=user_id_instruction(user_id, spec),
        item_noun=spec.item_noun,
        availability_kind=spec.availability_kind,
        availability_argument=spec.tool_availability_argument,
        history_contents=spec.history_contents,
        rating_note=spec.rating_note,
        search_tool_instruction=search_tool_instruction,
        recommend_rule=recommend_rule.format(item_noun=spec.item_noun),
    )
    if prompt_variant in AGENT_PROMPT_VARIANTS:
        rendered += AGENT_PROMPT_VARIANTS[prompt_variant].format(item_noun=spec.item_noun)
    if spec.id == "books" and not no_tools:
        rendered += BOOKS_CANDIDATE_PROTOCOL
    return rendered


def _empty_chat_reply(response) -> bool:
    message = response.choices[0].message
    return not (message.content or "").strip() and not getattr(message, "tool_calls", None)


def _empty_responses_reply(response) -> bool:
    output = getattr(response, "output", []) or []
    return not (getattr(response, "output_text", None) or "").strip() and not any(
        getattr(item, "type", None) == "function_call" for item in output
    )


class LiteLLMAgent(BaseAgent):
    def __init__(
        self, model: str, system_prompt: str, tool_definitions: list[dict],
        reasoning_effort: str | None = None, provider_order: list[str] | None = None,
        harness: HarnessProfile = LEGACY,
    ) -> None:
        self.model = model
        self._harness = harness
        self._system_prompt = system_prompt
        self._tool_definitions = tool_definitions
        self._reasoning_effort = reasoning_effort
        self._provider_order = list(provider_order) if provider_order else None
        self._message_history: list[dict] = []
        self._responses_history: list = []
        self._call_log: list[dict] = []

    @property
    def call_log(self) -> list[dict]:
        return [dict(row) for row in self._call_log]

    def reset(self) -> None:
        self._message_history = []
        self._responses_history = []
        self._call_log = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the internal history."""
        history = (
            self._responses_history if _uses_responses_api(self.model)
            else self._message_history
        )
        history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Inject a pre-generated assistant message (e.g., a hardcoded opener)."""
        history = (
            self._responses_history if _uses_responses_api(self.model)
            else self._message_history
        )
        history.append({"role": "assistant", "content": content})

    async def _complete_responses(self, kwargs: dict):
        import litellm

        for attempt in range(MAX_RETRIES):
            started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
            call_start = time.perf_counter()
            try:
                from tau_rec.cost import check_provider_budget
                check_provider_budget()
                response = await timed_call(litellm.aresponses(**kwargs), self._harness.call_timeout_seconds)
                from tau_rec.cost import call_cost, record_usage, response_usage
                record_usage("agent", response)
                usage = response_usage(response)
                hidden_value = getattr(response, "_hidden_params", {}) or {}
                hidden = hidden_value if isinstance(hidden_value, dict) else {}
                self._call_log.append({
                    "attempt": attempt + 1,
                    "status": "ok",
                    "started_at": started_at,
                    "latency_seconds": time.perf_counter() - call_start,
                    "requested_model": self.model,
                    "response_model": getattr(response, "model", None),
                    "response_id": getattr(response, "id", None),
                    "system_fingerprint": (
                        getattr(response, "system_fingerprint", None)
                        or hidden.get("system_fingerprint")
                    ),
                    "provider": hidden.get("custom_llm_provider", "openai"),
                    "provider_model_id": hidden.get("model_id"),
                    "request_parameters": {
                        "api": "responses",
                        "temperature": "omitted",
                        "top_p": "omitted",
                        "seed": "omitted",
                        "reasoning_effort": self._reasoning_effort,
                        "thinking": None,
                        "timeout_seconds": kwargs["timeout"],
                    },
                    "tokens": usage,
                    "provider_reported_cost_usd": hidden.get("response_cost"),
                    "estimated_cost_usd": call_cost(self.model, usage),
                })
                if self._harness.log_finish_reason:
                    incomplete = getattr(response, "incomplete_details", None)
                    self._call_log[-1]["finish_reason"] = (
                        getattr(incomplete, "reason", None) or getattr(response, "status", None)
                    )
                break
            except CallTimeout as error:
                retrying = attempt < MAX_RETRIES - 1
                self._call_log.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "started_at": started_at,
                    "latency_seconds": time.perf_counter() - call_start,
                    "requested_model": self.model,
                    "error_type": "TimeoutError",
                    "error": str(error),
                    "retrying": retrying,
                })
                if not retrying:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
            except litellm.ContextWindowExceededError:
                self._call_log.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "started_at": started_at,
                    "latency_seconds": time.perf_counter() - call_start,
                    "requested_model": self.model,
                    "error_type": "ContextWindowExceededError",
                    "retrying": False,
                })
                raise
            except Exception as error:
                retryable, short_delay = _retryable_provider_error(error)
                retrying = attempt < MAX_RETRIES - 1 and retryable
                self._call_log.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "started_at": started_at,
                    "latency_seconds": time.perf_counter() - call_start,
                    "requested_model": self.model,
                    "error_type": type(error).__name__,
                    "error": str(error)[:300],
                    "retrying": retrying,
                })
                if retrying:
                    await asyncio.sleep(
                        2 * (attempt + 1)
                        if short_delay else RETRY_DELAY * (attempt + 1)
                    )
                else:
                    raise

        return response

    async def _complete_chat(self, kwargs: dict):
        import litellm

        for attempt in range(MAX_RETRIES):
            started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
            call_start = time.perf_counter()
            try:
                from tau_rec.cost import check_provider_budget
                check_provider_budget()
                response = await timed_call(litellm.acompletion(**kwargs), self._harness.call_timeout_seconds)
                from tau_rec.cost import call_cost, record_usage, response_usage
                record_usage("agent", response)
                usage = response_usage(response)
                hidden_value = getattr(response, "_hidden_params", {}) or {}
                hidden = hidden_value if isinstance(hidden_value, dict) else {}
                self._call_log.append({
                    "attempt": attempt + 1,
                    "status": "ok",
                    "started_at": started_at,
                    "latency_seconds": time.perf_counter() - call_start,
                    "requested_model": self.model,
                    "response_model": getattr(response, "model", None),
                    "response_id": getattr(response, "id", None),
                    "system_fingerprint": (
                        getattr(response, "system_fingerprint", None)
                        or hidden.get("system_fingerprint")
                    ),
                    "provider": hidden.get("custom_llm_provider"),
                    "provider_model_id": hidden.get("model_id"),
                    "provider_order": self._provider_order,
                    "served_provider": getattr(response, "provider", None),
                    "request_parameters": {
                        "temperature": kwargs.get("temperature", "omitted"),
                        "top_p": kwargs.get("top_p", "omitted"),
                        "seed": kwargs.get("seed", "omitted"),
                        "reasoning_effort": kwargs.get("reasoning_effort"),
                        "thinking": kwargs.get("thinking"),
                        "extra_body": kwargs.get("extra_body"),
                        "timeout_seconds": kwargs["timeout"],
                        # Present only when sent, so other models' rows are unchanged.
                        **({"output_config": kwargs["output_config"]} if "output_config" in kwargs else {}),
                    },
                    "tokens": usage,
                    "provider_reported_cost_usd": hidden.get("response_cost"),
                    "estimated_cost_usd": call_cost(self.model, usage),
                })
                if self._harness.log_finish_reason:
                    self._call_log[-1]["finish_reason"] = getattr(
                        response.choices[0], "finish_reason", None
                    )
                break
            except CallTimeout as error:
                retrying = attempt < MAX_RETRIES - 1
                self._call_log.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "started_at": started_at,
                    "latency_seconds": time.perf_counter() - call_start,
                    "requested_model": self.model,
                    "error_type": "TimeoutError",
                    "error": str(error),
                    "retrying": retrying,
                })
                if not retrying:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
            except litellm.ContextWindowExceededError:
                self._call_log.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "started_at": started_at,
                    "latency_seconds": time.perf_counter() - call_start,
                    "requested_model": self.model,
                    "error_type": "ContextWindowExceededError",
                    "retrying": False,
                })
                raise
            except Exception as e:
                retryable, short_delay = _retryable_provider_error(e)
                retrying = attempt < MAX_RETRIES - 1 and retryable
                self._call_log.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "started_at": started_at,
                    "latency_seconds": time.perf_counter() - call_start,
                    "requested_model": self.model,
                    "error_type": type(e).__name__,
                    "error": str(e)[:300],
                    "retrying": retrying,
                })
                if retrying:
                    await asyncio.sleep(
                        2 * (attempt + 1)
                        if short_delay else RETRY_DELAY * (attempt + 1)
                    )
                else:
                    raise
        return response

    async def _respond_via_responses(
        self, tool_results: list[dict] | None = None
    ) -> AgentResponse:
        """Run one direct OpenAI Responses turn, preserving reasoning items."""
        import litellm

        if tool_results:
            self._responses_history.extend({
                "type": "function_call_output",
                "call_id": result["tool_call_id"],
                "output": result["content"],
            } for result in tool_results)

        kwargs = {
            "model": self.model,
            "instructions": self._system_prompt,
            "input": list(self._responses_history),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "timeout": 120,
        }
        if self._tool_definitions:
            kwargs["tools"] = _responses_tool_definitions(self._tool_definitions)
        if self._reasoning_effort:
            kwargs["reasoning"] = {"effort": self._reasoning_effort}
        if self._harness.agent_max_tokens:
            kwargs["max_output_tokens"] = self._harness.agent_max_tokens

        response = await self._complete_responses(kwargs)
        for sample in range(self._harness.empty_reply_resamples):
            if not _empty_responses_reply(response):
                break
            self._call_log[-1]["status"] = "empty"
            await asyncio.sleep(2 * (sample + 1))
            response = await self._complete_responses(kwargs)

        # The official stateless flow appends all response items, including
        # encrypted reasoning, before sending any function_call_output items.
        output_items = list(getattr(response, "output", []) or [])
        self._responses_history.extend(output_items)
        tool_calls = []
        for item in output_items:
            if getattr(item, "type", None) != "function_call":
                continue
            arguments = getattr(item, "arguments", "{}")
            tool_calls.append({
                "id": item.call_id,
                "name": item.name,
                "arguments": (
                    json.loads(arguments) if isinstance(arguments, str)
                    else dict(arguments)
                ),
            })
        return AgentResponse(
            message=getattr(response, "output_text", None),
            tool_calls=tool_calls,
        )

    async def respond(self, tool_results: list[dict] | None = None) -> AgentResponse:
        import litellm

        _ensure_pinned_models()

        if _uses_responses_api(self.model):
            return await self._respond_via_responses(tool_results)

        # Build messages: system + history
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._message_history)

        if tool_results:
            # Track tool results in history so future calls see them
            self._message_history.extend(tool_results)
            messages.extend(tool_results)

        if _marks_cache_explicitly(self.model):
            messages = _with_cache_breakpoints(messages)

        kwargs = {"model": self.model, "messages": messages}
        # Some models (o-series, gpt-5, Claude 4.7+) don't support temperature
        if not _rejects_temperature(self.model):
            kwargs["temperature"] = 0.0
        if self._tool_definitions:
            kwargs["tools"] = self._tool_definitions
        if "deepseek" in self.model.lower():
            thinking_mode = "enabled" if self._reasoning_effort else "disabled"
            # litellm refuses a top-level `thinking` for OpenRouter models its
            # param map does not list, failing every call before it is sent.
            if not self.model.startswith("openrouter/"):
                kwargs["thinking"] = {"type": thinking_mode}
            kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}
        # Legacy Qwen3 leaderboard runs used the non-thinking mode. Preserve
        # that behavior when no effort is requested, but allow newer Qwen3.x
        # models to participate in explicitly labelled reasoning experiments.
        if _disables_qwen_thinking(self.model, self._reasoning_effort):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if self._reasoning_effort:
            if _direct_output_config_effort(self.model):
                kwargs["output_config"] = {"effort": self._reasoning_effort}
            elif _supports_reasoning_effort(self.model):
                kwargs["reasoning_effort"] = self._reasoning_effort
            else:
                # OpenRouter's own `reasoning` field is the only way to steer
                # these. litellm forwards extra_body verbatim. A model may
                # accept a level it does not actually honour, and single
                # requests are far too noisy to tell the difference — reasoning
                # length varies severalfold on identical input. Compare token
                # totals across whole runs before labelling one with a level.
                kwargs.setdefault("extra_body", {})["reasoning"] = {"effort": self._reasoning_effort}
        routing = provider_routing(self._provider_order)
        if routing and self.model.startswith("openrouter/"):
            kwargs.setdefault("extra_body", {})["provider"] = routing
        # Per-request timeout to prevent hung OpenRouter requests
        kwargs["timeout"] = 120
        if self._harness.agent_max_tokens:
            kwargs["max_tokens"] = self._harness.agent_max_tokens

        response = await self._complete_chat(kwargs)
        for sample in range(self._harness.empty_reply_resamples):
            if not _empty_chat_reply(response):
                break
            self._call_log[-1]["status"] = "empty"
            await asyncio.sleep(2 * (sample + 1))
            response = await self._complete_chat(kwargs)
        choice = response.choices[0].message

        # Track assistant response in history
        assistant_msg = {"role": "assistant", "content": choice.content or ""}
        # Preserve reasoning_content for DeepSeek reasoning models (required in multi-turn history)
        if hasattr(choice, "reasoning_content") and choice.reasoning_content:
            assistant_msg["reasoning_content"] = choice.reasoning_content
        # Claude binds thinking blocks to the conversation: send them back
        # unchanged so later turns keep the earlier reasoning.
        if _marks_cache_explicitly(self.model) and getattr(choice, "thinking_blocks", None):
            assistant_msg["thinking_blocks"] = choice.thinking_blocks
        if hasattr(choice, "tool_calls") and choice.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in choice.tool_calls
            ]
        self._message_history.append(assistant_msg)

        tool_calls = []
        if hasattr(choice, "tool_calls") and choice.tool_calls:
            for tc in choice.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })
        return AgentResponse(
            message=choice.content,
            tool_calls=tool_calls,
        )
