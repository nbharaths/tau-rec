from __future__ import annotations
import asyncio
import functools
import json
import re
from tau_rec.agents.base import BaseAgent, AgentResponse

MAX_RETRIES = 8
RETRY_DELAY = 30  # seconds

# Anthropic deprecated `temperature` at Claude Opus 4.7 — 4.6 and earlier still
# accept it, later models reject the request outright. Match on the version
# number rather than a list of names: the same model reaches us as
# `claude-opus-4-8` direct and `anthropic/claude-opus-4.8` via OpenRouter, and
# the 5 family omits the minor version entirely (`claude-opus-5`).
_CLAUDE_VERSION = re.compile(r"claude-[a-z]+-(\d+)(?:[-.](\d+))?")
_TEMPERATURE_DEPRECATED_FROM = (4, 7)


@functools.lru_cache(maxsize=None)
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
# every resend at full rate. That difference cost a Sonnet 4.6 calibration 4x
# what the same token volume cost on gpt-5.4.
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


def _rejects_temperature(model: str) -> bool:
    name = model.lower()
    if any(tag in name for tag in ("o1", "o3", "o4", "gpt-5")):
        return True
    match = _CLAUDE_VERSION.search(name)
    if match is None:
        return False
    major, minor = int(match.group(1)), int(match.group(2) or 0)
    return (major, minor) >= _TEMPERATURE_DEPRECATED_FROM

AGENT_SYSTEM_PROMPT_TEMPLATE = """\
You are a movie recommendation assistant. Your goal is to help the user find
a movie they will enjoy.

You have access to the following tools:
- search_catalog(query): Search the movie catalog by keywords.
- get_metadata(item_id): Get full details for a specific movie.
- check_availability(item_id, services): Check streaming availability.
- get_user_history(user_id): Get the user's watch history and ratings.
- recommend(item_id): Make your final decision. THIS ENDS THE CONVERSATION.
  Call with item_id=<movie id> to recommend that movie. Call with no item_id
  (or item_id=null) ONLY when you have exhaustively searched the catalog
  and are confident no movie even approximately satisfies the user's
  requirements. Abstention is a strong statement: "the catalog has no
  answer." On satisfiable tasks, abstaining counts as a failure. When in
  doubt, commit to your best-effort pick rather than abstaining —
  availability uncertainty, a slightly lower rating, or an imperfect genre
  match are NOT grounds for abstention. You MUST call this tool exactly
  once per conversation.

The user's messages (including any "###ACCEPTED###" / "###REJECTED###"
responses) are informational feedback only. They do NOT end the conversation
— only your call to recommend() does. Use rejections as feedback to pick a
better candidate, then call recommend() with your updated pick.

You must follow all policies described below.

<policy>
{policy}
</policy>

{user_id_instruction}

Note on content ratings: "NR" (Not Rated) is NOT the same as "R". Only treat
them as interchangeable if the user explicitly says they are fine with
unrated content.

Engage the user in natural conversation to understand their preferences before
making a recommendation."""

AGENT_NO_TOOLS_SYSTEM_PROMPT_TEMPLATE = """\
You are a movie recommendation assistant. Your goal is to help the user find
a movie they will enjoy. Recommend a specific movie by name.

You must follow all policies described below.

<policy>
{policy}
</policy>

{user_id_instruction}

Note on content ratings: "NR" (Not Rated) is NOT the same as "R". Only treat
them as interchangeable if the user explicitly says they are fine with
unrated content.

Engage the user in natural conversation to understand their preferences before
making a recommendation."""


class LiteLLMAgent(BaseAgent):
    def __init__(self, model: str, system_prompt: str, tool_definitions: list[dict], reasoning_effort: str | None = None) -> None:
        self.model = model
        self._system_prompt = system_prompt
        self._tool_definitions = tool_definitions
        self._reasoning_effort = reasoning_effort
        self._message_history: list[dict] = []

    def reset(self) -> None:
        self._message_history = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the internal history."""
        self._message_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Inject a pre-generated assistant message (e.g., a hardcoded opener)."""
        self._message_history.append({"role": "assistant", "content": content})

    async def respond(self, tool_results: list[dict] | None = None) -> AgentResponse:
        import litellm

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
            kwargs["thinking"] = {"type": thinking_mode}
            kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}
        # Disable extended thinking for Qwen3 via OpenRouter
        if "openrouter" in self.model and "qwen3" in self.model.lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if self._reasoning_effort:
            if _supports_reasoning_effort(self.model):
                kwargs["reasoning_effort"] = self._reasoning_effort
            else:
                # OpenRouter's own `reasoning` field is the only way to steer
                # these. litellm forwards extra_body verbatim. A model may
                # accept a level it does not actually honour, and single
                # requests are far too noisy to tell the difference — reasoning
                # length varies severalfold on identical input. Compare token
                # totals across whole runs before labelling one with a level.
                kwargs.setdefault("extra_body", {})["reasoning"] = {"effort": self._reasoning_effort}
        # Per-request timeout to prevent hung OpenRouter requests
        kwargs["timeout"] = 120

        for attempt in range(MAX_RETRIES):
            try:
                response = await litellm.acompletion(**kwargs)
                from tau_rec.cost import record_usage
                record_usage("agent", response)
                break
            except litellm.ContextWindowExceededError:
                raise
            except Exception as e:
                if attempt < MAX_RETRIES - 1 and ("503" in str(e) or "429" in str(e) or "rate_limit" in str(e).lower() or "overloaded" in str(e).lower() or "unavailable" in str(e).lower() or "connection" in str(e).lower()):
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise
        choice = response.choices[0].message

        # Track assistant response in history
        assistant_msg = {"role": "assistant", "content": choice.content or ""}
        # Preserve reasoning_content for DeepSeek reasoning models (required in multi-turn history)
        if hasattr(choice, "reasoning_content") and choice.reasoning_content:
            assistant_msg["reasoning_content"] = choice.reasoning_content
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
