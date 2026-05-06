from __future__ import annotations
import asyncio
import json
from tau_rec.agents.base import BaseAgent, AgentResponse

MAX_RETRIES = 8
RETRY_DELAY = 30  # seconds

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

        kwargs = {"model": self.model, "messages": messages}
        # Some models (o-series, gpt-5) don't support temperature
        if not any(tag in self.model for tag in ["o1", "o3", "o4", "gpt-5"]):
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
            kwargs["reasoning_effort"] = self._reasoning_effort
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
