from __future__ import annotations
import asyncio
import json
import time
from tau_rec.agents.base import BaseAgent
from tau_rec.simulator.user_simulator import UserSimulator
from tau_rec.environment.tools import ToolKit
from tau_rec.data_model.conversation import (
    ABSTAIN_SENTINELS, ConversationTrace, Message, Role, ToolCall, StopReason,
)
from tau_rec.domain import MOVIES, DomainSpec
from tau_rec.harness import LEGACY, HarnessProfile

OPENING_GREETING = MOVIES.greeting


def _is_abstention(value) -> bool:
    normalized = "" if value is None else str(value).strip().lower()
    return not normalized or normalized in ABSTAIN_SENTINELS


class _Stop(Exception):
    """Ends a trial early; the reason is mapped through the harness profile."""

    def __init__(self, reason: StopReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


class Orchestrator:
    def __init__(
        self,
        agent: BaseAgent,
        simulator,  # UserSimulator or mock
        toolkit: ToolKit,
        max_turns: int = 20,
        spec: DomainSpec = MOVIES,
        trial_timeout_seconds: float | None = None,
        harness: HarnessProfile = LEGACY,
    ) -> None:
        self._agent = agent
        self._simulator = simulator
        self._toolkit = toolkit
        self._max_turns = max_turns
        self._greeting = spec.greeting
        self._trial_timeout_seconds = trial_timeout_seconds
        self._harness = harness
        self._agent_seconds = 0.0

    def _reason(self, reason: StopReason) -> StopReason:
        """The legacy harness records every early stop as "timeout"."""
        return reason if self._harness.detailed_stop_reasons else StopReason.TIMEOUT

    def _finish(self, trace: ConversationTrace, trial_start: float, reason: StopReason) -> ConversationTrace:
        trace.stop_reason = self._reason(reason)
        trace.trial_runtime_s = time.perf_counter() - trial_start
        return trace

    def _is_context_overflow(self, error: Exception) -> bool:
        message = str(error).lower()
        if not self._harness.detailed_stop_reasons:
            return "context" in message and "length" in message
        return type(error).__name__ == "ContextWindowExceededError" or (
            "context" in message and any(word in message for word in ("length", "window", "too long"))
        )

    async def _respond(self, tool_results: list[dict] | None):
        """One agent call. Under an agent-time cap only this time is counted."""
        agent_cap = (
            self._trial_timeout_seconds if self._harness.cap_counts_agent_time_only else None
        )
        try:
            if agent_cap is None:
                return await self._agent.respond(tool_results)
            remaining = agent_cap - self._agent_seconds
            if remaining <= 0:
                raise _Stop(StopReason.WALL_CAP)
            started = time.perf_counter()
            try:
                return await asyncio.wait_for(self._agent.respond(tool_results), timeout=remaining)
            except asyncio.TimeoutError:
                raise _Stop(StopReason.WALL_CAP) from None
            finally:
                self._agent_seconds += time.perf_counter() - started
        except _Stop:
            raise
        except Exception as error:
            if self._is_context_overflow(error):
                raise _Stop(StopReason.CONTEXT_OVERFLOW) from error
            raise

    async def run(self, task_id: str, model: str, trial: int) -> ConversationTrace:
        trace = ConversationTrace(task_id=task_id, model=model, trial=trial)
        trial_start = time.perf_counter()
        self._agent_seconds = 0.0
        if self._trial_timeout_seconds is None or self._harness.cap_counts_agent_time_only:
            return await self._run_trace(trace, trial_start)
        try:
            return await asyncio.wait_for(
                self._run_trace(trace, trial_start),
                timeout=self._trial_timeout_seconds,
            )
        except asyncio.TimeoutError:
            trace.stop_reason = StopReason.TIMEOUT
            trace.trial_runtime_s = time.perf_counter() - trial_start
            return trace

    async def _run_trace(
        self,
        trace: ConversationTrace,
        trial_start: float,
    ) -> ConversationTrace:
        conversation: list[Message] = []
        tool_results: list[dict] | None = None

        # Seed with a hardcoded greeting so the user opens with their request.
        greeting = Message(role=Role.AGENT, content=self._greeting)
        trace.add_message(greeting)
        conversation.append(greeting)
        if hasattr(self._agent, "add_assistant_message"):
            self._agent.add_assistant_message(self._greeting)

        user_msg = await self._simulator.respond(conversation)
        trace.add_message(user_msg)
        conversation.append(user_msg)
        if hasattr(self._agent, "add_user_message"):
            self._agent.add_user_message(user_msg.content)

        step_start: float | None = None
        for turn in range(self._max_turns):
            # Agent responds. A "step" spans from the prior user message to the
            # agent's next user-facing message (or terminal recommend), including
            # any tool-call loop in between. step_start is only reset when the
            # previous step closed (message emitted or recommend()).
            if step_start is None:
                step_start = time.perf_counter()
                step_tool_calls = budget_errors = 0
            try:
                agent_response = await self._respond(tool_results)
            except _Stop as stop:
                return self._finish(trace, trial_start, stop.reason)
            tool_results = None
            recommended_this_turn = False
            budget = self._harness.max_tool_calls_per_step

            # Handle tool calls
            while agent_response.has_tool_calls:
                if self._harness.log_tool_call_text and agent_response.message:
                    trace.add_agent_tool_text(agent_response.message)
                for tc in agent_response.tool_calls:
                    if budget is not None and step_tool_calls >= budget and tc["name"] != "recommend":
                        # Past the step's budget a call is answered, not executed.
                        budget_errors += 1
                        result = json.dumps({"error": (
                            f"Tool-call budget for this turn is used up ({budget} calls). "
                            "Reply to the user, or call recommend()."
                        )})
                        trace.add_tool_call(ToolCall(name=tc["name"], arguments=tc["arguments"], result=result))
                        tool_results = tool_results or []
                        tool_results.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                        continue
                    step_tool_calls += 1
                    result = self._toolkit.call(tc["name"], tc["arguments"])
                    trace.add_tool_call(ToolCall(
                        name=tc["name"],
                        arguments=tc["arguments"],
                        result=result,
                    ))
                    # recommend() is always terminal (with or without an item).
                    # Calling with a valid item_id registers the recommendation;
                    # calling with no/null item_id is an explicit abstention.
                    if tc["name"] == "recommend":
                        args = tc["arguments"]
                        if getattr(self._toolkit, "mode", "tools") == "memory":
                            # Memory mode names a title. Resolved or not, the
                            # trial ends here: an unresolved title is registered
                            # as given and scores 0, never bounced back for another
                            # try — that would turn the resolver into a catalog
                            # membership oracle.
                            title = args.get("title")
                            if _is_abstention(title):
                                recommended_this_turn = "abstained"
                            else:
                                availability_key = getattr(
                                    self._toolkit, "memory_availability_argument", "platform"
                                )
                                availability = args.get(availability_key, args.get("platform"))
                                res = self._toolkit.resolve_title(str(title), args.get("year"), availability)
                                trace.resolution = res
                                trace.add_recommendation(res.resolved_id or f"unresolved:{title}")
                                recommended_this_turn = "recommended"
                        else:
                            item_id = args.get("item_id")
                            if not _is_abstention(item_id):
                                trace.add_recommendation(item_id)
                                recommended_this_turn = "recommended"
                            else:
                                recommended_this_turn = "abstained"

                    tool_results = tool_results or []
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                if recommended_this_turn:
                    break
                if budget is not None and budget_errors >= self._harness.budget_errors_before_stop:
                    trace.agent_step_latencies_s.append(time.perf_counter() - step_start)
                    return self._finish(trace, trial_start, StopReason.TOOL_BUDGET)
                try:
                    agent_response = await self._respond(tool_results)
                except _Stop as stop:
                    trace.agent_step_latencies_s.append(time.perf_counter() - step_start)
                    return self._finish(trace, trial_start, stop.reason)
                tool_results = None

            # Capture the agent's follow-up text (e.g. "I recommend X because...") if any
            if agent_response.message:
                agent_msg = Message(role=Role.AGENT, content=agent_response.message)
                trace.add_message(agent_msg)
                conversation.append(agent_msg)

            # If the agent committed via recommend(), the trial ends here — the
            # simulator does not get to accept/reject. Policy/constraint scoring
            # still evaluates what the agent recommended.
            if recommended_this_turn:
                trace.agent_step_latencies_s.append(time.perf_counter() - step_start)
                trace.stop_reason = (
                    StopReason.RECOMMENDED
                    if recommended_this_turn == "recommended"
                    else StopReason.ABSTAINED
                )
                trace.trial_runtime_s = time.perf_counter() - trial_start
                return trace

            # No recommendation and no text — keep the step open and skip to next loop iteration
            if not agent_response.message:
                continue

            # Agent step closed by emitting a user-facing message
            trace.agent_step_latencies_s.append(time.perf_counter() - step_start)
            step_start = None

            # Simulator responds. Its ###ACCEPTED###/###REJECTED### tokens are
            # informational only now; only recommend() (or max_turns) ends a trial.
            user_msg = await self._simulator.respond(conversation)
            trace.add_message(user_msg)
            conversation.append(user_msg)

            if hasattr(self._agent, "add_user_message"):
                self._agent.add_user_message(user_msg.content)

        # Hit max turns
        return self._finish(trace, trial_start, StopReason.TURN_LIMIT)
