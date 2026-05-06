from __future__ import annotations
import time
from tau_rec.agents.base import BaseAgent
from tau_rec.simulator.user_simulator import UserSimulator
from tau_rec.environment.tools import ToolKit
from tau_rec.data_model.conversation import (
    ConversationTrace, Message, Role, ToolCall, StopReason,
)

OPENING_GREETING = (
    "Hi! I can help you find a movie to watch. "
    "What kind of movie are you in the mood for?"
)


class Orchestrator:
    def __init__(
        self,
        agent: BaseAgent,
        simulator,  # UserSimulator or mock
        toolkit: ToolKit,
        max_turns: int = 20,
    ) -> None:
        self._agent = agent
        self._simulator = simulator
        self._toolkit = toolkit
        self._max_turns = max_turns

    async def run(self, task_id: str, model: str, trial: int) -> ConversationTrace:
        trace = ConversationTrace(task_id=task_id, model=model, trial=trial)
        trial_start = time.perf_counter()
        conversation: list[Message] = []
        tool_results: list[dict] | None = None

        # Seed with a hardcoded greeting so the user opens with their request.
        greeting = Message(role=Role.AGENT, content=OPENING_GREETING)
        trace.add_message(greeting)
        conversation.append(greeting)
        if hasattr(self._agent, "add_assistant_message"):
            self._agent.add_assistant_message(OPENING_GREETING)

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
            try:
                agent_response = await self._agent.respond(tool_results)
            except Exception as e:
                if "context" in str(e).lower() and "length" in str(e).lower():
                    trace.stop_reason = StopReason.TIMEOUT
                    trace.trial_runtime_s = time.perf_counter() - trial_start
                    return trace
                raise
            tool_results = None
            recommended_this_turn = False

            # Handle tool calls
            while agent_response.has_tool_calls:
                for tc in agent_response.tool_calls:
                    result = self._toolkit.call(tc["name"], tc["arguments"])
                    trace.add_tool_call(ToolCall(
                        name=tc["name"],
                        arguments=tc["arguments"],
                        result=result,
                    ))
                    # recommend() is always terminal (with or without item_id).
                    # Calling with a valid item_id registers the recommendation;
                    # calling with no/null item_id is an explicit abstention.
                    if tc["name"] == "recommend":
                        item_id = tc["arguments"].get("item_id")
                        if item_id:
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
                try:
                    agent_response = await self._agent.respond(tool_results)
                except Exception as e:
                    if "context" in str(e).lower() and "length" in str(e).lower():
                        trace.stop_reason = StopReason.TIMEOUT
                        trace.agent_step_latencies_s.append(time.perf_counter() - step_start)
                        trace.trial_runtime_s = time.perf_counter() - trial_start
                        return trace
                    raise
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
        trace.stop_reason = StopReason.TIMEOUT
        trace.trial_runtime_s = time.perf_counter() - trial_start
        return trace
