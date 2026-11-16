"""One τ-Rec trial per Inspect sample, run by the native ``Orchestrator``.

The movies-v1 content is loaded once per process (``environment()``) and the
per-task ``ToolKit`` and system prompt are built once per task id, as
``tau-rec run`` does (``cli.py``, full-tools mode, keyword retrieval). The
solver builds a fresh agent, simulator, orchestrator and ``ConversationTrace``
for every sample epoch.

The trace lives in the sample store under ``TRACE_KEY`` (its model dump) from
before the first model call, and is refreshed before every agent and simulator
call and when the trial ends, so an interrupted trial still leaves its partial
trace for the scorer. The conversation is mirrored into ``state.messages`` at
the same points for the log viewer.
"""
from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolCall as InspectToolCall
from inspect_ai.util import LimitExceededError, message_limit

from tau_rec.agents.litellm_agent import build_agent_system_prompt
from tau_rec.catalog.search import CatalogSearch
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.conversation import (
    AgentToolTextEvent,
    ConversationTrace,
    MessageEvent,
    Role,
    StopReason,
    ToolCallEvent,
)
from tau_rec.data_model.task import Task
from tau_rec.domain import DomainSpec
from tau_rec.environment.tools import ToolKit
from tau_rec.evaluator.evaluator import CombinedEvaluator
from tau_rec.harness import HarnessProfile
from tau_rec.inspect_support.agent import InspectAgent
from tau_rec.inspect_support.simulator import InspectSimulator, user_model
from tau_rec.orchestrator.orchestrator import Orchestrator
from tau_rec.track import Track, load_track

TRACE_KEY = "tau_rec_trace"

# The native full-tools configuration the leaderboard ran under.
MODE = "tools"
RETRIEVAL_MODE = "keyword"


@dataclass
class MoviesEnvironment:
    """The track's content, loaded once; per-task tool kits built on demand."""

    track: Track
    catalog: Catalog
    search: CatalogSearch
    evaluator: CombinedEvaluator
    policy_text: str
    tasks: dict[str, Task]
    _setups: dict[str, tuple[ToolKit, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, data_root: Path) -> MoviesEnvironment:
        track = load_track(data_root)
        spec = track.spec
        catalog = Catalog.from_json(str(track.require_catalog()), spec)
        task_files = sorted(track.tasks_path.glob("*.json"))
        tasks = {task.id: task for task in (Task.from_json(str(p)) for p in task_files)}
        return cls(
            track=track,
            catalog=catalog,
            search=CatalogSearch(catalog, spec, profile=track.search_profile),
            evaluator=CombinedEvaluator(catalog, spec),
            policy_text=track.policy_for(MODE).read_text(),
            tasks=tasks,
        )

    @property
    def spec(self) -> DomainSpec:
        return self.track.spec

    @property
    def harness(self) -> HarnessProfile:
        return self.track.harness

    def setup(self, task_id: str) -> tuple[ToolKit, str]:
        """The task's ``ToolKit`` and agent system prompt (stateless; shared by epochs)."""
        if task_id not in self._setups:
            task = self.tasks[task_id]
            toolkit = ToolKit(
                catalog=self.catalog,
                search=self.search,
                user_histories=task.user_history or {},
                spec=self.spec,
                mode=MODE,
                retrieval_mode=RETRIEVAL_MODE,
                structured_profile=self.track.structured_profile,
            )
            system_prompt = build_agent_system_prompt(
                self.policy_text,
                task.user_id,
                spec=self.spec,
                mode=MODE,
                retrieval_mode=RETRIEVAL_MODE,
                structured_profile=self.track.structured_profile,
            )
            self._setups[task_id] = (toolkit, system_prompt)
        return self._setups[task_id]


@functools.cache
def environment() -> MoviesEnvironment:
    """The movies-v1 environment from the repository's ``data/``.

    The root comes from the task module (``repo_root()``), which raises a clear
    error when the repository is not an editable ``uv sync`` install.
    """
    # Imported here: the task module imports this one.
    from tau_rec.inspect_task import repo_root

    return MoviesEnvironment.load(repo_root() / "data")


def trace_messages(trace: ConversationTrace) -> list[ChatMessage]:
    """The trace as Inspect chat messages, in trace order.

    Agent and user turns become assistant and user messages. Each tool call
    becomes an assistant message carrying the call, followed by its
    ``ToolKit`` result as a tool message. Text the agent sent alongside tool
    calls (books harness only) is the content of the next tool-call message.
    """
    messages: list[ChatMessage] = []
    pending_text = ""
    calls = 0
    for event in trace.events:
        if isinstance(event, MessageEvent):
            message_type = ChatMessageAssistant if event.role == Role.AGENT else ChatMessageUser
            messages.append(message_type(content=event.content))
        elif isinstance(event, AgentToolTextEvent):
            pending_text = event.content
        elif isinstance(event, ToolCallEvent):
            calls += 1
            call_id = f"tau_rec_call_{calls}"
            messages.append(ChatMessageAssistant(
                content=pending_text,
                tool_calls=[InspectToolCall(
                    id=call_id, function=event.name, arguments=event.arguments
                )],
            ))
            messages.append(ChatMessageTool(
                content=event.result, tool_call_id=call_id, function=event.name
            ))
            pending_text = ""
    return messages


def _before(call: Callable[..., Awaitable[Any]], hook: Callable[[], None]):
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        hook()
        return await call(*args, **kwargs)

    return wrapped


@solver
def tau_rec_solver(max_turns: int = 20) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        env = environment()
        task = env.tasks[str(state.sample_id)]
        toolkit, system_prompt = env.setup(task.id)

        active = get_model()
        model_name = str(active)
        trial = state.epoch - 1  # native trials count from 0
        trace = ConversationTrace(task_id=task.id, model=model_name, trial=trial)

        def sync() -> None:
            state.store.set(TRACE_KEY, trace.model_dump(mode="json"))
            # A view of the trace: the sample's message limit applies to the
            # model calls (Inspect checks each call's input), not to this.
            with message_limit(None):
                state.messages = [
                    ChatMessageSystem(content=system_prompt), *trace_messages(trace)
                ]

        agent = InspectAgent(
            active, system_prompt, toolkit.tool_definitions(), harness=env.harness
        )
        simulator = InspectSimulator(user_model(), task, spec=env.spec, harness=env.harness)
        agent.respond = _before(agent.respond, sync)  # type: ignore[method-assign]
        simulator.respond = _before(simulator.respond, sync)  # type: ignore[method-assign]
        orchestrator = Orchestrator(
            agent=agent,
            simulator=simulator,
            toolkit=toolkit,
            max_turns=max_turns,
            spec=env.spec,
            trial_timeout_seconds=None,
            harness=env.harness,
        )

        sync()
        try:
            await orchestrator.run(task.id, model_name, trial, trace=trace)
        except LimitExceededError:
            # An Inspect limit interrupted the trial: score the partial trace
            # as the native harness scores any early stop.
            trace.stop_reason = StopReason.TIMEOUT
            raise
        finally:
            sync()
        return state

    return solve
