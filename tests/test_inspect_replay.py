"""Replay equivalence: the same scripted outputs on the native and Inspect paths (REQ-007, REQ-021).

Each script in ``fixtures/inspect_replay/`` is a list of agent outputs (text
and/or tool calls) and simulator texts. ``recommendation``, ``abstention`` and
``timeout`` were converted from archived g1 traces (v1.2.0 release,
``tau-rec-g1-traces.tar.gz``; see each script's ``source``); the other four are
synthetic failure cases.

Both paths run the unchanged native ``Orchestrator`` and ``CombinedEvaluator``
on the same per-task ``ToolKit`` and system prompt:

- native: ``LiteLLMAgent`` + ``UserSimulator`` (``catalog=catalog``, as
  ``tau-rec run`` builds it in full-tools mode), with ``litellm.acompletion``
  answered from the script;
- Inspect: ``InspectAgent`` + ``InspectSimulator`` on two ``mockllm`` models
  whose ``custom_outputs`` answer from the same script, with the tool kit and
  prompt from the Inspect solver's own ``MoviesEnvironment``.

The replayed traces need not match the archived ones (the archive came from
tau_rec 1.1.0, and tool results are recomputed by the current ``ToolKit``);
the two current paths must agree with each other.

Failure mechanisms, per path:

- ``model_length``: Inspect ``stop_reason="model_length"``; native litellm
  raises ``ContextWindowExceededError``. Both end the trial as ``TIMEOUT``.
- ``content_filter``: Inspect ``stop_reason="content_filter"``; native gets the
  HTTP-400 form (litellm ``ContentPolicyViolationError``). Both raise. (Native
  passes a normal ``finish_reason=content_filter`` through; that asymmetry is
  documented in the spec and not replayed here.)
- ``malformed_recommend``: ``recommend`` with malformed ``{``-prefixed JSON.
  Native ``json.loads`` raises. mockllm has no raw OpenAI-shaped response, so
  the adapter's raw re-parse does not run (documented fallback); instead the
  Inspect ``ToolCall`` is built by Inspect's own ``parse_tool_call`` from the
  raw text, exactly as its OpenAI provider does, which sets ``parse_error``
  with ``{}`` arguments. The adapter must raise on ``parse_error`` rather than
  pass ``{}`` on (an abstention).
- ``empty_simulator``: eight empty or whitespace-only simulator replies; the
  inherited retry loop raises ``RuntimeError`` on both paths.
"""
from __future__ import annotations

import copy
import functools
import json
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

# litellm's model map is frozen (no fetch at import) for the whole session in
# tests/conftest.py, as runs do.

import litellm  # noqa: E402
import pytest  # noqa: E402
from inspect_ai.model import (  # noqa: E402
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.model._call_tools import parse_tool_call  # noqa: E402
from inspect_ai.model._generate_config import active_generate_config_context_var  # noqa: E402
from inspect_ai.model._model import active_model_context_var  # noqa: E402
from inspect_ai.tool import ToolCall as InspectToolCall  # noqa: E402

from tau_rec.agents import litellm_agent  # noqa: E402
from tau_rec.agents.litellm_agent import LiteLLMAgent, build_agent_system_prompt  # noqa: E402
from tau_rec.catalog.search import CatalogSearch  # noqa: E402
from tau_rec.data_model.catalog import Catalog  # noqa: E402
from tau_rec.data_model.conversation import (  # noqa: E402
    ConversationTrace,
    MessageEvent,
    StopReason,
    ToolCallEvent,
)
from tau_rec.data_model.task import Task  # noqa: E402
from tau_rec.environment.tools import ToolKit  # noqa: E402
from tau_rec.evaluator.evaluator import CombinedEvaluator, TrialResult  # noqa: E402
from tau_rec.inspect_support.agent import (  # noqa: E402
    AgentCallError,
    InspectAgent,
    tool_infos,
)
from tau_rec.inspect_support.simulator import InspectSimulator  # noqa: E402
from tau_rec.inspect_support.solver import MODE, RETRIEVAL_MODE, MoviesEnvironment  # noqa: E402
from tau_rec.orchestrator.orchestrator import Orchestrator  # noqa: E402
from tau_rec.simulator import user_simulator  # noqa: E402
from tau_rec.simulator.user_simulator import UserSimulator  # noqa: E402
from tau_rec.track import load_track  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIXTURES = Path(__file__).parent / "fixtures" / "inspect_replay"

# The trace's model label, the same on both paths so whole traces compare.
MODEL_LABEL = "gpt-5-mini"
NATIVE_AGENT_MODEL = "openai/gpt-5-mini"
NATIVE_SIM_MODEL = "openai/gpt-5-mini"
MOCK_MODEL = "mockllm/model"
MAX_TURNS = 20

REPLAY_CASES = ["recommendation", "abstention", "timeout"]
EXPECTED_STOP = {
    "recommendation": StopReason.RECOMMENDED,
    "abstention": StopReason.ABSTAINED,
    # Max turns; the legacy harness records every early stop as TIMEOUT.
    "timeout": StopReason.TIMEOUT,
}
FAILURE_CASES = ["model_length", "content_filter", "malformed_recommend", "empty_simulator"]


def load_case(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


# --------------------------------------------------------------------------- setup


@dataclass
class NativeSetup:
    """The objects ``tau-rec run`` builds for movies-v1 in full-tools mode (cli.py)."""

    track: object
    catalog: Catalog
    search: CatalogSearch
    evaluator: CombinedEvaluator
    policy_text: str
    _setups: dict = field(default_factory=dict)

    @property
    def spec(self):
        return self.track.spec

    @property
    def harness(self):
        return self.track.harness

    def task(self, task_id: str) -> Task:
        return Task.from_json(str(self.track.tasks_path / f"{task_id}.json"))

    def setup(self, task: Task) -> tuple[ToolKit, str]:
        if task.id not in self._setups:
            toolkit = ToolKit(
                catalog=self.catalog,
                search=self.search,
                user_histories=task.user_history or {},
                spec=self.spec,
                mode="tools",
                retrieval_mode="keyword",
                structured_profile=self.track.structured_profile,
            )
            system_prompt = build_agent_system_prompt(
                self.policy_text,
                task.user_id,
                spec=self.spec,
                mode="tools",
                retrieval_mode="keyword",
                structured_profile=self.track.structured_profile,
                prompt_variant=None,
            )
            self._setups[task.id] = (toolkit, system_prompt)
        return self._setups[task.id]


@functools.cache
def native_setup() -> NativeSetup:
    track = load_track(DATA)
    spec = track.spec
    catalog = Catalog.from_json(str(track.require_catalog()), spec)
    return NativeSetup(
        track=track,
        catalog=catalog,
        search=CatalogSearch(catalog, spec, profile=track.search_profile),
        evaluator=CombinedEvaluator(catalog, spec),
        policy_text=track.policy_for("tools").read_text(),
    )


@functools.cache
def inspect_setup() -> MoviesEnvironment:
    """The Inspect solver's own environment, loaded from the same ``data/``."""
    return MoviesEnvironment.load(DATA)


def new_trace(case: dict) -> ConversationTrace:
    return ConversationTrace(task_id=case["task_id"], model=MODEL_LABEL, trial=case["trial"])


# ------------------------------------------------------------------ recorded inputs


@dataclass
class Inputs:
    """What each model was sent, normalized to the same shape on both paths."""

    agent: list[list[tuple]] = field(default_factory=list)
    agent_tools: list[list[str]] = field(default_factory=list)
    simulator: list[list[tuple]] = field(default_factory=list)


def _norm_args(arguments) -> str:
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return json.dumps(arguments, sort_keys=True)


def _native_messages(messages: list[dict]) -> list[tuple]:
    normalized = []
    for message in messages:
        calls = tuple(
            (call["id"], call["function"]["name"], _norm_args(call["function"]["arguments"]))
            for call in message.get("tool_calls") or []
        )
        normalized.append((
            message["role"], message.get("content") or "", calls, message.get("tool_call_id"),
        ))
    return normalized


def _inspect_messages(messages: list[ChatMessage]) -> list[tuple]:
    normalized = []
    for message in messages:
        calls = tuple(
            (call.id, call.function, _norm_args(call.arguments))
            for call in (getattr(message, "tool_calls", None) or [])
        )
        normalized.append((
            message.role,
            message.text or "",
            calls,
            message.tool_call_id if isinstance(message, ChatMessageTool) else None,
        ))
    return normalized


# --------------------------------------------------------------------- native path


CONTEXT_WINDOW_TEXT = "This model's maximum context length is 400000 tokens."
CONTENT_POLICY_TEXT = (
    "The response was filtered due to the prompt triggering content management policy."
)


class NativeScript:
    """``litellm.acompletion`` answered from a script.

    Agent calls carry ``tools``; simulator calls do not.
    """

    def __init__(self, case: dict, inputs: Inputs) -> None:
        self.agent = deque(case["agent"])
        self.simulator = deque(case["simulator"])
        self.inputs = inputs

    async def acompletion(self, **kwargs):
        if "tools" in kwargs:
            return self._agent(kwargs)
        self.inputs.simulator.append(
            [(m["role"], m["content"], (), None) for m in kwargs["messages"]]
        )
        text = self.simulator.popleft()
        message = SimpleNamespace(content=text)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None, model=NATIVE_SIM_MODEL, id="sim",
        )

    def _agent(self, kwargs: dict):
        self.inputs.agent.append(_native_messages(copy.deepcopy(kwargs["messages"])))
        self.inputs.agent_tools.append([t["function"]["name"] for t in kwargs["tools"]])
        step = self.agent.popleft()
        stop = step.get("stop_reason")
        if stop == "model_length":
            raise litellm.ContextWindowExceededError(
                message=CONTEXT_WINDOW_TEXT, model=MODEL_LABEL, llm_provider="openai"
            )
        if stop == "content_filter":
            raise litellm.ContentPolicyViolationError(
                message=CONTENT_POLICY_TEXT, model=MODEL_LABEL, llm_provider="openai"
            )
        calls = [
            SimpleNamespace(
                id=call["id"],
                type="function",
                function=SimpleNamespace(
                    name=call["name"],
                    arguments=(
                        call["raw_arguments"] if "raw_arguments" in call
                        else json.dumps(call["arguments"])
                    ),
                ),
            )
            for call in step["tool_calls"]
        ]
        message = SimpleNamespace(content=step["content"], tool_calls=calls or None)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=message, finish_reason="tool_calls" if calls else "stop",
            )],
            usage=None, model=NATIVE_AGENT_MODEL, id="agent",
        )


async def replay_native(
    case: dict, trace: ConversationTrace | None = None, inputs: Inputs | None = None,
) -> tuple[ConversationTrace, TrialResult]:
    env = native_setup()
    task = env.task(case["task_id"])
    toolkit, system_prompt = env.setup(task)
    agent = LiteLLMAgent(
        model=NATIVE_AGENT_MODEL,
        system_prompt=system_prompt,
        tool_definitions=toolkit.tool_definitions(),
        reasoning_effort=None,
        provider_order=None,
        harness=env.harness,
    )
    simulator = UserSimulator(
        model=NATIVE_SIM_MODEL,
        task=task,
        spec=env.spec,
        catalog=env.catalog,  # cli.py: mode == "tools"
        interaction_mode="llm",
        provider_order=None,
        harness=env.harness,
    )
    orchestrator = Orchestrator(
        agent=agent, simulator=simulator, toolkit=toolkit,
        max_turns=MAX_TURNS, spec=env.spec, harness=env.harness,
    )
    trace = trace if trace is not None else new_trace(case)
    script = NativeScript(case, inputs if inputs is not None else Inputs())
    with patch.object(litellm, "acompletion", script.acompletion):
        await orchestrator.run(case["task_id"], MODEL_LABEL, case["trial"], trace=trace)
    assert not script.agent and not script.simulator, "native replay left script entries unused"
    return trace, env.evaluator.evaluate(task, trace)


# -------------------------------------------------------------------- Inspect path


def inspect_agent_output(step: dict) -> ModelOutput:
    stop = step.get("stop_reason")
    if stop is not None:
        return ModelOutput.from_content(MOCK_MODEL, step["content"] or "", stop_reason=stop)
    if not step["tool_calls"]:
        return ModelOutput.from_content(MOCK_MODEL, step["content"])
    tools = tool_infos(inspect_setup().setup(TASK_FOR_TOOLS)[0].tool_definitions())
    calls = []
    for call in step["tool_calls"]:
        if "raw_arguments" in call:
            # Inspect's own parser, as its OpenAI provider applies it to the raw text.
            calls.append(parse_tool_call(call["id"], call["name"], call["raw_arguments"], tools))
        else:
            calls.append(InspectToolCall(
                id=call["id"], function=call["name"], arguments=call["arguments"]
            ))
    message = ChatMessageAssistant(
        content=step["content"] or "", tool_calls=calls, model=MOCK_MODEL, source="generate",
    )
    return ModelOutput(
        model=MOCK_MODEL,
        choices=[ChatCompletionChoice(message=message, stop_reason="tool_calls")],
    )


# Tool definitions are the same for every movies task in tools mode.
TASK_FOR_TOOLS = "task_004"


class InspectScript:
    """Two mockllm ``custom_outputs`` callables answering from a script."""

    def __init__(self, case: dict, inputs: Inputs) -> None:
        self.agent = deque(case["agent"])
        self.simulator = deque(case["simulator"])
        self.inputs = inputs

    def agent_output(self, input, tools, tool_choice, config) -> ModelOutput:
        self.inputs.agent.append(_inspect_messages(list(input)))
        self.inputs.agent_tools.append([tool.name for tool in tools])
        return inspect_agent_output(self.agent.popleft())

    def simulator_output(self, input, tools, tool_choice, config) -> ModelOutput:
        self.inputs.simulator.append(_inspect_messages(list(input)))
        return ModelOutput.from_content(MOCK_MODEL, self.simulator.popleft())


@contextmanager
def active(model, config: GenerateConfig = GenerateConfig()):
    """Make ``model`` Inspect's active model, as it is inside a sample."""
    model_token = active_model_context_var.set(model)
    config_token = active_generate_config_context_var.set(config)
    try:
        yield
    finally:
        active_generate_config_context_var.reset(config_token)
        active_model_context_var.reset(model_token)


async def replay_inspect(
    case: dict, trace: ConversationTrace | None = None, inputs: Inputs | None = None,
) -> tuple[ConversationTrace, TrialResult]:
    env = inspect_setup()
    task = env.tasks[case["task_id"]]
    toolkit, system_prompt = env.setup(task.id)
    script = InspectScript(case, inputs if inputs is not None else Inputs())
    agent_model = get_model(MOCK_MODEL, custom_outputs=script.agent_output, memoize=False)
    user_model = get_model(MOCK_MODEL, custom_outputs=script.simulator_output, memoize=False)
    agent = InspectAgent(agent_model, system_prompt, toolkit.tool_definitions(), harness=env.harness)
    simulator = InspectSimulator(user_model, task, spec=env.spec, harness=env.harness)
    orchestrator = Orchestrator(
        agent=agent, simulator=simulator, toolkit=toolkit,
        max_turns=MAX_TURNS, spec=env.spec, harness=env.harness,
    )
    trace = trace if trace is not None else new_trace(case)
    with active(agent_model):
        await orchestrator.run(case["task_id"], MODEL_LABEL, case["trial"], trace=trace)
    assert not script.agent and not script.simulator, "Inspect replay left script entries unused"
    return trace, env.evaluator.evaluate(task, trace)


# ------------------------------------------------------------------------ helpers

TIMING_FIELDS = {"agent_step_latencies_s", "trial_runtime_s"}


def trace_doc(trace: ConversationTrace) -> dict:
    doc = trace.model_dump(mode="json", exclude=TIMING_FIELDS)
    doc["n_agent_steps"] = len(trace.agent_step_latencies_s)
    return doc


def result_doc(result: TrialResult) -> dict:
    return result.model_dump(
        mode="json",
        exclude={"efficiency": {
            "agent_step_latencies_s", "mean_agent_step_latency_s", "trial_runtime_s",
        }},
    )


def event_rows(trace: ConversationTrace) -> list[tuple]:
    rows = []
    for event in trace.events:
        if isinstance(event, MessageEvent):
            rows.append(("message", event.role.value, event.content, None, None, None))
        elif isinstance(event, ToolCallEvent):
            rows.append((
                "tool_call", None, None, event.name,
                json.dumps(event.arguments, sort_keys=True), event.result,
            ))
        else:
            rows.append((event.type, None, getattr(event, "content", None), None, None, None))
    return rows


def message_texts(trace: ConversationTrace) -> list[str]:
    return [event.content for event in trace.events if isinstance(event, MessageEvent)]


@pytest.fixture
def no_sleep(monkeypatch):
    """Patch ``asyncio.sleep`` where the native agent and simulator retry loops call it.

    Both modules use the ``asyncio`` module object, so this is one patch seen by
    both (and by ``InspectSimulator``, which inherits the simulator loop).
    """
    sleep = AsyncMock()
    monkeypatch.setattr(litellm_agent.asyncio, "sleep", sleep)
    monkeypatch.setattr(user_simulator.asyncio, "sleep", sleep)
    return sleep


# -------------------------------------------------------------------------- tests


def test_fixtures_are_scripts_only():
    """Only small scripts are committed: no archived trace bodies or tool results."""
    for path in sorted(FIXTURES.glob("*.json")):
        script = json.loads(path.read_text())
        assert set(script) <= {"case", "source", "note", "task_id", "trial", "agent", "simulator"}
        for step in script["agent"]:
            assert set(step) <= {"content", "tool_calls", "stop_reason"}
            for call in step["tool_calls"]:
                assert "result" not in call
        assert path.stat().st_size < 20_000


def test_both_paths_share_the_native_setup():
    """The solver's tool kit and prompt are the ones ``tau-rec run`` builds."""
    native, inspect_env = native_setup(), inspect_setup()
    assert (MODE, RETRIEVAL_MODE) == ("tools", "keyword")
    assert native.harness == inspect_env.harness
    for name in REPLAY_CASES + FAILURE_CASES:
        task_id = load_case(name)["task_id"]
        native_toolkit, native_prompt = native.setup(native.task(task_id))
        inspect_toolkit, inspect_prompt = inspect_env.setup(task_id)
        assert native_prompt == inspect_prompt
        assert native_toolkit.tool_definitions() == inspect_toolkit.tool_definitions()


@pytest.mark.parametrize("name", REPLAY_CASES)
async def test_replay_equivalent(name, no_sleep):
    case = load_case(name)
    native_inputs, inspect_inputs = Inputs(), Inputs()
    native_trace, native_result = await replay_native(case, inputs=native_inputs)
    inspect_trace, inspect_result = await replay_inspect(case, inputs=inspect_inputs)

    # Event sequences: role, content, tool name, arguments, result.
    assert event_rows(inspect_trace) == event_rows(native_trace)
    assert inspect_trace.recommendations == native_trace.recommendations
    assert inspect_trace.stop_reason == native_trace.stop_reason == EXPECTED_STOP[name]
    assert trace_doc(inspect_trace) == trace_doc(native_trace)

    # Scores.
    assert inspect_result.constraint_score == native_result.constraint_score
    assert inspect_result.policy_score == native_result.policy_score
    assert inspect_result.primary_reward == native_result.primary_reward
    assert result_doc(inspect_result) == result_doc(native_result)

    # Both models were sent the same conversation on every call.
    assert inspect_inputs.agent == native_inputs.agent
    assert inspect_inputs.agent_tools == native_inputs.agent_tools
    assert inspect_inputs.simulator == native_inputs.simulator
    assert len(native_inputs.agent) == len(case["agent"])
    assert len(native_inputs.simulator) == len(case["simulator"])
    no_sleep.assert_not_awaited()


def test_replay_cases_cover_the_scripted_outcomes():
    """The recommendation scores, the abstention does not, the timeout runs max turns."""
    timeout = load_case("timeout")
    assert sum(1 for step in timeout["agent"] if step["content"] and not step["tool_calls"]) == MAX_TURNS
    assert any(
        call["name"] == "recommend" and not call["arguments"].get("item_id")
        for step in load_case("abstention")["agent"] for call in step["tool_calls"]
    )


# Text that would mean an error or a stop leaked into the conversation.
LEAK_MARKERS = (
    "stop_reason", "content_filter", "content filter", "model_length", "context length",
    "parse", "Error parsing", "arguments", "Traceback", "Error", "user simulator",
)


def _assert_no_leak(texts: list[str], forbidden: list[str]) -> None:
    for text in texts:
        for marker in [*LEAK_MARKERS, *forbidden]:
            assert marker not in text, f"{marker!r} reached the conversation"


@pytest.mark.parametrize("name", FAILURE_CASES)
async def test_replay_failure_paths(name, no_sleep):
    case = load_case(name)
    native_trace, inspect_trace = new_trace(case), new_trace(case)
    native_inputs, inspect_inputs = Inputs(), Inputs()
    native_error = inspect_error = None
    native_result = inspect_result = None
    try:
        _, native_result = await replay_native(case, native_trace, native_inputs)
    except Exception as error:  # noqa: BLE001 - compared below
        native_error = error
    native_sleeps = [c.args for c in no_sleep.await_args_list]
    no_sleep.reset_mock()
    try:
        _, inspect_result = await replay_inspect(case, inspect_trace, inspect_inputs)
    except Exception as error:  # noqa: BLE001 - compared below
        inspect_error = error
    inspect_sleeps = [c.args for c in no_sleep.await_args_list]

    # The partial traces and what each model was sent agree on both paths.
    assert event_rows(inspect_trace) == event_rows(native_trace)
    assert trace_doc(inspect_trace) == trace_doc(native_trace)
    assert inspect_inputs.agent == native_inputs.agent
    assert inspect_inputs.simulator == native_inputs.simulator
    assert inspect_sleeps == native_sleeps

    if name == "model_length":
        # Context length: the trial ends as a TIMEOUT on both paths, and scores.
        assert native_error is None and inspect_error is None
        assert native_trace.stop_reason == inspect_trace.stop_reason == StopReason.TIMEOUT
        assert result_doc(inspect_result) == result_doc(native_result)
        assert inspect_result.primary_reward == native_result.primary_reward == 0.0
    else:
        # Content filter, parse error and empty replies: the trial errors on both paths.
        assert native_error is not None and inspect_error is not None
        assert native_trace.stop_reason is None and inspect_trace.stop_reason is None

    if name == "content_filter":
        assert isinstance(native_error, litellm.ContentPolicyViolationError)
        assert isinstance(inspect_error, AgentCallError)
        # Inspect retries a content-filter stop no more than native retries the 400.
        assert len(inspect_inputs.agent) == len(native_inputs.agent) == 2
    elif name == "malformed_recommend":
        assert isinstance(native_error, json.JSONDecodeError)
        assert isinstance(inspect_error, AgentCallError)
        assert inspect_error.function == "recommend"
        assert inspect_error.parse_error
        # The mechanism: Inspect's parser yields {} arguments plus parse_error,
        # which would be an abstention if passed on.
        raw = case["agent"][-1]["tool_calls"][-1]["raw_arguments"]
        parsed = parse_tool_call("probe", "recommend", raw, tool_infos(
            inspect_setup().setup(case["task_id"])[0].tool_definitions()
        ))
        assert parsed.parse_error and parsed.arguments == {}
        # No abstention is registered, and no call in the failing response ran.
        for trace in (native_trace, inspect_trace):
            assert trace.stop_reason != StopReason.ABSTAINED
            assert trace.recommendations == []
            assert [e.name for e in trace.events if isinstance(e, ToolCallEvent)] == [
                "get_user_history"
            ]
    elif name == "empty_simulator":
        for error in (native_error, inspect_error):
            assert type(error) is RuntimeError
            assert str(error) == "user simulator returned no text in 8 attempts"
        assert len(inspect_inputs.simulator) == len(native_inputs.simulator) == 1 + 8
        assert inspect_sleeps == [(2 * (attempt + 1),) for attempt in range(8)]

    # No Inspect error text reaches the conversation (MessageEvents) or the
    # simulator as an agent turn.
    forbidden = [str(e) for e in (native_error, inspect_error) if e is not None and str(e)]
    forbidden += [type(e).__name__ for e in (native_error, inspect_error) if e is not None]
    if name == "content_filter":
        forbidden.append(case["agent"][-1]["content"])  # the refusal text
    for trace in (native_trace, inspect_trace):
        _assert_no_leak(message_texts(trace), forbidden)
    for inputs in (native_inputs, inspect_inputs):
        _assert_no_leak([row[1] for call in inputs.simulator for row in call[1:]], forbidden)
