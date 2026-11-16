"""The tau_rec_movies Inspect task: solver, scorer, pass^k, error policy, metadata.

Every eval here is offline: both roles are mockllm and logs go to tmp_path.
"""
from __future__ import annotations

import contextvars
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest
from inspect_ai._util.error import PrerequisiteError
from inspect_ai import Epochs, eval as inspect_eval, eval_retry, task_with
from inspect_ai.log import read_eval_log, resolve_sample_attachments
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.model._model import init_active_model, init_model_roles
from inspect_ai.scorer import Score, scorer
from inspect_ai.scorer import mean as mean_metric
from inspect_ai.scorer._reducer.registry import reducer_log_names
from inspect_ai.solver import solver

import tau_rec
from tau_rec import inspect_task
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.conversation import (
    ConversationTrace,
    MessageEvent,
    Role,
    ToolCallEvent,
)
from tau_rec.data_model.task import Task
from tau_rec.domain import MOVIES
from tau_rec.evaluator.evaluator import CombinedEvaluator
from tau_rec.inspect_support.metadata import comparability_metadata
from tau_rec.inspect_support.solver import TRACE_KEY, environment
from tau_rec.inspect_task import tau_rec_movies
from tau_rec.metrics.pass_k import aggregate_pass_k

ROOT = Path(__file__).resolve().parent.parent
TASK_FILE = "src/tau_rec/inspect_task.py@tau_rec_movies"
G1_ENTRY = ROOT / "leaderboard" / "entries" / "g1-gpt5-mini.json"

# task_001 asks for a comedy on the user's services; this one passes both the
# constraint and the policy checks (recommend_tool, availability).
TASK = "task_001"
PASSING_ITEM = "tmdb_803796"


def _data(*names: str) -> Path:
    return ROOT / "data" / Path(*names)


def _task(task_id: str = TASK) -> Task:
    return Task.from_json(str(_data("tasks", f"{task_id}.json")))


def _recommend(item_id: str | None, call_id: str = "call_rec") -> ModelOutput:
    return ModelOutput.for_tool_call(
        "mockllm/model", "recommend", {"item_id": item_id}, tool_call_id=call_id, content=""
    )


def _tool(name: str, arguments: dict, call_id: str) -> ModelOutput:
    return ModelOutput.for_tool_call(
        "mockllm/model", name, arguments, tool_call_id=call_id, content=""
    )


def _mock(outputs) -> object:
    return get_model("mockllm/model", custom_outputs=outputs, memoize=False)


def _run(task=tau_rec_movies, **kwargs):
    kwargs.setdefault("model_roles", {"user": "mockllm/model"})
    kwargs.setdefault("display", "none")
    (log,) = inspect_eval(task, **kwargs)
    return log


def _one(log):
    assert log.status == "success", log.error
    (sample,) = log.samples
    return resolve_sample_attachments(sample)


def _stored_trace(sample) -> ConversationTrace:
    return ConversationTrace.model_validate(sample.store[TRACE_KEY])


def _in_eval_context(fn, model=None, roles=None):
    """Run ``fn`` with an active model and roles, as Inspect does at task creation."""
    def run():
        init_active_model(model or get_model("mockllm/model"), GenerateConfig())
        init_model_roles(roles if roles is not None else {"user": get_model("mockllm/model")})
        return fn()

    return contextvars.copy_context().run(run)


# --- dataset -------------------------------------------------------------------


def test_dataset_has_60_movies_ids():
    task = _in_eval_context(tau_rec_movies)
    ids = [sample.id for sample in task.dataset]
    expected = sorted(path.stem for path in _data("tasks").glob("*.json"))
    assert len(ids) == 60
    assert ids == expected
    first = next(iter(task.dataset))
    assert first.metadata == {
        "complexity": _task(first.id).complexity,
        "reveal_difficulty": _task(first.id).reveal_difficulty,
    }
    assert task.version == tau_rec.__version__
    assert task.fail_on_error is False
    assert task.epochs == 4
    assert reducer_log_names(task.epochs_reducer) == ["pass_k_1", "pass_k_2", "pass_k_4"]


def test_missing_data_root_clear_error(monkeypatch, tmp_path):
    fake = tmp_path / "site-packages" / "tau_rec" / "inspect_task.py"
    monkeypatch.setattr(inspect_task, "__file__", str(fake))
    with pytest.raises(RuntimeError) as raised:
        inspect_task.repo_root()
    assert "uv sync" in str(raised.value)
    assert "editable" in str(raised.value)
    with pytest.raises(RuntimeError, match="uv sync"):
        _in_eval_context(tau_rec_movies)


def test_task_built_outside_an_eval_has_dataset_scorer_and_caveat():
    # As a checker does: import the function, call it, inspect the Task.
    task = contextvars.copy_context().run(tau_rec_movies)
    assert len(task.dataset) == 60
    assert task.scorer
    meta = task.metadata
    assert meta["caveats"] == ["model_settings_unrecorded"]
    assert meta["request_settings"] is None
    assert meta["model_args"] is None
    assert meta["simulator_model"] is None
    entry = json.loads(G1_ENTRY.read_text())
    assert meta["digests"]["tasks_sha256"] == entry["tasks_sha256"]
    assert (meta["track"], meta["generation"]) == ("movies-v1", "g1")


def test_eval_of_task_instance_completes_offline(tmp_path):
    task = contextvars.copy_context().run(tau_rec_movies)
    log = _run(task, model="mockllm/model", model_roles={"user": "mockllm/model"}, limit=1,
               epochs=Epochs(1, ["pass_k_1"]), log_dir=str(tmp_path))
    sample = _one(log)
    assert sample.scores["tau_rec_scorer"].value == 0.0
    assert [s.reducer for s in log.results.scores] == ["pass_k_1"]
    meta = log.eval.metadata
    assert meta["caveats"] == ["model_settings_unrecorded"]
    assert meta["simulator_model"] is None
    assert set(meta["digests"]) == {"policy_sha256", "tasks_sha256", "catalog_sha256"}
    assert meta["tau_rec_version"] == tau_rec.__version__
    assert "harness_commit" in meta


def test_keyless_eval_without_user_role_builds_task(tmp_path, monkeypatch):
    # No OPENAI_API_KEY and no user role: the task (and its metadata) still
    # builds; only the simulator call fails, as a sample error.
    for name in ("OPENAI_API_KEY", "AZUREAI_OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # An earlier test may have memoized a keyed openai/gpt-5-mini client.
    from inspect_ai.model import _model as inspect_model

    monkeypatch.setattr(inspect_model, "_models", {})
    log = _run(model="mockllm/model", model_roles=None, sample_id=TASK,
               epochs=Epochs(1, ["pass_k_1"]), log_dir=str(tmp_path))
    assert log.status == "success", log.error
    meta = log.eval.metadata
    assert meta["simulator_model"] == "openai/gpt-5-mini"
    assert meta["caveats"] == ["simulator_settings_unresolved"]
    assert meta["request_settings"]["simulator"] is None
    assert meta["request_settings"]["agent"] == {"temperature": 0.0}
    assert meta["model_args"] == {"agent": {}, "simulator": None}
    (sample,) = log.samples
    assert sample.error is not None
    assert "OPENAI_API_KEY" in sample.error.message
    assert not sample.scores


# --- REQ-001: the documented command ------------------------------------------


def test_req001_mock_eval_one_sample(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("INSPECT_")}
    env.pop("OPENAI_API_KEY", None)
    proc = subprocess.run(
        [
            str(Path(sys.executable).parent / "inspect"), "eval", TASK_FILE,
            "--model", "mockllm/model", "--model-role", "user=mockllm/model",
            "--limit", "1", "--epochs", "1", "--epochs-reducer", "pass_k_1",
            "--log-dir", str(tmp_path), "--display", "none",
        ],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    (path,) = tmp_path.glob("*.eval")
    log = read_eval_log(str(path))
    sample = _one(log)
    assert sample.id == "task_001"
    assert sample.scores["tau_rec_scorer"].value in (0.0, 1.0)
    assert [s.reducer for s in log.results.scores] == ["pass_k_1"]


# --- REQ-003 / REQ-005: the agent, the toolkit and the native score -------------


def test_score_equals_native_evaluator(tmp_path):
    agent = _mock([_recommend(PASSING_ITEM)])
    sample = _one(_run(model=agent, sample_id=TASK, epochs=Epochs(1, ["pass_k_1"]),
                       log_dir=str(tmp_path)))

    trace = _stored_trace(sample)
    catalog = Catalog.from_json(str(_data("catalog.json")), MOVIES)
    native = CombinedEvaluator(catalog, MOVIES).evaluate(_task(), trace)

    score = sample.scores["tau_rec_scorer"]
    assert score.value == native.primary_reward == 1.0
    assert score.metadata == {
        "constraint_score": native.constraint_score,
        "policy_score": native.policy_score,
        "stop_reason": "recommended",
        "recommendations": [PASSING_ITEM],
    }
    assert score.answer == PASSING_ITEM
    assert (trace.task_id, trace.trial) == (TASK, 0)


def test_failing_recommendation_scores_zero(tmp_path):
    # Constraint-satisfying but not on the user's services: availability fails.
    agent = _mock([_recommend("tmdb_467905")])
    sample = _one(_run(model=agent, sample_id=TASK, epochs=Epochs(1, ["pass_k_1"]),
                       log_dir=str(tmp_path)))
    assert sample.scores["tau_rec_scorer"].value == 0.0


def test_tool_results_come_from_toolkit(tmp_path, monkeypatch):
    from tau_rec.environment.tools import ToolKit

    calls: list[tuple[str, dict]] = []
    original = ToolKit.call

    def spy(self, name, arguments):
        calls.append((name, arguments))
        return original(self, name, arguments)

    monkeypatch.setattr(ToolKit, "call", spy)
    agent = _mock([
        _tool("get_metadata", {"item_id": PASSING_ITEM}, "call_meta"),
        _tool("check_availability", {"item_id": PASSING_ITEM}, "call_avail"),
        _recommend(PASSING_ITEM),
    ])
    log = _run(model=agent, sample_id=TASK, epochs=Epochs(1, ["pass_k_1"]),
               log_dir=str(tmp_path))
    sample = _one(log)

    trace = _stored_trace(sample)
    toolkit, _ = environment().setup(TASK)
    assert [(c.name, c.arguments) for c in trace.tool_calls] == calls == [
        ("get_metadata", {"item_id": PASSING_ITEM}),
        ("check_availability", {"item_id": PASSING_ITEM}),
        ("recommend", {"item_id": PASSING_ITEM}),
    ]
    for call in trace.tool_calls:
        assert call.result == original(toolkit, call.name, call.arguments)
    # Tool results reach the agent's next generation as tool messages.
    model_events = [e for e in sample.events if e.event == "model"]
    agent_events = [e for e in model_events if e.role is None]
    assert len(agent_events) == 3
    tool_inputs = [m for m in agent_events[-1].input if isinstance(m, ChatMessageTool)]
    assert [m.content for m in tool_inputs] == [c.result for c in trace.tool_calls[:2]]
    assert {e.model for e in agent_events} == {"mockllm/model"}
    assert {e.role for e in model_events if e.role is not None} == {"user"}


def test_state_messages_mirror_trace_order(tmp_path):
    agent = _mock([
        ModelOutput.from_content("mockllm/model", "Any genre you like?"),
        _tool("get_metadata", {"item_id": PASSING_ITEM}, "call_meta"),
        _recommend(PASSING_ITEM),
    ])
    sample = _one(_run(model=agent, sample_id=TASK, epochs=Epochs(1, ["pass_k_1"]),
                       log_dir=str(tmp_path)))
    trace = _stored_trace(sample)

    messages = sample.messages
    assert isinstance(messages[0], ChatMessageSystem)
    assert messages[0].content == environment().setup(TASK)[1]

    mirrored = []
    for message in messages[1:]:
        if isinstance(message, ChatMessageTool):
            continue
        if isinstance(message, ChatMessageAssistant) and message.tool_calls:
            (call,) = message.tool_calls
            mirrored.append(("tool", call.function, call.arguments))
        elif isinstance(message, ChatMessageAssistant):
            mirrored.append((Role.AGENT, message.text))
        else:
            assert isinstance(message, ChatMessageUser)
            mirrored.append((Role.USER, message.text))
    expected = [
        (e.role, e.content) if isinstance(e, MessageEvent) else ("tool", e.name, e.arguments)
        for e in trace.events
    ]
    assert mirrored == expected
    assert expected[0] == (Role.AGENT, MOVIES.greeting)
    assert [e.type for e in trace.events] == [
        "message", "message", "message", "message", "tool_call", "tool_call",
    ]
    results = [m.content for m in messages if isinstance(m, ChatMessageTool)]
    assert results == [e.result for e in trace.events if isinstance(e, ToolCallEvent)]


def test_store_holds_trace_before_each_model_call(tmp_path):
    from inspect_ai.util import store

    seen: list[list[str]] = []
    outputs = [
        ModelOutput.from_content("mockllm/model", "Any genre you like?"),
        _recommend(PASSING_ITEM),
    ]

    def agent(input, tools, tool_choice, config):
        stored = store().get(TRACE_KEY)
        seen.append([e["type"] + ":" + e.get("role", "") for e in stored["events"]])
        return outputs.pop(0)

    sample = _one(_run(model=_mock(agent), sample_id=TASK, epochs=Epochs(1, ["pass_k_1"]),
                       log_dir=str(tmp_path)))
    assert seen == [
        ["message:agent", "message:user"],
        ["message:agent", "message:user", "message:agent", "message:user"],
    ]
    assert _stored_trace(sample).stop_reason.value == "recommended"


# --- REQ-006: pass^k ------------------------------------------------------------


@solver
def _no_op():
    async def solve(state, generate):
        return state

    return solve


# Per-sample success pattern over 4 epochs: c = 0..4 successes.
PATTERNS = {
    "task_001": [0, 0, 0, 0],
    "task_002": [1, 0, 0, 0],
    "task_003": [0, 1, 1, 0],
    "task_004": [1, 1, 0, 1],
    "task_005": [1, 1, 1, 1],
}


@scorer(metrics=[mean_metric()])
def _scripted():
    async def score(state, target):
        return Score(value=float(PATTERNS[state.sample_id][state.epoch - 1]))

    return score


def test_pass_k_matches_aggregate_pass_k(tmp_path):
    task = task_with(_in_eval_context(tau_rec_movies), solver=_no_op(), scorer=_scripted())
    log = _run(task, model="mockllm/model", sample_id=list(PATTERNS), log_dir=str(tmp_path))
    assert log.status == "success", log.error

    counts = {sid: {"n": 4, "c": sum(p)} for sid, p in PATTERNS.items()}
    reported = {s.reducer: s.metrics["mean"].value for s in log.results.scores}
    assert set(reported) == {"pass_k_1", "pass_k_2", "pass_k_4"}
    for k in (1, 2, 4):
        assert reported[f"pass_k_{k}"] == pytest.approx(aggregate_pass_k(counts, k))


def test_epochs_1_with_reducer_pass_k_1(tmp_path):
    sample = _one(log := _run(model=_mock([_recommend(PASSING_ITEM)]), sample_id=TASK,
                              epochs=Epochs(1, ["pass_k_1"]), log_dir=str(tmp_path)))
    assert [s.reducer for s in log.results.scores] == ["pass_k_1"]
    assert log.results.scores[0].metrics["mean"].value == 1.0
    assert sample.epoch == 1


def test_epochs_1_rejects_pass_k_above_epochs(tmp_path):
    # Inspect refuses the task's own pass_k(2)/pass_k(4) at 1 epoch, so other
    # epoch counts must name their reducers (--epochs-reducer).
    with pytest.raises(PrerequisiteError, match="requires 2 epochs"):
        _run(model="mockllm/model", sample_id=TASK, epochs=1, log_dir=str(tmp_path))


def test_concurrent_epochs_independent_traces(tmp_path):
    counter = itertools.count(1)

    async def user(input, tools, tool_choice, config):
        n = next(counter)
        await anyio.sleep(0.01 * (5 - n % 5))  # interleave the epochs
        return ModelOutput.from_content("mockllm/model", f"user reply {n}")

    async def agent(input, tools, tool_choice, config):
        await anyio.sleep(0.01)
        users = [m for m in input if isinstance(m, ChatMessageUser)]
        if len(users) < 2:
            return ModelOutput.from_content("mockllm/model", "Tell me more?")
        return _recommend(PASSING_ITEM)

    log = _run(model=_mock(agent), model_roles={"user": _mock(user)}, sample_id=TASK,
               log_dir=str(tmp_path), max_samples=4)
    assert log.status == "success", log.error
    assert len(log.samples) == 4

    traces = sorted((_stored_trace(s) for s in log.samples), key=lambda t: t.trial)
    assert [t.trial for t in traces] == [0, 1, 2, 3]
    assert sorted(s.epoch - 1 for s in log.samples) == [0, 1, 2, 3]
    for sample in log.samples:
        assert _stored_trace(sample).trial == sample.epoch - 1
    user_texts = []
    for trace in traces:
        assert [e.type for e in trace.events] == [
            "message", "message", "message", "message", "tool_call",
        ]
        user_texts += [e.content for e in trace.events if getattr(e, "role", None) == Role.USER]
        assert trace.recommendations == [PASSING_ITEM]
    assert len(user_texts) == len(set(user_texts)) == 8
    assert {s.scores["tau_rec_scorer"].value for s in log.samples} == {1.0}


# --- REQ-021: Inspect limits ----------------------------------------------------


def test_limit_partial_trace_scored_timeout(tmp_path):
    # The simulator's opening call has 2 messages; the agent's first call has
    # 3 (system, greeting, user), which message_limit=3 refuses.
    agent = _mock([_recommend(PASSING_ITEM)])
    sample = _one(_run(model=agent, sample_id=TASK, epochs=Epochs(1, ["pass_k_1"]),
                       message_limit=3, log_dir=str(tmp_path)))
    assert sample.limit is not None and sample.limit.type == "message"
    score = sample.scores["tau_rec_scorer"]
    assert score.value == 0.0
    assert score.metadata["stop_reason"] == "timeout"
    trace = _stored_trace(sample)
    assert trace.stop_reason.value == "timeout"
    assert [e.role for e in trace.events] == [Role.AGENT, Role.USER]


def test_token_limit_partial_trace_scored_timeout(tmp_path):
    sample = _one(_run(model="mockllm/model", sample_id=TASK, epochs=Epochs(1, ["pass_k_1"]),
                       token_limit=1500, log_dir=str(tmp_path)))
    assert sample.limit is not None and sample.limit.type == "token"
    assert sample.scores["tau_rec_scorer"].value == 0.0
    assert _stored_trace(sample).stop_reason.value == "timeout"
    assert len(_stored_trace(sample).events) >= 1


def test_time_limit_cancelled_trial_scored_timeout(tmp_path):
    # A time limit cancels the solver from outside, so the solver never sees a
    # LimitExceededError; the scorer scores the unfinished trace as TIMEOUT.
    async def slow_agent(input, tools, tool_choice, config):
        await anyio.sleep(30)
        return _recommend(PASSING_ITEM)

    sample = _one(_run(model=_mock(slow_agent), sample_id=TASK,
                       epochs=Epochs(1, ["pass_k_1"]), time_limit=1, log_dir=str(tmp_path)))
    assert sample.limit is not None and sample.limit.type == "time"
    score = sample.scores["tau_rec_scorer"]
    assert score.value == 0.0
    assert score.metadata["stop_reason"] == "timeout"
    assert [e.role for e in _stored_trace(sample).events] == [Role.AGENT, Role.USER]


# --- REQ-022: errors ------------------------------------------------------------

FAILING = "task_002"


def _raises_for(task_id: str):
    persona = _task(task_id).persona

    def user(input, tools, tool_choice, config):
        if persona in input[0].text:
            raise RuntimeError("forced simulator failure")
        return ModelOutput.from_content("mockllm/model", "Something fun, please.")

    return user


def test_forced_error_does_not_abort(tmp_path):
    log = _run(model="mockllm/model", model_roles={"user": _mock(_raises_for(FAILING))},
               sample_id=[TASK, FAILING, "task_003"], epochs=Epochs(1, ["pass_k_1"]),
               log_dir=str(tmp_path))
    assert log.status == "success", log.error
    by_id = {s.id: s for s in log.samples}
    assert by_id[FAILING].error is not None
    assert "forced simulator failure" in by_id[FAILING].error.message
    assert not by_id[FAILING].scores
    for sid in (TASK, "task_003"):
        assert by_id[sid].error is None
        assert by_id[sid].scores["tau_rec_scorer"].value == 0.0


def test_eval_retry_completes_errored_sample(tmp_path, monkeypatch):
    from tau_rec.inspect_support.simulator import InspectSimulator

    original = InspectSimulator._call_once
    failing = {"on": True}

    async def flaky(self, kwargs, attempt):
        if failing["on"] and self.task.id == FAILING:
            raise RuntimeError("forced simulator failure")
        return await original(self, kwargs, attempt)

    monkeypatch.setattr(InspectSimulator, "_call_once", flaky)
    first = _run(TASK_FILE, model="mockllm/model", sample_id=[TASK, FAILING],
                 epochs=Epochs(1, ["pass_k_1"]), log_dir=str(tmp_path))
    assert first.status == "success", first.error
    assert {s.id: s.error is not None for s in first.samples} == {TASK: False, FAILING: True}

    failing["on"] = False
    (retried,) = eval_retry(first, log_dir=str(tmp_path), display="none")
    assert retried.status == "success", retried.error
    assert {s.id: s.error for s in retried.samples} == {TASK: None, FAILING: None}
    assert all("tau_rec_scorer" in s.scores for s in retried.samples)
    assert retried.eval.metadata["track"] == "movies-v1"


# --- REQ-023: comparability metadata --------------------------------------------


def test_log_metadata_comparability(tmp_path):
    log = _run(model=_mock([_recommend(PASSING_ITEM)]), sample_id=TASK,
               epochs=Epochs(1, ["pass_k_1"]), log_dir=str(tmp_path))
    assert log.status == "success", log.error
    meta = log.eval.metadata
    assert set(meta) >= {
        "track", "generation", "digests", "tau_rec_version", "harness_commit",
        "request_settings", "model_args", "simulator_model", "caveats",
    }
    entry = json.loads(G1_ENTRY.read_text())
    assert meta["track"] == "movies-v1"
    assert meta["generation"] == "g1"
    assert meta["digests"] == {
        "policy_sha256": entry["policy_sha256"],
        "tasks_sha256": entry["tasks_sha256"],
        "catalog_sha256": entry["catalog_sha256"],
    }
    assert meta["tau_rec_version"] == tau_rec.__version__
    assert log.eval.task_version == tau_rec.__version__
    assert meta["simulator_model"] == "mockllm/model"
    assert "same_model_as_simulator" in meta["caveats"]
    assert set(meta["request_settings"]) == {"agent", "simulator"}
    assert meta["request_settings"]["simulator"]["temperature"] == 1.0
    assert set(meta["model_args"]) == {"agent", "simulator"}
    json.dumps(meta)  # the header is plain JSON


def test_caveat_absent_when_models_differ(tmp_path):
    log = _run(model="mockllm/model", model_roles={"user": "mockllm/other"}, sample_id=TASK,
               epochs=Epochs(1, ["pass_k_1"]), log_dir=str(tmp_path))
    assert log.eval.metadata["simulator_model"] == "mockllm/other"
    assert "same_model_as_simulator" not in log.eval.metadata["caveats"]


def test_metadata_default_user_model_and_agent_settings(monkeypatch):
    from inspect_ai.model._generate_config import active_generate_config

    from tau_rec.inspect_support.agent import InspectAgent
    from tau_rec.inspect_support.simulator import user_model

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

    def build():
        active = get_model("openai/gpt-5-mini")
        meta = comparability_metadata(active, user_model())
        agent = InspectAgent(active, "", [])
        return meta, agent._generate_config(), active_generate_config()

    meta, agent_config, _ = _in_eval_context(
        build, model=get_model("openai/gpt-5-mini"), roles={}
    )
    assert meta["simulator_model"] == "openai/gpt-5-mini"
    assert meta["model_args"]["simulator"] == {"responses_api": False}
    assert meta["request_settings"]["simulator"]["temperature"] == 1.0
    assert meta["request_settings"]["agent"] == agent_config.model_dump(exclude_none=True)
    assert "same_model_as_simulator" in meta["caveats"]
