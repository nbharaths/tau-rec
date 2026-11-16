"""InspectSimulator: the native UserSimulator, with only the provider call on Inspect."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from inspect_ai import Task as InspectTask
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.model._generate_config import active_generate_config_context_var
from inspect_ai.model._model import (
    ModelGenerateError,
    ModelRefusalError,
    active_model_context_var,
)
from inspect_ai.solver import solver

from tau_rec.data_model.conversation import Message, Role
from tau_rec.data_model.task import Task
from tau_rec.inspect_support.simulator import (
    DEFAULT_USER_MODEL,
    InspectSimulator,
    SimulatorCallError,
    SimulatorProviderError,
    user_model,
)
from tau_rec.orchestrator.orchestrator import OPENING_GREETING
from tau_rec.simulator import user_simulator
from tau_rec.simulator.user_simulator import UserSimulator, build_simulator_prompt

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_TASKS = ["task_029", "task_031", "task_050"]


def _task(task_id: str = "task_029") -> Task:
    return Task.from_json(str(ROOT / "data" / "tasks" / f"{task_id}.json"))


class Recorder:
    """A mockllm custom_outputs callable that records every generate call."""

    def __init__(self, outputs: list[ModelOutput]):
        self._outputs = list(outputs)
        self.calls: list[dict] = []

    def __call__(self, input, tools, tool_choice, config):
        self.calls.append({"input": list(input), "config": config})
        output = self._outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


def _reply(content: str, **kwargs) -> ModelOutput:
    return ModelOutput.from_content("mockllm/model", content, **kwargs)


def _model(recorder: Recorder, name: str = "mockllm/model", **kwargs):
    return get_model(name, custom_outputs=recorder, memoize=False, **kwargs)


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


@pytest.fixture
def no_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(user_simulator.asyncio, "sleep", sleep)
    return sleep


# --- the default user model --------------------------------------------------


def test_default_user_model_resolution_no_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    model = user_model()
    assert DEFAULT_USER_MODEL == "openai/gpt-5-mini"
    assert str(model) == "openai/gpt-5-mini"
    assert model.role == "user"
    assert model.config.temperature == 1.0
    assert model.model_args["responses_api"] is False
    assert model.api.responses_api is False


def test_user_model_distinct_from_agent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("INSPECT_EVAL_MODEL", "openai/gpt-5-mini")
    agent = get_model()
    simulator = user_model()
    assert str(agent) == str(simulator) == "openai/gpt-5-mini"
    assert simulator is not agent
    # The agent keeps Inspect's own defaults for the name.
    assert agent.api.responses_api is True
    assert agent.config.temperature is None


# --- inherited native behavior -----------------------------------------------


@pytest.mark.parametrize("task_id", GOLDEN_TASKS)
def test_simulator_prompt_byte_identical(task_id):
    task = _task(task_id)
    simulator = InspectSimulator(_model(Recorder([])), task)
    golden = (ROOT / "tests" / "goldens" / f"simulator_prompt_{task_id}.txt").read_text()
    assert simulator.system_prompt == build_simulator_prompt(task)
    assert simulator.system_prompt == golden


def test_simulator_overrides_only_call_once():
    defined = {
        name
        for name, value in vars(InspectSimulator).items()
        if callable(value) or isinstance(value, (property, staticmethod, classmethod))
    }
    assert defined - {"__init__"} == {"_call_once"}
    assert issubclass(InspectSimulator, UserSimulator)


async def test_role_flip_inherited():
    recorder = Recorder([_reply("I want a comedy.")])
    simulator = InspectSimulator(_model(recorder), _task())
    conversation = [
        Message(role=Role.AGENT, content=OPENING_GREETING),
        Message(role=Role.USER, content="Something fun."),
        Message(role=Role.AGENT, content="Which genre?"),
    ]

    reply = await simulator.respond(conversation)

    assert reply == Message(role=Role.USER, content="I want a comedy.")
    sent = recorder.calls[0]["input"]
    assert [type(m) for m in sent] == [
        ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageUser,
    ]
    assert [m.text for m in sent] == [
        simulator.system_prompt, OPENING_GREETING, "Something fun.", "Which genre?",
    ]


async def test_empty_replies_resampled_then_error(no_sleep):
    recorder = Recorder([_reply("") for _ in range(8)])
    simulator = InspectSimulator(_model(recorder), _task())

    with pytest.raises(RuntimeError, match="no text in 8 attempts"):
        await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])

    assert len(recorder.calls) == 8
    assert [row["status"] for row in simulator.call_log] == ["empty"] * 8
    assert no_sleep.await_count == 8


async def test_empty_reply_resampled_then_ok(no_sleep):
    recorder = Recorder([_reply("   "), _reply("Hi, I want a thriller.")])
    simulator = InspectSimulator(_model(recorder), _task())

    reply = await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])

    assert reply.content == "Hi, I want a thriller."
    assert [(row["attempt"], row["status"]) for row in simulator.call_log] == [
        (1, "empty"), (2, "ok"),
    ]


async def test_call_log_row_per_call():
    recorder = Recorder([_reply("Hello.")])
    simulator = InspectSimulator(_model(recorder), _task())

    await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])

    (row,) = simulator.call_log
    assert row["attempt"] == 1
    assert row["status"] == "ok"
    assert row["requested_model"] == "mockllm/model"
    assert row["usage"]["output_tokens"] >= 1


# --- request settings (REQ-018) ----------------------------------------------


async def test_simulator_temperature_mirrored_when_unset():
    recorder = Recorder([_reply("ok")])
    simulator = InspectSimulator(_model(recorder), _task())
    await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])
    assert recorder.calls[0]["config"].temperature == 1.0


async def test_role_config_temperature_is_explicit():
    recorder = Recorder([_reply("ok")])
    model = _model(recorder, config=GenerateConfig(temperature=0.4))
    simulator = InspectSimulator(model, _task())
    await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])
    assert recorder.calls[0]["config"].temperature == 0.4


async def test_run_config_does_not_leak_into_simulator():
    agent = _model(Recorder([]))
    recorder = Recorder([_reply("ok")])
    simulator = InspectSimulator(_model(recorder), _task())
    run_config = GenerateConfig(temperature=0.0, reasoning_effort="high")
    with active(agent, run_config):
        await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])
    config = recorder.calls[0]["config"]
    assert config.temperature == 1.0
    assert config.reasoning_effort is None


async def test_settings_use_provider_qualified_name(monkeypatch):
    from tau_rec.inspect_support import simulator as module

    seen = []

    def fake(model):
        seen.append(model)
        return {"temperature": 1.0, "thinking": {"type": "disabled"}}

    monkeypatch.setattr(module, "simulator_request_settings", fake)
    recorder = Recorder([_reply("ok")])
    simulator = InspectSimulator(_model(recorder), _task())
    await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])
    assert seen == ["mockllm/model"]
    assert recorder.calls[0]["config"].extra_body == {"thinking": {"type": "disabled"}}


# --- failure signals (REQ-021) -----------------------------------------------


@pytest.mark.parametrize("stop_reason", ["model_length", "content_filter"])
async def test_abnormal_stop_raises_without_model_text(stop_reason, no_sleep):
    secret = "MODEL-TEXT connection unavailable 429"
    recorder = Recorder([_reply(secret, stop_reason=stop_reason)])
    simulator = InspectSimulator(_model(recorder), _task())

    with pytest.raises(SimulatorCallError) as raised:
        await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])

    assert "MODEL-TEXT" not in str(raised.value)
    assert "MODEL-TEXT" not in simulator.call_log[0]["error"]
    assert stop_reason in str(raised.value)
    # Not retried as a transient error: one call, one error row.
    assert len(recorder.calls) == 1
    assert [row["status"] for row in simulator.call_log] == ["error"]
    assert simulator.call_log[0]["retrying"] is False


# Inspect's ModelGenerateError message carries the request JSON: here it holds
# words the native loop treats as transient.
REQUEST_TEXT = "REQUEST-TEXT the connection was unavailable, error 429 overloaded rate_limit 503"


def _generate_error(status_code: int | None, cause: Exception | None = None):
    error = ModelGenerateError(
        f"\nRequest:\n{REQUEST_TEXT}\n\nBadRequestError('{REQUEST_TEXT}')",
        status_code=status_code,
        provider_message=f"provider says {REQUEST_TEXT}",
    )
    error.__cause__ = cause
    return error


async def test_provider_error_request_text_not_retried(no_sleep):
    recorder = Recorder([_generate_error(400)])
    simulator = InspectSimulator(_model(recorder), _task())

    with pytest.raises(SimulatorProviderError) as raised:
        await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])

    assert len(recorder.calls) == 1
    (row,) = simulator.call_log
    assert row["status"] == "error"
    assert row["retrying"] is False
    assert no_sleep.await_count == 0
    assert str(raised.value) == "user simulator provider call failed (status 400)"
    assert "REQUEST-TEXT" not in row["error"]
    assert raised.value.status_code == 400
    assert raised.value.provider_message == f"provider says {REQUEST_TEXT}"
    assert isinstance(raised.value.__cause__, ModelGenerateError)


@pytest.mark.parametrize("status_code", [429, 503])
async def test_genuine_transient_status_retried(status_code, no_sleep):
    recorder = Recorder([_generate_error(status_code), _reply("A comedy, please.")])
    simulator = InspectSimulator(_model(recorder), _task())

    reply = await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])

    assert reply.content == "A comedy, please."
    assert len(recorder.calls) == 2
    assert [(row["status"], row.get("retrying")) for row in simulator.call_log] == [
        ("error", True), ("ok", None),
    ]
    assert str(status_code) in simulator.call_log[0]["error"]
    assert "REQUEST-TEXT" not in simulator.call_log[0]["error"]
    assert no_sleep.await_count == 1


async def test_statusless_error_keeps_only_cause_class_name(no_sleep):
    class APIConnectionError(Exception):
        pass

    recorder = Recorder([_generate_error(None, APIConnectionError(REQUEST_TEXT)), _reply("ok")])
    simulator = InspectSimulator(_model(recorder), _task())

    await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])

    error = simulator.call_log[0]["error"]
    assert error == "user simulator provider call failed (APIConnectionError)"
    assert simulator.call_log[0]["retrying"] is True


async def test_refusal_error_raised_without_model_text(no_sleep):
    refusal = _reply(f"MODEL-TEXT {REQUEST_TEXT}", stop_reason="content_filter")
    recorder = Recorder([ModelRefusalError(refusal, "mockllm/model", "user")])
    simulator = InspectSimulator(_model(recorder), _task())

    with pytest.raises(SimulatorCallError) as raised:
        await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])

    assert len(recorder.calls) == 1
    assert "MODEL-TEXT" not in str(raised.value)
    assert "REQUEST-TEXT" not in simulator.call_log[0]["error"]
    assert simulator.call_log[0]["retrying"] is False
    assert isinstance(raised.value.__cause__, ModelRefusalError)


async def test_max_tokens_passes_through():
    recorder = Recorder([_reply("A truncated rep", stop_reason="max_tokens")])
    simulator = InspectSimulator(_model(recorder), _task())
    reply = await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])
    assert reply.content == "A truncated rep"


# --- usage split in a mocked eval (REQ-004) ----------------------------------


USER_USAGE = ModelUsage(input_tokens=7, output_tokens=3, total_tokens=10)


@solver
def _simulated_turn():
    async def solve(state, generate):
        await get_model().generate("agent turn")
        simulator = InspectSimulator(user_model(), _task())
        reply = await simulator.respond([Message(role=Role.AGENT, content=OPENING_GREETING)])
        state.metadata["reply"] = reply.content
        state.metadata["sim_log"] = simulator.call_log
        return state

    return solve


def test_role_usage_split(tmp_path):
    user_output = _reply("I'd like a heist movie.")
    user_output.usage = USER_USAGE
    user = get_model("mockllm/model", custom_outputs=[user_output])
    agent = get_model("mockllm/model", custom_outputs=[_reply("Hello, how can I help?")])

    (log,) = inspect_eval(
        InspectTask(dataset=[Sample(input="x")], solver=_simulated_turn()),
        model=agent,
        model_roles={"user": user},
        log_dir=str(tmp_path),
        display="none",
    )

    assert log.status == "success", log.error
    (sample,) = log.samples
    assert sample.metadata["reply"] == "I'd like a heist movie."
    assert sample.metadata["sim_log"][0]["requested_model"] == "mockllm/model"
    assert set(log.stats.role_usage) == {"user"}
    assert log.stats.role_usage["user"].total_tokens == 10
    assert log.stats.model_usage["mockllm/model"].total_tokens > 10
