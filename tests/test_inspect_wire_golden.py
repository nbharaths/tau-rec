"""Offline wire golden: the Inspect path sends native's request bodies (REQ-018).

Both paths run the unchanged native ``Orchestrator`` over the same fixed
``task_029`` conversation:

1. the opening greeting, then the simulator's first turn (simulator request 1);
2. the agent calls ``get_user_history`` (agent request 1);
3. the tool result goes back and the agent replies in text (agent request 2,
   which carries the tool-call / tool-result round-trip);
4. the simulator answers that agent turn (simulator request 2);
5. the agent calls ``recommend`` and the trial ends (agent request 3).

Every Chat Completions body each path puts on the wire is captured at the HTTP
layer, with no network and dummy keys:

- Inspect: ``get_model("openai/gpt-5-mini", responses_api=False,
  http_client=httpx.AsyncClient(transport=httpx.MockTransport(...)))`` for the
  agent (made Inspect's active model, as inside a sample) and for the default
  ``user`` role model (``user_model()``, with the same client injected);
- native: ``LiteLLMAgent`` and ``UserSimulator`` (``catalog=``, as ``tau-rec
  run`` builds them in tools mode) on litellm 1.83.0, whose OpenAI client uses
  ``litellm.aclient_session``, set to an ``httpx.AsyncClient`` on a
  ``MockTransport``.

Both transports answer from the same script with OpenAI-shaped completions
(tool-call arguments compact, as OpenAI returns them). The bodies must be equal
after normalizing only ``ALLOWED_DIFFERENCES``, the spec's "Known wire
differences"; any other difference is an adapter bug.
"""
from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.model._generate_config import active_generate_config_context_var
from inspect_ai.model._model import active_model_context_var

from tau_rec.agents.litellm_agent import LiteLLMAgent
from tau_rec.inspect_support import simulator as inspect_simulator
from tau_rec.inspect_support.agent import InspectAgent
from tau_rec.inspect_support.simulator import InspectSimulator, user_model
from tau_rec.inspect_support.solver import MoviesEnvironment
from tau_rec.orchestrator.orchestrator import Orchestrator
from tau_rec.simulator.user_simulator import UserSimulator

DATA = Path(__file__).resolve().parent.parent / "data"
TASK_ID = "task_029"
MAX_TURNS = 20
CHAT_PATH = "/v1/chat/completions"

# The spec's "Known wire differences", and nothing else. Each entry names the
# requests it applies to ("agent" / "simulator") and is applied by
# ``normalize_pair``. Extending this list changes the spec and needs Bharath's
# sign-off.
#
# Inspect would also add ``tools[*].function.parameters.additionalProperties:
# false``; ``tool_infos`` drops it (``additionalProperties=None``), so it is
# not on the wire and is not listed.
ALLOWED_DIFFERENCES: tuple[tuple[str, frozenset[str]], ...] = (
    # OpenAI treats system as developer for gpt-5 / o-series reasoning models.
    ("messages[0].role: developer (Inspect) vs system (native)", frozenset({"agent", "simulator"})),
    # The API default; Inspect sends it whenever tools are offered.
    ("tool_choice: 'auto' on Inspect only", frozenset({"agent"})),
    # Inspect re-serializes history tool-call arguments with spaces (accepted 2026-11-16).
    ("messages[*].tool_calls[*].function.arguments: JSON whitespace only", frozenset({"agent"})),
    # gpt-5 supports only temperature 1, which Inspect omits (accepted 2026-11-16).
    ("temperature: 1.0 on native only, gpt-5 models", frozenset({"simulator"})),
)
ROLE, TOOL_CHOICE, ARGUMENTS, TEMPERATURE = (name for name, _ in ALLOWED_DIFFERENCES)


# ------------------------------------------------------------------ the script

SIM_TURNS = [
    "Hi! I'm in the mood for a crime movie, ideally under about two hours and ten minutes.",
    "Something rated at least 5.5 would be good.",
]
GET_HISTORY = {"id": "call_history_1", "name": "get_user_history", "arguments": {"user_id": "user_1"}}
RECOMMEND = {"id": "call_recommend_1", "name": "recommend", "arguments": {"item_id": "movie_001"}}
AGENT_TURNS: list[dict] = [
    {"content": None, "tool_calls": [GET_HISTORY]},
    {"content": "Thanks! Do you have a minimum rating in mind?", "tool_calls": []},
    {"content": None, "tool_calls": [RECOMMEND]},
]


def _completion(content: str | None, tool_calls: list[dict]) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content, "refusal": None}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    # Compact, as OpenAI returns tool-call arguments.
                    "arguments": json.dumps(call["arguments"], separators=(",", ":")),
                },
            }
            for call in tool_calls
        ]
    return {
        "id": "chatcmpl-wire-golden",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-5-mini-2025-08-07",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class Recorder:
    """A ``MockTransport`` handler: records each request body, answers from the script.

    Agent requests carry ``tools``; simulator requests do not.
    """

    def __init__(self) -> None:
        self.agent: list[dict] = []
        self.simulator: list[dict] = []
        self.other: list[str] = []
        self._agent_turns = list(AGENT_TURNS)
        self._sim_turns = list(SIM_TURNS)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or request.url.path != CHAT_PATH:
            self.other.append(f"{request.method} {request.url}")
            return httpx.Response(404, json={"error": {"message": "not scripted"}})
        body = json.loads(request.content)
        if "tools" in body:
            self.agent.append(body)
            turn = self._agent_turns.pop(0)
            return httpx.Response(200, json=_completion(turn["content"], turn["tool_calls"]))
        self.simulator.append(body)
        return httpx.Response(200, json=_completion(self._sim_turns.pop(0), []))

    def check_complete(self) -> None:
        assert not self.other, f"unexpected requests: {self.other}"
        assert not self._agent_turns and not self._sim_turns, "script entries left unused"
        assert len(self.agent) == len(AGENT_TURNS)
        assert len(self.simulator) == len(SIM_TURNS)


# ------------------------------------------------------------------ the two paths


@pytest.fixture(scope="module")
def env() -> MoviesEnvironment:
    # The solver's environment; tests/test_inspect_replay.py checks its tool
    # kit and prompt equal the ones `tau-rec run` builds.
    return MoviesEnvironment.load(DATA)


@pytest.fixture
def offline(monkeypatch):
    """Dummy keys, litellm's bundled model map, and no socket connections at all."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-wire-golden")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    def refuse(*args, **kwargs):
        raise AssertionError("the wire golden must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _orchestrator(env: MoviesEnvironment, agent, simulator, toolkit) -> Orchestrator:
    return Orchestrator(
        agent=agent, simulator=simulator, toolkit=toolkit,
        max_turns=MAX_TURNS, spec=env.spec, harness=env.harness,
    )


async def native_bodies(env: MoviesEnvironment, monkeypatch) -> Recorder:
    import litellm

    from tau_rec import cost

    recorder = Recorder()
    # litellm caches OpenAI clients by key/settings; start from an empty cache
    # so the client is built on the mock session.
    litellm.in_memory_llm_clients_cache.flush_cache()
    monkeypatch.setattr(
        litellm, "aclient_session", httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    )
    cost.reset()

    task = env.tasks[TASK_ID]
    toolkit, system_prompt = env.setup(TASK_ID)
    agent = LiteLLMAgent(
        "gpt-5-mini", system_prompt, toolkit.tool_definitions(), harness=env.harness,
    )
    simulator = UserSimulator(
        "gpt-5-mini", task, spec=env.spec, catalog=env.catalog, harness=env.harness,
    )
    try:
        await _orchestrator(env, agent, simulator, toolkit).run(TASK_ID, "gpt-5-mini", 0)
    finally:
        litellm.in_memory_llm_clients_cache.flush_cache()
        cost.reset()
    recorder.check_complete()
    return recorder


async def inspect_bodies(env: MoviesEnvironment, monkeypatch) -> Recorder:
    recorder = Recorder()

    def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(recorder))

    agent_model = get_model(
        "openai/gpt-5-mini", responses_api=False, http_client=client(),
        api_key="dummy", memoize=False,
    )

    # The default user role model, exactly as user_model() builds it, with the
    # mock client and a dummy key added.
    real_get_model = inspect_simulator.get_model

    def get_user_model(*args, **kwargs):
        return real_get_model(
            *args, **kwargs, http_client=client(), api_key="dummy", memoize=False,
        )

    monkeypatch.setattr(inspect_simulator, "get_model", get_user_model)
    sim_model = user_model()

    task = env.tasks[TASK_ID]
    toolkit, system_prompt = env.setup(TASK_ID)
    agent = InspectAgent(agent_model, system_prompt, toolkit.tool_definitions(), harness=env.harness)
    simulator = InspectSimulator(sim_model, task, spec=env.spec, harness=env.harness)

    # Inspect's active model and run config, as inside a sample.
    model_token = active_model_context_var.set(agent_model)
    config_token = active_generate_config_context_var.set(GenerateConfig())
    try:
        await _orchestrator(env, agent, simulator, toolkit).run(TASK_ID, "gpt-5-mini", 0)
    finally:
        active_generate_config_context_var.reset(config_token)
        active_model_context_var.reset(model_token)
    recorder.check_complete()
    return recorder


# ------------------------------------------------------------------ comparison


def _applies(entry: str, kind: str) -> bool:
    return kind in dict(ALLOWED_DIFFERENCES)[entry]


def _parsed_arguments(left: dict, right: dict) -> None:
    """Replace each paired history tool-call ``arguments`` string by its parsed JSON.

    Only where both sides have the same message/tool-call structure and both
    strings parse; anything else is left as is and fails the comparison.
    """
    left_messages, right_messages = left.get("messages"), right.get("messages")
    if not (isinstance(left_messages, list) and isinstance(right_messages, list)):
        return
    for left_message, right_message in zip(left_messages, right_messages):
        left_calls = left_message.get("tool_calls") if isinstance(left_message, dict) else None
        right_calls = right_message.get("tool_calls") if isinstance(right_message, dict) else None
        if not (isinstance(left_calls, list) and isinstance(right_calls, list)):
            continue
        for left_call, right_call in zip(left_calls, right_calls):
            left_function = left_call.get("function") if isinstance(left_call, dict) else None
            right_function = right_call.get("function") if isinstance(right_call, dict) else None
            if not (isinstance(left_function, dict) and isinstance(right_function, dict)):
                continue
            try:
                left_args = json.loads(left_function["arguments"])
                right_args = json.loads(right_function["arguments"])
            except (KeyError, TypeError, ValueError):
                continue
            left_function["arguments"] = {"parsed": left_args}
            right_function["arguments"] = {"parsed": right_args}


def normalize_pair(inspect_body: dict, native_body: dict, kind: str) -> tuple[dict, dict]:
    """Both bodies with only the ``ALLOWED_DIFFERENCES`` for ``kind`` normalized.

    The unconditional entries (role, tool_choice) are asserted to be present
    with their expected values, so a stale entry fails too. The conditional
    ones normalize only when their condition holds; otherwise the bodies are
    left as sent and the equality check fails.
    """
    left, right = copy.deepcopy(inspect_body), copy.deepcopy(native_body)

    if _applies(ROLE, kind):
        assert left["messages"][0]["role"] == "developer", "Inspect messages[0].role"
        assert right["messages"][0]["role"] == "system", "native messages[0].role"
        left["messages"][0]["role"] = right["messages"][0]["role"]

    if _applies(TOOL_CHOICE, kind):
        assert left.get("tool_choice") == "auto", "Inspect tool_choice"
        assert "tool_choice" not in right, "native tool_choice"
        del left["tool_choice"]

    if _applies(ARGUMENTS, kind):
        _parsed_arguments(left, right)

    if (
        _applies(TEMPERATURE, kind)
        and "temperature" not in left
        and right.get("temperature") == 1.0
        and isinstance(right.get("temperature"), float)
        and str(left.get("model", "")).startswith("gpt-5")
        and str(right.get("model", "")).startswith("gpt-5")
    ):
        del right["temperature"]

    return left, right


def diff_paths(a: Any, b: Any, path: str = "") -> list[str]:
    """JSON paths where ``a`` and ``b`` differ (keys and indices only, no values)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else key
            if key not in a:
                out.append(f"{sub} (native only)")
            elif key not in b:
                out.append(f"{sub} (inspect only)")
            else:
                out.extend(diff_paths(a[key], b[key], sub))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} vs {len(b)})"]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in diff_paths(x, y, f"{path}[{i}]")]
    return [] if a == b else [path]


def assert_bodies_match(inspect_bodies: list[dict], native_bodies: list[dict], kind: str) -> None:
    assert len(inspect_bodies) == len(native_bodies)
    for index, (inspect_body, native_body) in enumerate(zip(inspect_bodies, native_bodies)):
        left, right = normalize_pair(inspect_body, native_body, kind)
        assert left == right, f"{kind} request {index} differs at {diff_paths(left, right)}"


_CAPTURED: list[tuple[Recorder, Recorder]] = []


@pytest.fixture
async def captured(env, offline, monkeypatch) -> tuple[Recorder, Recorder]:
    """Both paths' recorded bodies; captured once per session (they are plain data)."""
    if not _CAPTURED:
        native = await native_bodies(env, monkeypatch)
        inspect = await inspect_bodies(env, monkeypatch)
        _CAPTURED.append((inspect, native))
    return copy.deepcopy(_CAPTURED[0])


# ------------------------------------------------------------------ tests


async def test_agent_wire_body_matches_native(captured):
    inspect, native = captured
    # The second request carries the tool-call / tool-result round-trip.
    roles = [message["role"] for message in native.agent[1]["messages"]]
    assert roles == ["system", "assistant", "user", "assistant", "tool"]
    assert_bodies_match(inspect.agent, native.agent, "agent")


async def test_simulator_wire_body_matches_native(captured):
    inspect, native = captured
    # The second request follows an agent turn.
    roles = [message["role"] for message in native.simulator[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert_bodies_match(inspect.simulator, native.simulator, "simulator")


def test_additional_properties_not_on_the_wire_list():
    """``tool_infos`` drops ``additionalProperties``, so it is not an accepted difference."""
    assert len(ALLOWED_DIFFERENCES) == 4
    assert not any("additionalProperties" in name for name, _ in ALLOWED_DIFFERENCES)


def _native_call(body: dict) -> dict:
    return body["messages"][3]["tool_calls"][0]["function"]


def _mutations() -> dict:
    """Real differences the normalizer must not hide: (kind, mutate(inspect, native))."""

    def argument_value(inspect, native):
        _native_call(native)["arguments"] = '{"user_id":"user_2"}'

    def argument_key(inspect, native):
        _native_call(native)["arguments"] = '{"user":"user_1"}'

    def argument_not_json(inspect, native):
        _native_call(native)["arguments"] = '{"user_id":"user_1"'

    def tool_call_name(inspect, native):
        _native_call(native)["name"] = "get_metadata"

    def tool_message_content(inspect, native):
        native["messages"][4]["content"] += " "

    def native_temperature_07(inspect, native):
        native["temperature"] = 0.7

    def inspect_temperature_07(inspect, native):
        inspect["temperature"] = 0.7

    def temperature_on_non_gpt5(inspect, native):
        inspect["model"] = native["model"] = "gpt-4.1-mini"

    def extra_native_field(inspect, native):
        native["max_completion_tokens"] = 4096

    def extra_inspect_field(inspect, native):
        inspect["parallel_tool_calls"] = True

    def system_prompt_text(inspect, native):
        native["messages"][0]["content"] += "\n"

    def native_tool_choice(inspect, native):
        native["tool_choice"] = "auto"

    return {
        "argument_value": ("agent", argument_value),
        "argument_key": ("agent", argument_key),
        "argument_not_json": ("agent", argument_not_json),
        "tool_call_name": ("agent", tool_call_name),
        "tool_message_content": ("agent", tool_message_content),
        "extra_native_field_agent": ("agent", extra_native_field),
        "extra_inspect_field_agent": ("agent", extra_inspect_field),
        "native_tool_choice": ("agent", native_tool_choice),
        "native_temperature_07": ("simulator", native_temperature_07),
        "inspect_temperature_07": ("simulator", inspect_temperature_07),
        "temperature_on_non_gpt5": ("simulator", temperature_on_non_gpt5),
        "extra_native_field_simulator": ("simulator", extra_native_field),
        "system_prompt_text": ("simulator", system_prompt_text),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
async def test_normalizer_still_fails_on_a_real_difference(captured, name):
    kind, mutate = _mutations()[name]
    inspect, native = captured
    # The second agent request has the tool round-trip; any simulator request will do.
    inspect_body = copy.deepcopy(getattr(inspect, kind)[1])
    native_body = copy.deepcopy(getattr(native, kind)[1])
    assert_bodies_match([inspect_body], [native_body], kind)  # the unmutated pair passes
    mutate(inspect_body, native_body)
    with pytest.raises(AssertionError):
        assert_bodies_match([inspect_body], [native_body], kind)
