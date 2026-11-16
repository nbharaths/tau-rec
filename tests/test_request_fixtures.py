"""Pin the exact kwargs the native agent and simulator hand to `litellm.acompletion`.

These are characterization tests: the fixtures were generated from the code
before the request-building refactor, and the refactor must leave every
request byte-identical (a changed request is a changed reproduction).
Each fixture key is "<model>|<harness id>|<case>".
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tau_rec.agents.litellm_agent import LiteLLMAgent, build_agent_system_prompt
from tau_rec.catalog.search import CatalogSearch
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.conversation import Message, Role
from tau_rec.data_model.task import Task
from tau_rec.environment.tools import ToolKit
from tau_rec.harness import BOOKS_V4, LEGACY
from tau_rec.orchestrator.orchestrator import OPENING_GREETING
from tau_rec.simulator.user_simulator import UserSimulator
from tau_rec.track import load_track

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
MOVIES_TRACK = ROOT / "data"
BOOKS_TRACK = ROOT / "data" / "tracks" / "books-wikidata-v2"
MOVIES_TASK = "task_029"

MOVIES_MODELS = [
    "gpt-5-mini",
    "openrouter/deepseek/deepseek-v4-flash",
    "deepseek/deepseek-chat",
    "openrouter/qwen/qwen3-32b",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]
EFFORT_MODELS = [
    "openrouter/deepseek/deepseek-v4-flash",
    "deepseek/deepseek-chat",
    "openrouter/qwen/qwen3-32b",
]
BOOKS_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"
BOOKS_ORDER = ["deepinfra/fp8"]

# Reasoning-effort routing reads litellm's model map; pin the answers the
# fixtures were recorded under (tests/conftest.py).
pytestmark = pytest.mark.usefixtures("pinned_reasoning_effort_params")


def _chat_response(content="Hello"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        id="r", model="m", system_fingerprint=None, _hidden_params={}, usage=None,
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        provider="P",
    )


def _norm(kwargs: dict) -> dict:
    return json.loads(json.dumps(kwargs))


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _greeting() -> Message:
    return Message(role=Role.AGENT, content=OPENING_GREETING)


def _book_task() -> Task:
    first = sorted((BOOKS_TRACK / "tasks").glob("*.json"))[0]
    return Task.from_json(str(first))


async def _simulator_kwargs(monkeypatch, model, task, spec, harness, provider_order=None):
    import litellm

    completion = AsyncMock(return_value=_chat_response())
    monkeypatch.setattr(litellm, "acompletion", completion)
    monkeypatch.setattr("tau_rec.simulator.user_simulator.asyncio.sleep", AsyncMock())
    sim = UserSimulator(model, task, spec, harness=harness, provider_order=provider_order)
    await sim.respond([_greeting()])
    assert completion.await_count == 1
    return _norm(completion.await_args.kwargs)


def _simulator_cases():
    movies = load_track(MOVIES_TRACK)
    task = Task.from_json(str(MOVIES_TRACK / "tasks" / f"{MOVIES_TASK}.json"))
    for model in MOVIES_MODELS:
        yield f"{model}|{LEGACY.id}|simulator", (model, task, movies.spec, LEGACY, None)
    books = load_track(BOOKS_TRACK)
    yield f"{BOOKS_MODEL}|{BOOKS_V4.id}|simulator", (
        BOOKS_MODEL, _book_task(), books.spec, BOOKS_V4, BOOKS_ORDER,
    )


async def test_native_simulator_request_kwargs_pinned(monkeypatch):
    expected = _load("simulator_request_kwargs.json")
    actual = {}
    for key, (model, task, spec, harness, order) in list(_simulator_cases()):
        actual[key] = await _simulator_kwargs(monkeypatch, model, task, spec, harness, order)
    assert set(actual) == set(expected)
    for key in actual:
        assert actual[key] == expected[key], key


def _movies_agent_inputs():
    track = load_track(MOVIES_TRACK)
    spec = track.spec
    catalog = Catalog.from_json(str(track.catalog_path), spec)
    task = Task.from_json(str(MOVIES_TRACK / "tasks" / f"{MOVIES_TASK}.json"))
    toolkit = ToolKit(
        catalog=catalog,
        search=CatalogSearch(catalog, spec, profile=track.search_profile),
        user_histories=task.user_history or {},
        spec=spec,
        mode="tools",
        retrieval_mode="keyword",
    )
    policy = track.policy_path.read_text()
    system_prompt = build_agent_system_prompt(
        policy, task.user_id, spec=spec, mode="tools", retrieval_mode="keyword",
    )
    return system_prompt, toolkit.tool_definitions()


def _agent_cases():
    for model in MOVIES_MODELS:
        yield f"{model}|{LEGACY.id}|default", (model, None)
    for model in EFFORT_MODELS:
        yield f"{model}|{LEGACY.id}|effort-high", (model, "high")


async def test_native_agent_request_kwargs_movies_pinned(monkeypatch):
    import litellm

    expected = _load("agent_request_kwargs_movies.json")
    system_prompt, tools = _movies_agent_inputs()
    actual = {}
    for key, (model, effort) in _agent_cases():
        completion = AsyncMock(return_value=_chat_response())
        monkeypatch.setattr(litellm, "acompletion", completion)
        monkeypatch.setattr("tau_rec.agents.litellm_agent.asyncio.sleep", AsyncMock())
        agent = LiteLLMAgent(model, system_prompt, tools, reasoning_effort=effort, harness=LEGACY)
        agent.add_user_message(OPENING_GREETING)
        await agent.respond()
        assert completion.await_count == 1
        actual[key] = _norm(completion.await_args.kwargs)
    assert set(actual) == set(expected)
    for key in actual:
        assert actual[key] == expected[key], key
