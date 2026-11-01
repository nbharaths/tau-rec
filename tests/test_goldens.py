"""Byte-for-byte snapshots of every string the movies-v1 harness shows a model.

Generalising the harness over domains must leave movies-v1 untouched. These snapshots pin what the
current harness renders — the agent prompts, the tool schemas, the simulator
prompt, the greeting, and the tool result strings — so the refactor can be
shown, not argued, to leave movies-v1 untouched. Prompt text is scored content
under leaderboard/GENERATIONS.md: a change here that is not byte-identical is
a new harness generation, not a refactor.

Regenerate deliberately, never by accident:

    UPDATE_GOLDENS=1 uv run pytest tests/test_goldens.py

and then explain the diff in the commit that carries it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tau_rec.agents.litellm_agent import build_agent_system_prompt
from tau_rec.catalog.search import CatalogSearch
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.environment.tools import ToolKit
from tau_rec.orchestrator.orchestrator import OPENING_GREETING
from tau_rec.simulator.user_simulator import build_simulator_prompt

ROOT = Path(__file__).resolve().parent.parent
GOLDENS = Path(__file__).resolve().parent / "goldens"

# One real task per simulator-prompt branch: services listed, no services,
# and no-valid-recommendation. Task files are frozen content, so they are
# stable fixtures.
SIM_TASKS = ("task_031", "task_029", "task_050")

# A movie that is on Netflix in the movies-v1 catalog, so the availability result has
# both a true and a false in it. task_029 carries a watched item for user_1.
FOUND_ID = "tmdb_1268127"
MISSING_ID = "tmdb_0"
HISTORY_TASK = "task_029"


def _task(task_id: str) -> Task:
    return Task.from_json(str(ROOT / "data" / "tasks" / f"{task_id}.json"))


def render_all() -> dict[str, str]:
    policy = (ROOT / "data" / "policy.md").read_text()
    task = _task(HISTORY_TASK)

    # The goldens were first generated from the inline expressions that
    # tau-rec run used to carry, then the rendering moved behind
    # build_agent_system_prompt. The bytes did not change, which is what makes
    # that move a refactor.
    memory_policy = (ROOT / "data" / "policy_memory.md").read_text()
    out: dict[str, str] = {
        "agent_system_prompt.txt": build_agent_system_prompt(policy, task.user_id),
        "agent_no_tools_prompt.txt": build_agent_system_prompt(policy, task.user_id, no_tools=True),
        # Memory mode is new text with no prior rendering to match; it is
        # pinned so it cannot drift silently either.
        "agent_memory_prompt.txt": build_agent_system_prompt(memory_policy, task.user_id, mode="memory"),
        "opening_greeting.txt": OPENING_GREETING,
    }
    for tid in SIM_TASKS:
        out[f"simulator_prompt_{tid}.txt"] = build_simulator_prompt(_task(tid))

    catalog = Catalog.from_json(str(ROOT / "data" / "catalog.json"))
    toolkit = ToolKit(
        catalog=catalog, search=CatalogSearch(catalog), user_histories=task.user_history or {}
    )
    out["tool_definitions.json"] = json.dumps(toolkit.tool_definitions(), indent=2) + "\n"
    memory_kit = ToolKit(catalog=catalog, search=CatalogSearch(catalog), user_histories={}, mode="memory")
    out["tool_definitions_memory.json"] = json.dumps(memory_kit.tool_definitions(), indent=2) + "\n"

    # One result per ToolKit branch. These strings go straight into the
    # model's context, so they are prompt text too.
    calls = {
        "get_metadata:found": ("get_metadata", {"item_id": FOUND_ID}),
        "get_metadata:not_found": ("get_metadata", {"item_id": MISSING_ID}),
        "check_availability:found": (
            "check_availability",
            {"item_id": FOUND_ID,
             "services": ["Netflix", "Amazon Prime Video", "Hulu", "Amazon Prime"]},
        ),
        "check_availability:not_found": (
            "check_availability", {"item_id": MISSING_ID, "services": ["Netflix"]}
        ),
        "check_content_preference:R": ("check_content_preference", {"content_rating": "R"}),
        "check_content_preference:NC-17": ("check_content_preference", {"content_rating": "NC-17"}),
        "check_content_preference:PG-13": ("check_content_preference", {"content_rating": "PG-13"}),
        "check_content_preference:NR": ("check_content_preference", {"content_rating": "NR"}),
        "recommend:valid": ("recommend", {"item_id": FOUND_ID}),
        "recommend:abstain": ("recommend", {"item_id": None}),
        "recommend:not_found": ("recommend", {"item_id": MISSING_ID}),
        "get_user_history:known": ("get_user_history", {"user_id": task.user_id}),
        "get_user_history:unknown": ("get_user_history", {"user_id": "user_9"}),
        "search_catalog:thriller": ("search_catalog", {"query": "thriller"}),
        "search_catalog:no_match": ("search_catalog", {"query": "zzzz qqqq"}),
        "call:unknown_tool": ("frobnicate", {}),
        "call:missing_argument": ("get_metadata", {}),
    }
    results = {key: toolkit.call(name, args) for key, (name, args) in calls.items()}
    out["tool_results.json"] = json.dumps(results, indent=2) + "\n"
    return out


@pytest.fixture(scope="module")
def rendered() -> dict[str, str]:
    return render_all()


def test_golden_set_is_complete(rendered):
    """Every golden on disk is still rendered, and vice versa."""
    if os.environ.get("UPDATE_GOLDENS"):
        pytest.skip("regenerating")
    on_disk = {p.name for p in GOLDENS.iterdir() if p.is_file()}
    assert on_disk == set(rendered), (
        f"stale: {sorted(on_disk - set(rendered))}  missing: {sorted(set(rendered) - on_disk)}"
    )


@pytest.mark.parametrize("name", sorted(render_all()))
def test_golden(name: str, rendered):
    path = GOLDENS / name
    fresh = rendered[name]
    if os.environ.get("UPDATE_GOLDENS"):
        GOLDENS.mkdir(exist_ok=True)
        path.write_text(fresh)
        return
    assert path.exists(), f"no golden at {path}; run with UPDATE_GOLDENS=1 to create it"
    assert fresh == path.read_text(), (
        f"{name} changed. If that is intended it is a new harness generation "
        f"(leaderboard/GENERATIONS.md); regenerate with UPDATE_GOLDENS=1 and say why."
    )
