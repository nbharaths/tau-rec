"""Memory mode end to end: one tool, exact-match resolution, terminal either way."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tau_rec.agents.base import AgentResponse
from tau_rec.catalog.search import CatalogSearch
from tau_rec.data_model.catalog import Catalog, Movie
from tau_rec.data_model.conversation import Message, Role, StopReason
from tau_rec.data_model.task import Constraint, RevealTag, Task, TaskConstraint
from tau_rec.domain import MOVIES
from tau_rec.environment.tools import ToolKit
from tau_rec.evaluator.evaluator import CombinedEvaluator
from tau_rec.orchestrator.orchestrator import Orchestrator


def _movie(i, title, genre="Thriller", rating="R"):
    return Movie(id=i, title=title, release_date="2025-06-01", runtime=100, genres=[genre], overview="",
                 cast=[], director="D", rating=7.0, vote_count=50, streaming_services=["Netflix"],
                 content_rating=rating)


CATALOG = Catalog(items=[_movie("m1", "Heat"), _movie("m2", "Collateral", genre="Drama")])
TASK = Task(
    id="t1", persona="p", complexity="simple", reveal_difficulty="volunteer",
    constraints=[TaskConstraint(constraint=Constraint(field="genres", op="contains", value="Thriller"),
                                reveal=RevealTag.VOLUNTEER)],
    policy_flags=["recommend_tool", "single_recommendation"],
    user_history={"user_1": {"watched": [], "ratings": {}}},
)


def _kit():
    return ToolKit(CATALOG, CatalogSearch(CATALOG), {}, spec=MOVIES, mode="memory")


async def _run(tool_args: dict):
    agent = AsyncMock()
    agent.respond = AsyncMock(side_effect=[
        AgentResponse(message=None, tool_calls=[{"id": "tc1", "name": "recommend", "arguments": tool_args}]),
        AgentResponse(message="Done."),   # must never be reached: recommend is terminal
    ])
    agent.add_user_message = MagicMock()
    agent.add_assistant_message = MagicMock()
    simulator = AsyncMock()
    simulator.respond = AsyncMock(return_value=Message(role=Role.USER, content="A thriller, please."))
    kit = _kit()
    orch = Orchestrator(agent=agent, simulator=simulator, toolkit=kit, max_turns=5, spec=MOVIES)
    trace = await orch.run(task_id="t1", model="m", trial=0)
    return trace, agent


def test_memory_toolkit_exposes_exactly_one_tool():
    kit = _kit()
    defs = kit.tool_definitions()
    assert [d["function"]["name"] for d in defs] == ["recommend"]
    props = defs[0]["function"]["parameters"]["properties"]
    assert set(props) == {"title", "year", "platform"}
    # The tooled harness is not reachable from memory mode.
    assert json.loads(kit.call("search_catalog", {"query": "x"})) == {"error": "Unknown tool: search_catalog"}
    assert json.loads(kit.call("get_metadata", {"item_id": "m1"})) == {"error": "Unknown tool: get_metadata"}


@pytest.mark.asyncio
async def test_resolved_title_is_scored_as_that_item():
    trace, agent = await _run({"title": "heat"})
    assert trace.stop_reason == StopReason.RECOMMENDED
    assert trace.recommendations == ["m1"]
    assert trace.resolution.method == "exact" and trace.resolution.resolved_id == "m1"
    result = CombinedEvaluator(CATALOG).evaluate(TASK, trace)
    assert result.constraint_score == 1.0
    assert agent.respond.await_count == 1        # terminal: the second response is never requested


@pytest.mark.asyncio
async def test_unresolved_title_is_terminal_and_scores_zero():
    """A miss is recorded as given and the trial ends. Bouncing it back would
    make the resolver a catalog-membership oracle the agent could probe."""
    trace, agent = await _run({"title": "Thief"})
    assert trace.stop_reason == StopReason.RECOMMENDED
    assert trace.recommendations == ["unresolved:Thief"]
    assert trace.resolution.method == "unresolved" and trace.resolution.resolved_id is None
    assert agent.respond.await_count == 1
    result = CombinedEvaluator(CATALOG).evaluate(TASK, trace)
    assert result.constraint_score == 0.0
    # The unresolved entry counts as the one recommendation: the tool was
    # used and only once, so the tool-free policies still pass.
    assert result.policy_detail.violations == []
    # And the tool result told the agent nothing about the catalog.
    tool_result = json.loads(next(tc.result for tc in trace.tool_calls if tc.name == "recommend"))
    assert tool_result == {"status": "recommended", "title": "Thief", "note": "Recorded as given. Trial will end."}


@pytest.mark.asyncio
async def test_null_title_is_an_abstention():
    for args in ({"title": None}, {"title": "null"}, {}):
        trace, _ = await _run(args)
        assert trace.stop_reason == StopReason.ABSTAINED
        assert trace.recommendations == [] and trace.resolution is None


def test_memory_mode_traces_round_trip_and_old_traces_still_load(tmp_path):
    from pathlib import Path
    from tau_rec.data_model.conversation import ConversationTrace, Resolution
    t = ConversationTrace(task_id="t", model="m", trial=0,
                          resolution=Resolution(query="Heat", resolved_id="m1", method="exact"))
    again = ConversationTrace.model_validate_json(t.model_dump_json())
    assert again.resolution.resolved_id == "m1"
    archived = Path(__file__).parent / "fixtures" / "sample_trace.json"
    assert ConversationTrace.model_validate_json(archived.read_text()).resolution is None
