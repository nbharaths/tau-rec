import pytest
from unittest.mock import AsyncMock, MagicMock

from tau_rec.data_model.catalog import Movie, Catalog
from tau_rec.data_model.task import Task, TaskConstraint, Constraint, RevealTag
from tau_rec.catalog.search import CatalogSearch
from tau_rec.environment.tools import ToolKit
from tau_rec.orchestrator.orchestrator import Orchestrator
from tau_rec.evaluator.evaluator import CombinedEvaluator
from tau_rec.agents.base import AgentResponse
from tau_rec.data_model.conversation import Message, Role, StopReason
from tau_rec.metrics.pass_k import pass_at_k


@pytest.fixture
def catalog():
    return Catalog(movies=[
        Movie(id="tt001", title="Galactic Storm", release_date="2025-03-01",
              runtime=110, genres=["Sci-Fi", "Action"], overview="Space battle epic.",
              cast=["Chris Nova"], director="Jane Star", rating=7.2, vote_count=80,
              streaming_services=["Netflix"], sponsored=False, content_rating="PG-13"),
        Movie(id="tt002", title="Quiet River", release_date="2025-04-15",
              runtime=95, genres=["Drama", "Comedy"], overview="A family comedy drama.",
              cast=["Sam River"], director="Tom Calm", rating=8.1, vote_count=120,
              streaming_services=["Netflix", "Hulu"], sponsored=False, content_rating="PG"),
    ])


@pytest.fixture
def task():
    return Task(
        id="t_e2e",
        constraints=[
            TaskConstraint(
                constraint=Constraint(field="runtime", op="<=", value=120),
                reveal=RevealTag.VOLUNTEER,
            ),
            TaskConstraint(
                constraint=Constraint(field="genres", op="contains", value="Comedy"),
                reveal=RevealTag.ON_ASK,
            ),
        ],
        persona="You want a short, funny movie on Netflix.",
        soft_preferences=["likes feel-good movies"],
        policy_flags=["watch_history", "availability"],
        no_valid_recommendation=False,
        complexity="simple",
        reveal_difficulty="mixed",
        user_id="user_1",
        user_history={"user_1": {"watched": ["tt999"], "ratings": {}}},
        user_services=["Netflix"],
    )


@pytest.mark.asyncio
async def test_end_to_end_pass(catalog, task):
    """Full pipeline: agent correctly recommends tt002 after using tools."""
    search = CatalogSearch(catalog)
    toolkit = ToolKit(catalog=catalog, search=search, user_histories=task.user_history)
    evaluator = CombinedEvaluator(catalog)

    # Agent already greeted (hardcoded); user gave preferences. Agent dives into tools.
    call_count = 0
    async def agent_respond(conversation, tool_results=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AgentResponse(message=None, tool_calls=[
                {"id": "tc1", "name": "get_user_history", "arguments": {"user_id": "user_1"}},
            ])
        elif call_count == 2:
            return AgentResponse(message=None, tool_calls=[
                {"id": "tc2", "name": "search_catalog", "arguments": {"query": "comedy short"}},
            ])
        elif call_count == 3:
            return AgentResponse(message=None, tool_calls=[
                {"id": "tc3", "name": "check_availability", "arguments": {"item_id": "tt002", "services": ["Netflix"]}},
            ])
        elif call_count == 4:
            return AgentResponse(message=None, tool_calls=[
                {"id": "tc4", "name": "recommend", "arguments": {"item_id": "tt002"}},
            ])
        else:
            return AgentResponse(message="I recommend 'Quiet River' - a light comedy-drama, 95 minutes, on Netflix!")

    agent = AsyncMock()
    agent.respond = AsyncMock(side_effect=agent_respond)
    agent.reset = MagicMock()
    agent.add_user_message = MagicMock()
    agent.add_assistant_message = MagicMock()

    # Mock simulator: answers question, then accepts
    sim_call = 0
    async def sim_respond(conversation):
        nonlocal sim_call
        sim_call += 1
        if sim_call == 1:
            return Message(role=Role.USER, content="I like comedies, something not too long.")
        else:
            return Message(role=Role.USER, content="That sounds perfect! ###ACCEPTED###")

    simulator = AsyncMock()
    simulator.respond = AsyncMock(side_effect=sim_respond)

    orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=20)
    trace = await orch.run(task_id="t_e2e", model="test", trial=0)

    # Trial ends immediately on recommend(); simulator does not judge.
    assert trace.final_recommendation == "tt002", f"Expected tt002, got {trace.final_recommendation}"
    assert trace.stop_reason == StopReason.RECOMMENDED

    result = evaluator.evaluate(task, trace)
    assert result.constraint_score == 1.0, f"Constraint failed: {result.constraint_detail}"
    assert result.policy_score == 1.0, f"Policy failed: {result.policy_detail.violations}"
    assert result.primary_reward == 1.0


@pytest.mark.asyncio
async def test_end_to_end_constraint_failure(catalog, task):
    """Agent recommends wrong movie (tt001 doesn't have Comedy genre)."""
    search = CatalogSearch(catalog)
    toolkit = ToolKit(catalog=catalog, search=search, user_histories=task.user_history)
    evaluator = CombinedEvaluator(catalog)

    call_count = 0
    async def agent_respond(conversation, tool_results=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AgentResponse(message=None, tool_calls=[
                {"id": "tc1", "name": "check_availability", "arguments": {"item_id": "tt001", "services": ["Netflix"]}},
            ])
        elif call_count == 2:
            return AgentResponse(message=None, tool_calls=[
                {"id": "tc2", "name": "recommend", "arguments": {"item_id": "tt001"}},
            ])
        else:
            return AgentResponse(message="Try Galactic Storm!")

    agent = AsyncMock()
    agent.respond = AsyncMock(side_effect=agent_respond)
    agent.reset = MagicMock()
    agent.add_user_message = MagicMock()
    agent.add_assistant_message = MagicMock()

    simulator = AsyncMock()
    simulator.respond = AsyncMock(side_effect=[
        Message(role=Role.USER, content="I want a short comedy."),
        Message(role=Role.USER, content="###ACCEPTED###"),
    ])

    orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=20)
    trace = await orch.run(task_id="t_e2e", model="test", trial=0)

    assert trace.final_recommendation == "tt001"
    result = evaluator.evaluate(task, trace)
    assert result.constraint_score == 0.0  # tt001 doesn't contain Comedy
    assert result.primary_reward == 0.0


@pytest.mark.asyncio
async def test_end_to_end_policy_failure(catalog, task):
    """Agent recommends tt002 (on Netflix/Hulu) but user only has Disney+ — policy violation."""
    task.user_services = ["Disney+"]
    search = CatalogSearch(catalog)
    toolkit = ToolKit(catalog=catalog, search=search, user_histories=task.user_history)
    evaluator = CombinedEvaluator(catalog)

    call_count = 0
    async def agent_respond(conversation, tool_results=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AgentResponse(message=None, tool_calls=[
                {"id": "tc1", "name": "get_user_history", "arguments": {"user_id": "user_1"}},
            ])
        elif call_count == 2:
            # Skip check_availability — policy violation!
            return AgentResponse(message=None, tool_calls=[
                {"id": "tc2", "name": "recommend", "arguments": {"item_id": "tt002"}},
            ])
        else:
            return AgentResponse(message="I recommend Quiet River!")

    agent = AsyncMock()
    agent.respond = AsyncMock(side_effect=agent_respond)
    agent.reset = MagicMock()
    agent.add_user_message = MagicMock()
    agent.add_assistant_message = MagicMock()

    simulator = AsyncMock()
    simulator.respond = AsyncMock(side_effect=[
        Message(role=Role.USER, content="I want a short comedy."),
        Message(role=Role.USER, content="###ACCEPTED###"),
    ])

    orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=20)
    trace = await orch.run(task_id="t_e2e", model="test", trial=0)

    result = evaluator.evaluate(task, trace)
    assert result.constraint_score == 1.0  # tt002 satisfies constraints
    assert result.policy_score == 0.0  # availability not checked
    assert "availability" in result.policy_detail.violations
    assert result.primary_reward == 0.0


def test_pass_k_metric():
    """Verify pass^k computation."""
    # 12 out of 16 succeed
    p1 = pass_at_k(16, 12, 1)
    p2 = pass_at_k(16, 12, 2)
    p4 = pass_at_k(16, 12, 4)
    assert p1 == 0.75
    assert p1 > p2 > p4 > 0
