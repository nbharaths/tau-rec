import pytest
from unittest.mock import AsyncMock, MagicMock
from tau_rec.orchestrator.orchestrator import Orchestrator
from tau_rec.data_model.conversation import Message, Role, StopReason
from tau_rec.agents.base import AgentResponse


@pytest.mark.asyncio
async def test_orchestrator_recommend_flow():
    """Agent asks question -> user answers -> agent recommends via tool -> trial ends."""
    agent = AsyncMock()
    agent.respond = AsyncMock(side_effect=[
        # Turn 1: agent asks a question
        AgentResponse(message="What kind of movies do you like?"),
        # Turn 2: agent searches, recommends, then presents
        AgentResponse(message=None, tool_calls=[{"id": "tc1", "name": "recommend", "arguments": {"item_id": "tt001"}}]),
        AgentResponse(message="I recommend Galactic Storm!"),
    ])
    agent.reset = MagicMock()
    agent.add_user_message = MagicMock()
    agent.add_assistant_message = MagicMock()

    simulator = AsyncMock()
    simulator.respond = AsyncMock(side_effect=[
        # Response to greeting
        Message(role=Role.USER, content="I want something fun."),
        # Response to "What kind of movies?"
        Message(role=Role.USER, content="I want something under 2 hours."),
    ])

    toolkit = MagicMock()
    toolkit.call = MagicMock(return_value='{"status": "recommended", "id": "tt001", "title": "Galactic Storm"}')

    orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=20)
    trace = await orch.run(task_id="t1", model="test", trial=0)
    assert trace.stop_reason == StopReason.RECOMMENDED
    assert trace.final_recommendation == "tt001"


@pytest.mark.asyncio
async def test_orchestrator_timeout():
    """Agent never gets accepted — hits max_turns."""
    agent = AsyncMock()
    agent.respond = AsyncMock(return_value=AgentResponse(message="How about this movie?"))
    agent.reset = MagicMock()
    agent.add_user_message = MagicMock()
    agent.add_assistant_message = MagicMock()

    simulator = AsyncMock()
    simulator.respond = AsyncMock(return_value=Message(role=Role.USER, content="No, not that one. ###REJECTED###"))

    toolkit = MagicMock()

    orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=3)
    trace = await orch.run(task_id="t1", model="test", trial=0)
    assert trace.stop_reason == StopReason.TIMEOUT
    assert trace.turn_count == 4  # 1 greeting turn + 3 loop turns


@pytest.mark.asyncio
async def test_orchestrator_tool_calls():
    """Agent makes a tool call, then responds with text."""
    agent = AsyncMock()
    agent.respond = AsyncMock(side_effect=[
        AgentResponse(message=None, tool_calls=[{"id": "tc1", "name": "search_catalog", "arguments": {"query": "sci-fi"}}]),
        AgentResponse(message="I found some options!"),
    ])
    agent.reset = MagicMock()
    agent.add_user_message = MagicMock()
    agent.add_assistant_message = MagicMock()

    simulator = AsyncMock()
    simulator.respond = AsyncMock(side_effect=[
        Message(role=Role.USER, content="I want sci-fi."),
        Message(role=Role.USER, content="###ACCEPTED###"),
    ])

    toolkit = MagicMock()
    toolkit.call = MagicMock(return_value='[{"id":"tt001","title":"Galactic Storm"}]')

    orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=1)
    trace = await orch.run(task_id="t1", model="test", trial=0)
    assert trace.tool_call_count == 1
    toolkit.call.assert_called_once_with("search_catalog", {"query": "sci-fi"})


@pytest.mark.asyncio
async def test_orchestrator_recommend_tool_registers_recommendation():
    """Calling recommend(item_id) registers the recommendation."""
    call_count = 0
    async def agent_respond(conversation, tool_results=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AgentResponse(message=None, tool_calls=[
                {"id": "tc1", "name": "recommend", "arguments": {"item_id": "tt001"}},
            ])
        else:
            return AgentResponse(message="I recommend Galactic Storm!")

    agent = AsyncMock()
    agent.respond = AsyncMock(side_effect=agent_respond)
    agent.reset = MagicMock()
    agent.add_user_message = MagicMock()
    agent.add_assistant_message = MagicMock()

    simulator = AsyncMock()
    simulator.respond = AsyncMock(side_effect=[
        Message(role=Role.USER, content="I like action."),
        Message(role=Role.USER, content="###ACCEPTED###"),
    ])

    toolkit = MagicMock()
    toolkit.call = MagicMock(return_value='{"status": "recommended", "id": "tt001", "title": "Galactic Storm"}')

    orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=20)
    trace = await orch.run(task_id="t1", model="test", trial=0)
    assert trace.final_recommendation == "tt001"


@pytest.mark.asyncio
async def test_orchestrator_no_recommend_and_no_title_means_no_recommendation():
    """If agent never calls recommend() and doesn't mention titles, no rec registered."""
    agent = AsyncMock()
    agent.respond = AsyncMock(side_effect=[
        AgentResponse(message=None, tool_calls=[{"id": "tc1", "name": "search_catalog", "arguments": {"query": "comedy"}}]),
        AgentResponse(message="I'm still looking for something good for you."),
    ])
    agent.reset = MagicMock()
    agent.add_user_message = MagicMock()
    agent.add_assistant_message = MagicMock()

    simulator = AsyncMock()
    simulator.respond = AsyncMock(side_effect=[
        Message(role=Role.USER, content="I want comedy."),
        Message(role=Role.USER, content="###ACCEPTED###"),
    ])

    toolkit = MagicMock()
    toolkit.call = MagicMock(return_value='[{"id":"tt002","title":"Quiet River"}]')

    orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=1)
    trace = await orch.run(task_id="t1", model="test", trial=0)
    assert trace.final_recommendation is None


@pytest.mark.asyncio
async def test_orchestrator_stops_on_recommend():
    """Trial ends immediately when the agent calls recommend(); simulator does not judge."""
    call_count = 0
    async def agent_respond(conversation, tool_results=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AgentResponse(message=None, tool_calls=[
                {"id": "tc1", "name": "recommend", "arguments": {"item_id": "tt001"}},
            ])
        else:
            return AgentResponse(message="I recommend Galactic Storm!")

    agent = AsyncMock()
    agent.respond = AsyncMock(side_effect=agent_respond)
    agent.reset = MagicMock()
    agent.add_user_message = MagicMock()
    agent.add_assistant_message = MagicMock()

    simulator = AsyncMock()
    # Simulator should only be called once (pre-greeting response); stop happens before simulator
    # sees the recommendation.
    simulator.respond = AsyncMock(side_effect=[
        Message(role=Role.USER, content="I like action."),
    ])

    toolkit = MagicMock()
    toolkit.call = MagicMock(return_value='{"status": "recommended", "id": "tt001", "title": "Galactic Storm"}')

    orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=20)
    trace = await orch.run(task_id="t1", model="test", trial=0)
    assert trace.final_recommendation == "tt001"
    assert trace.stop_reason == StopReason.RECOMMENDED
    # Simulator should have been called exactly once (for the pre-loop user intro)
    assert simulator.respond.call_count == 1


@pytest.mark.asyncio
async def test_orchestrator_empty_response():
    """Agent returns empty response (no message, no tools) -- turn is skipped."""
    call_count = 0

    async def agent_respond(conversation, tool_results=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return AgentResponse(message=None, tool_calls=[])
        return AgentResponse(message="Hello!")

    agent = AsyncMock()
    agent.respond = AsyncMock(side_effect=agent_respond)
    agent.reset = MagicMock()
    agent.add_user_message = MagicMock()
    agent.add_assistant_message = MagicMock()

    simulator = AsyncMock()
    simulator.respond = AsyncMock(return_value=Message(role=Role.USER, content="Hmm."))

    toolkit = MagicMock()

    # max_turns=3 so the loop runs through: empty, empty, "Hello!"
    orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=3)
    trace = await orch.run(task_id="t1", model="test", trial=0)
    # Agent never called recommend() — trial hits max_turns
    assert trace.stop_reason == StopReason.TIMEOUT
    # Greeting turn + one turn where "Hello!" was emitted and simulator responded
    # = 2 agent messages + 2 user messages that completed full turns
    assert trace.turn_count == 2
