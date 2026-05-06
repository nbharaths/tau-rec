from tau_rec.agents.base import BaseAgent
from tau_rec.agents.litellm_agent import LiteLLMAgent

def test_base_agent_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        BaseAgent()

def test_litellm_agent_init():
    agent = LiteLLMAgent(
        model="gpt-4o",
        system_prompt="You are a movie assistant.",
        tool_definitions=[],
    )
    assert agent.model == "gpt-4o"

def test_litellm_agent_add_user_message():
    agent = LiteLLMAgent(
        model="gpt-4o",
        system_prompt="test",
        tool_definitions=[],
    )
    agent.add_user_message("hello")
    assert len(agent._message_history) == 1
    assert agent._message_history[0]["role"] == "user"
    assert agent._message_history[0]["content"] == "hello"

def test_litellm_agent_reset():
    agent = LiteLLMAgent(
        model="gpt-4o",
        system_prompt="test",
        tool_definitions=[],
    )
    agent.add_user_message("hello")
    agent.reset()
    assert len(agent._message_history) == 0
