from __future__ import annotations
from abc import ABC, abstractmethod

class AgentResponse:
    def __init__(self, message: str | None, tool_calls: list[dict] | None = None):
        self.message = message
        self.tool_calls = tool_calls or []

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

class BaseAgent(ABC):
    @abstractmethod
    async def respond(self, tool_results: list[dict] | None = None) -> AgentResponse:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...
