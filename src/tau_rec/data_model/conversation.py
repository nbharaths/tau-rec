from __future__ import annotations
from enum import Enum
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

class Role(str, Enum):
    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"

class StopReason(str, Enum):
    TIMEOUT = "timeout"
    RECOMMENDED = "recommended"  # agent called recommend() — trial ends immediately, simulator does not judge
    ABSTAINED = "abstained"  # agent called recommend() with no item_id — explicit abstention

class Message(BaseModel):
    role: Role
    content: str

class ToolCall(BaseModel):
    name: str
    arguments: dict
    result: str

class MessageEvent(BaseModel):
    type: Literal["message"] = "message"
    role: Role
    content: str

class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    name: str
    arguments: dict
    result: str

Event = Annotated[Union[MessageEvent, ToolCallEvent], Field(discriminator="type")]

class ConversationTrace(BaseModel):
    task_id: str
    model: str
    trial: int
    events: list[Event] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    stop_reason: StopReason | None = None
    agent_step_latencies_s: list[float] = Field(default_factory=list)
    trial_runtime_s: float | None = None

    def add_message(self, message: Message) -> None:
        self.events.append(MessageEvent(role=message.role, content=message.content))

    def add_tool_call(self, tool_call: ToolCall) -> None:
        self.events.append(ToolCallEvent(
            name=tool_call.name, arguments=tool_call.arguments, result=tool_call.result,
        ))

    def add_recommendation(self, item_id: str) -> None:
        self.recommendations.append(item_id)

    @property
    def messages(self) -> list[Message]:
        return [
            Message(role=e.role, content=e.content)
            for e in self.events if isinstance(e, MessageEvent)
        ]

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [
            ToolCall(name=e.name, arguments=e.arguments, result=e.result)
            for e in self.events if isinstance(e, ToolCallEvent)
        ]

    @property
    def final_recommendation(self) -> str | None:
        return self.recommendations[-1] if self.recommendations else None

    @property
    def all_recommendations(self) -> list[str]:
        return self.recommendations

    @property
    def turn_count(self) -> int:
        agent_msgs = sum(1 for e in self.events if isinstance(e, MessageEvent) and e.role == Role.AGENT)
        user_msgs = sum(1 for e in self.events if isinstance(e, MessageEvent) and e.role == Role.USER)
        return min(agent_msgs, user_msgs)

    @property
    def tool_call_count(self) -> int:
        return sum(1 for e in self.events if isinstance(e, ToolCallEvent))
