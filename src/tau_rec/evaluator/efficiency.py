from __future__ import annotations
from pydantic import BaseModel
from tau_rec.data_model.conversation import ConversationTrace, Role

class EfficiencyMetrics(BaseModel):
    turns_to_first_recommendation: int | None
    total_tool_calls: int
    redundant_tool_call_rate: float
    rejection_count: int  # times the simulated user emitted ###REJECTED### (lower is better; ideal is 0)
    agent_step_latencies_s: list[float]
    mean_agent_step_latency_s: float | None
    trial_runtime_s: float | None

    @classmethod
    def compute(cls, trace: ConversationTrace) -> EfficiencyMetrics:
        turns_before_rec = None
        if trace.recommendations:
            from tau_rec.data_model.conversation import MessageEvent, ToolCallEvent
            agent_msg_count = 0
            for event in trace.events:
                if isinstance(event, MessageEvent) and event.role == Role.AGENT:
                    agent_msg_count += 1
                elif isinstance(event, ToolCallEvent) and event.name == "recommend":
                    break
            turns_before_rec = max(0, agent_msg_count - 1)

        seen = set()
        redundant = 0
        for tc in trace.tool_calls:
            key = (tc.name, str(sorted(tc.arguments.items())))
            if key in seen:
                redundant += 1
            seen.add(key)

        rejections = sum(
            1 for msg in trace.messages
            if msg.role == Role.USER and "###REJECTED###" in msg.content
        )

        total = len(trace.tool_calls)
        latencies = list(trace.agent_step_latencies_s)
        mean_lat = sum(latencies) / len(latencies) if latencies else None
        return cls(
            turns_to_first_recommendation=turns_before_rec,
            total_tool_calls=total,
            redundant_tool_call_rate=redundant / total if total > 0 else 0.0,
            rejection_count=rejections,
            agent_step_latencies_s=latencies,
            mean_agent_step_latency_s=mean_lat,
            trial_runtime_s=trace.trial_runtime_s,
        )
