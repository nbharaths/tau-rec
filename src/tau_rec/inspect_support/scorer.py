"""The native ``CombinedEvaluator`` over the sample's stored trace."""
from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from tau_rec.data_model.conversation import ConversationTrace, StopReason
from tau_rec.inspect_support.solver import TRACE_KEY, environment


@scorer(metrics=[mean()])
def tau_rec_scorer() -> Scorer:
    """``primary_reward`` (1.0 or 0.0) of ``CombinedEvaluator.evaluate(task, trace)``.

    A trace with no stop reason was interrupted before the orchestrator could
    finish it (an Inspect limit delivered by cancellation, e.g. a time limit):
    it is scored as the native harness scores any early stop, ``TIMEOUT``.
    """

    async def score(state: TaskState, target: Target) -> Score:
        env = environment()
        stored = state.store.get(TRACE_KEY)
        if stored is None:
            raise RuntimeError("no τ-Rec trace in the sample store; the solver did not run")
        trace = ConversationTrace.model_validate(stored)
        if trace.stop_reason is None:
            trace.stop_reason = StopReason.TIMEOUT
            state.store.set(TRACE_KEY, trace.model_dump(mode="json"))
        result = env.evaluator.evaluate(env.tasks[trace.task_id], trace)
        return Score(
            value=result.primary_reward,
            answer=trace.final_recommendation,
            metadata={
                "constraint_score": result.constraint_score,
                "policy_score": result.policy_score,
                "stop_reason": trace.stop_reason.value,
                "recommendations": list(trace.recommendations),
            },
        )

    return score
