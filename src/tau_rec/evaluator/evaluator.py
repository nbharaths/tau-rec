from __future__ import annotations
from pydantic import BaseModel
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.data_model.conversation import ConversationTrace
from tau_rec.evaluator.constraint import ConstraintEvaluator, ConstraintResult
from tau_rec.evaluator.policy import PolicyEvaluator, PolicyResult
from tau_rec.evaluator.efficiency import EfficiencyMetrics

class TrialResult(BaseModel):
    task_id: str
    model: str
    trial: int
    constraint_score: float
    policy_score: float
    primary_reward: float
    constraint_detail: ConstraintResult
    policy_detail: PolicyResult
    efficiency: EfficiencyMetrics

class CombinedEvaluator:
    def __init__(self, catalog: Catalog) -> None:
        self._constraint_ev = ConstraintEvaluator(catalog)
        self._policy_ev = PolicyEvaluator(catalog)

    def evaluate(self, task: Task, trace: ConversationTrace) -> TrialResult:
        c = self._constraint_ev.evaluate(task, trace)
        pol = self._policy_ev.evaluate(task, trace)
        eff = EfficiencyMetrics.compute(trace)

        primary = c.score * pol.score

        return TrialResult(
            task_id=trace.task_id, model=trace.model, trial=trace.trial,
            constraint_score=c.score, policy_score=pol.score,
            primary_reward=primary,
            constraint_detail=c, policy_detail=pol, efficiency=eff,
        )
