from __future__ import annotations
from pydantic import BaseModel
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.data_model.conversation import ConversationTrace, StopReason

class PerConstraintResult(BaseModel):
    field: str
    op: str
    satisfied: bool

class ConstraintResult(BaseModel):
    score: float
    per_constraint: list[PerConstraintResult] = []

class ConstraintEvaluator:
    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    def evaluate(self, task: Task, trace: ConversationTrace) -> ConstraintResult:
        if task.no_valid_recommendation:
            if trace.stop_reason == StopReason.ABSTAINED:
                return ConstraintResult(score=1.0)
            else:
                return ConstraintResult(score=0.0)

        if trace.final_recommendation is None:
            return ConstraintResult(score=0.0)

        movie = self._catalog.get(trace.final_recommendation)
        if movie is None:
            return ConstraintResult(score=0.0)

        per_constraint = []
        for tc in task.constraints:
            satisfied = tc.constraint.evaluate(movie)
            per_constraint.append(PerConstraintResult(
                field=tc.constraint.field, op=tc.constraint.op, satisfied=satisfied,
            ))

        all_satisfied = all(r.satisfied for r in per_constraint)
        return ConstraintResult(score=1.0 if all_satisfied else 0.0, per_constraint=per_constraint)
