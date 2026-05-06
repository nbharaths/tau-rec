from __future__ import annotations
from enum import Enum
from pydantic import BaseModel
from tau_rec.data_model.catalog import Movie

class RevealTag(str, Enum):
    VOLUNTEER = "volunteer"
    ON_ASK = "on_ask"
    HIDDEN = "hidden"

class Constraint(BaseModel):
    field: str
    op: str  # "<=", ">=", "==", "!=", "contains", "contains_any", "not_contains", "in"
    value: int | float | str | list[str]

    def evaluate(self, movie: Movie) -> bool:
        actual = getattr(movie, self.field)
        match self.op:
            case "<=":
                return actual <= self.value
            case ">=":
                return actual >= self.value
            case "==":
                return actual == self.value
            case "!=":
                return actual != self.value
            case "contains":
                return self.value in actual
            case "contains_any":
                return any(v in actual for v in self.value)
            case "not_contains":
                return self.value not in actual
            case "in":
                return actual in self.value
            case _:
                raise ValueError(f"Unknown operator: {self.op}")

class TaskConstraint(BaseModel):
    constraint: Constraint
    reveal: RevealTag

class Task(BaseModel):
    id: str
    constraints: list[TaskConstraint]
    persona: str
    soft_preferences: list[str] = []
    policy_flags: list[str] = []
    no_valid_recommendation: bool = False
    complexity: str  # "simple", "medium", "complex"
    reveal_difficulty: str  # "volunteer", "mixed", "hidden"
    user_history: dict | None = None
    user_id: str = "user_1"
    user_services: list[str] = []

    @classmethod
    def from_json(cls, path: str) -> Task:
        import json
        from pathlib import Path
        return cls(**json.loads(Path(path).read_text()))
