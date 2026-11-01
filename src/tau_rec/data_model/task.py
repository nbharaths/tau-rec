from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from tau_rec.catalog.taxonomy import contains_label


class RevealTag(str, Enum):
    VOLUNTEER = "volunteer"
    ON_ASK = "on_ask"
    HIDDEN = "hidden"

class Constraint(BaseModel):
    field: str
    op: str  # "<=", ">=", "==", "!=", "contains", "contains_any", "not_contains", "in"
    value: int | float | str | list[str]

    def evaluate(self, item) -> bool:
        actual = getattr(item, self.field)
        # A constraint on an attribute the item does not carry is unsatisfied,
        # for every operator. Movies never hit this — every Movie field is
        # required — but a book with no recorded year must not satisfy
        # "publication_year >= 1990", and must not crash the evaluator mid-run.
        if actual is None:
            return False
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
                if self.field in {"languages", "countries"}:
                    return contains_label(self.field, actual, str(self.value))
                return self.value in actual
            case "contains_any":
                if self.field in {"languages", "countries"}:
                    return any(
                        contains_label(self.field, actual, str(value))
                        for value in self.value
                    )
                return any(v in actual for v in self.value)
            case "not_contains":
                if self.field in {"languages", "countries"}:
                    return not contains_label(self.field, actual, str(self.value))
                return self.value not in actual
            case "in":
                return actual in self.value
            case _:
                raise ValueError(f"Unknown operator: {self.op}")

class TaskConstraint(BaseModel):
    constraint: Constraint
    reveal: RevealTag

class Task(BaseModel):
    # A misspelled key must fail at load. Existing tracks use user_services;
    # book-native tracks use user_formats so public artifacts do not leak
    # movie terminology. DomainSpec selects the field and legacy input remains
    # supported.
    model_config = ConfigDict(extra="forbid")

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
    user_formats: list[str] = []

    def availability_values(self, spec) -> list[str]:
        primary = list(getattr(self, spec.task_availability_field, []) or [])
        legacy = list(self.user_services or [])
        if spec.task_availability_field != "user_services" and primary and legacy:
            raise ValueError(
                f"{self.id}: set {spec.task_availability_field} or user_services, not both"
            )
        return primary or legacy

    def history_values(self, spec) -> list:
        if not self.user_history or self.user_id not in self.user_history:
            return []
        history = self.user_history[self.user_id]
        return list(history.get(spec.history_key, history.get("watched", [])) or [])

    @classmethod
    def from_json(cls, path: str) -> Task:
        import json
        from pathlib import Path
        return cls(**json.loads(Path(path).read_text()))
