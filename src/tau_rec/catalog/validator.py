from __future__ import annotations
from pydantic import BaseModel
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.domain import MOVIES, DomainSpec


# Complexity tier → allowed constraint counts.
# Exception: NVR tasks are exempt — their difficulty is the abstain
# decision, not constraint count.
COMPLEXITY_BANDS: dict[str, set[int]] = {
    "simple":  {1, 2},
    "medium":  {3, 4},
    "complex": {5, 6},
}


class ValidationResult(BaseModel):
    task_id: str
    solvable: bool
    solution_set_size: int
    solution_ids: list[str]
    valid_as_designed: bool
    warnings: list[str] = []
    errors: list[str] = []


class CatalogValidator:
    def __init__(self, catalog: Catalog, spec: DomainSpec = MOVIES) -> None:
        self._catalog = catalog
        self._spec = spec

    def validate_task(self, task: Task) -> ValidationResult:
        noun = self._spec.item_noun
        matching = []
        for item in self._catalog.items:
            if all(tc.constraint.evaluate(item) for tc in task.constraints):
                matching.append(item.id)

        solvable = len(matching) > 0
        if task.no_valid_recommendation:
            valid_as_designed = not solvable
        else:
            valid_as_designed = solvable

        warnings: list[str] = []
        errors: list[str] = []

        # --- Structural checks ---

        # Check 1: Solution set size
        if not task.no_valid_recommendation and len(matching) <= 2:
            warnings.append(f"near-retrieval: only {len(matching)} solutions")

        # Check 2: User history consistency
        if task.user_history:
            if task.user_id not in task.user_history:
                errors.append(f"user_id '{task.user_id}' not in user_history keys: {list(task.user_history.keys())}")
            else:
                history = task.user_history[task.user_id]
                history_key = self._spec.history_key if self._spec.history_key in history else "watched"
                consumed = history.get(history_key, [])
                for item in consumed:
                    wid = item["id"] if isinstance(item, dict) else item
                    if self._catalog.get(wid) is None:
                        errors.append(f"{self._spec.consumed_adj} {noun} '{wid}' not in catalog")

                # History trap check. `watch_history` is the legacy flag;
                # book-native tracks use the domain-neutral `history`.
                if {"watch_history", "history"} & set(task.policy_flags):
                    consumed_ids = {(item["id"] if isinstance(item, dict) else item) for item in consumed}
                    overlap = consumed_ids & set(matching)
                    if not overlap and consumed:
                        warnings.append(
                            f"history policy: no {self._spec.consumed_adj} {noun} overlaps with solution set "
                            "(trap untestable)"
                        )
                    if not consumed:
                        warnings.append("history policy: history list is empty (policy trivially satisfied)")
        else:
            if task.user_id:
                errors.append("user_history is None but user_id is set")

        # Check 3: Policy flag testability
        if not task.no_valid_recommendation:
            matching_movies = [self._catalog.get(mid) for mid in matching]
            matching_movies = [m for m in matching_movies if m is not None]

            if "sponsored" in task.policy_flags:
                has_sponsored = any(m.sponsored for m in matching_movies)
                if not has_sponsored:
                    warnings.append(f"sponsored policy: no sponsored {noun} in solution set (policy untestable)")

            if "age_restricted" in task.policy_flags:
                restricted = self._spec.restricted_ratings
                has_restricted = any(m.content_rating in restricted for m in matching_movies)
                if not has_restricted:
                    warnings.append(f"age_restricted policy: no {'/'.join(restricted)} {noun} in solution set (policy untestable)")

            if {"availability", "format_verification", "format_verification_v2"} & set(task.policy_flags):
                has_streaming = any(getattr(m, self._spec.availability_field) for m in matching_movies)
                if not has_streaming:
                    warnings.append(f"availability policy: no {noun} with {self._spec.availability_noun} in solution set")
                accepted = set(task.availability_values(self._spec))
                if not accepted:
                    errors.append(f"format/availability policy requires non-empty {self._spec.task_availability_field}")
                elif not any(accepted & set(getattr(m, self._spec.availability_field)) for m in matching_movies):
                    warnings.append(f"format/availability policy: no solution matches the user's {self._spec.availability_noun}")

        if "transparency" in task.policy_flags and not task.no_valid_recommendation:
            errors.append("transparency policy on a non-NVR task (transparency only applies to NVR tasks)")

        # Check 4: Reveal tag sanity
        reveals = [tc.reveal.value for tc in task.constraints]
        if task.reveal_difficulty == "mixed" and "on_ask" not in reveals:
            warnings.append("reveal_difficulty is 'mixed' but no on_ask constraints")
        if task.reveal_difficulty == "hidden" and "hidden" not in reveals:
            warnings.append("reveal_difficulty is 'hidden' but no hidden constraints")
        if task.reveal_difficulty == "volunteer" and set(reveals) != {"volunteer"} and reveals:
            warnings.append("reveal_difficulty is 'volunteer' but has non-volunteer constraints")

        # Check 5: Complexity-tier constraint-count band (skipped for NVR)
        if not task.no_valid_recommendation:
            band = COMPLEXITY_BANDS.get(task.complexity)
            n = len(task.constraints)
            if band is None:
                errors.append(f"unknown complexity tier '{task.complexity}'")
            elif n not in band:
                expected = sorted(band)
                errors.append(
                    f"complexity '{task.complexity}' expects {expected} constraints, got {n}"
                )

        return ValidationResult(
            task_id=task.id,
            solvable=solvable,
            solution_set_size=len(matching),
            solution_ids=matching,
            valid_as_designed=valid_as_designed,
            warnings=warnings,
            errors=errors,
        )

    def validate_all(self, tasks: list[Task]) -> list[ValidationResult]:
        return [self.validate_task(t) for t in tasks]


def task_signature(task: Task, result: ValidationResult | None = None) -> dict:
    """Capture stable, diff-friendly properties of a task for baseline snapshots.

    A baseline JSON file pairs each task_id with this signature. Re-running
    `validate --baseline <file>` reports any field that drifts, catching
    silent regressions when constraints / reveal tags / policy flags are
    edited. `constraints_sha256` pins the exact constraints and reveal tags;
    `solution_ids_sha256` pins the solution set without listing it.
    """
    import hashlib
    import json

    reveals = [tc.reveal.value for tc in task.constraints]
    constraints_blob = json.dumps(
        [{"field": tc.constraint.field, "op": tc.constraint.op, "value": tc.constraint.value, "reveal": tc.reveal.value}
         for tc in task.constraints],
        sort_keys=True,
    )
    sig = {
        "complexity": task.complexity,
        "reveal_difficulty": task.reveal_difficulty,
        "n_constraints": len(task.constraints),
        "n_hidden": sum(1 for r in reveals if r == "hidden"),
        "n_on_ask": sum(1 for r in reveals if r == "on_ask"),
        "n_volunteer": sum(1 for r in reveals if r == "volunteer"),
        "policy_flags": sorted(task.policy_flags),
        "no_valid_recommendation": task.no_valid_recommendation,
        "user_services": sorted(task.user_services),
        "constraints_sha256": hashlib.sha256(constraints_blob.encode()).hexdigest(),
    }
    if task.user_formats:
        sig["user_formats"] = sorted(task.user_formats)
    if result is not None:
        sig["solvable"] = result.solvable
        sig["solution_set_size"] = result.solution_set_size
        sig["solution_ids_sha256"] = hashlib.sha256(json.dumps(sorted(result.solution_ids)).encode()).hexdigest()
    return sig


def grid_summary(tasks: list[Task]) -> dict[str, dict[str, int]]:
    """Count tasks per (complexity, reveal_difficulty) cell."""
    grid: dict[str, dict[str, int]] = {
        cx: {rd: 0 for rd in ("volunteer", "mixed", "hidden")}
        for cx in ("simple", "medium", "complex")
    }
    for t in tasks:
        if t.complexity in grid and t.reveal_difficulty in grid[t.complexity]:
            grid[t.complexity][t.reveal_difficulty] += 1
    return grid
