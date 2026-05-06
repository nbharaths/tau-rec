from __future__ import annotations
from pydantic import BaseModel
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task


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
    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog

    def validate_task(self, task: Task) -> ValidationResult:
        matching = []
        for movie in self._catalog.movies:
            if all(tc.constraint.evaluate(movie) for tc in task.constraints):
                matching.append(movie.id)

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
                watched = history.get("watched", [])
                for item in watched:
                    wid = item["id"] if isinstance(item, dict) else item
                    if self._catalog.get(wid) is None:
                        errors.append(f"watched movie '{wid}' not in catalog")

                # Watch history trap check
                if "watch_history" in task.policy_flags:
                    watched_ids = {(item["id"] if isinstance(item, dict) else item) for item in watched}
                    overlap = watched_ids & set(matching)
                    if not overlap and watched:
                        warnings.append("watch_history policy: no watched movie overlaps with solution set (trap untestable)")
                    if not watched:
                        warnings.append("watch_history policy: watched list is empty (policy trivially satisfied)")
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
                    warnings.append("sponsored policy: no sponsored movie in solution set (policy untestable)")

            if "age_restricted" in task.policy_flags:
                has_restricted = any(m.content_rating in ("R", "NC-17") for m in matching_movies)
                if not has_restricted:
                    warnings.append("age_restricted policy: no R/NC-17 movie in solution set (policy untestable)")

            if "availability" in task.policy_flags:
                has_streaming = any(m.streaming_services for m in matching_movies)
                if not has_streaming:
                    warnings.append("availability policy: no movie with streaming services in solution set")

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


def task_signature(task: Task, result: ValidationResult) -> dict:
    """Capture stable, diff-friendly properties of a task for baseline snapshots.

    A baseline JSON file pairs each task_id with this signature. Re-running
    `validate --baseline <file>` reports any field that drifts, catching
    silent regressions when constraints / reveal tags / policy flags are
    edited.
    """
    reveals = [tc.reveal.value for tc in task.constraints]
    return {
        "complexity": task.complexity,
        "reveal_difficulty": task.reveal_difficulty,
        "n_constraints": len(task.constraints),
        "n_hidden": sum(1 for r in reveals if r == "hidden"),
        "n_on_ask": sum(1 for r in reveals if r == "on_ask"),
        "n_volunteer": sum(1 for r in reveals if r == "volunteer"),
        "policy_flags": sorted(task.policy_flags),
        "no_valid_recommendation": task.no_valid_recommendation,
        "solvable": result.solvable,
        "solution_set_size": result.solution_set_size,
    }


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
