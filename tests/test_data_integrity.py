from __future__ import annotations
import json
from pathlib import Path
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.catalog.validator import CatalogValidator

_ROOT = Path(__file__).resolve().parent.parent
_CATALOG = Catalog.from_json(str(_ROOT / "data/catalog.json"))
_TASKS = [Task.from_json(str(p)) for p in sorted((_ROOT / "data/tasks").glob("*.json"))]
_ANSWERS = json.loads((_ROOT / "data/answers.json").read_text())


# ---------------------------------------------------------------------------
# Section 1 — Reachability
# ---------------------------------------------------------------------------

def test_availability_tasks_have_nonempty_user_services():
    """Tasks with availability policy flag must specify at least one streaming service."""
    problems = [
        task.id
        for task in _TASKS
        if not task.no_valid_recommendation
        and "availability" in task.policy_flags
        and not task.user_services
    ]
    assert not problems, (
        f"Tasks with 'availability' flag but empty user_services: {problems}"
    )


def test_availability_tasks_have_reachable_solutions():
    """Tasks with availability flag must have at least one solution streamable on user_services."""
    problems = [
        task.id
        for task in _TASKS
        if not task.no_valid_recommendation
        and "availability" in task.policy_flags
        and not _ANSWERS.get(task.id, {}).get("reachable_solutions")
    ]
    assert not problems, (
        f"Tasks with 'availability' flag but no reachable solutions: {problems}"
    )


def test_non_availability_non_nvr_tasks_are_catalog_solvable():
    """Non-NVR tasks without availability flag must have at least one catalog solution."""
    problems = [
        task.id
        for task in _TASKS
        if not task.no_valid_recommendation
        and "availability" not in task.policy_flags
        and not _ANSWERS.get(task.id, {}).get("constraint_solutions")
    ]
    assert not problems, (
        f"Non-NVR tasks with no constraint_solutions in answers.json: {problems}"
    )

# ---------------------------------------------------------------------------
# Section 2 — Cross-constraint contradictions
# ---------------------------------------------------------------------------

_NUMERIC_FIELDS = {"runtime", "rating", "vote_count"}
_RESTRICTED_RATINGS = {"R", "NC-17"}


def test_no_impossible_numeric_ranges():
    """No task should have an upper bound strictly less than a lower bound on the same field."""
    problems = []
    for task in _TASKS:
        constraints = [tc.constraint for tc in task.constraints]
        upper = {
            c.field: c.value
            for c in constraints
            if c.op == "<=" and c.field in _NUMERIC_FIELDS
        }
        lower = {
            c.field: c.value
            for c in constraints
            if c.op == ">=" and c.field in _NUMERIC_FIELDS
        }
        for field in upper:
            if field in lower and upper[field] < lower[field]:
                problems.append(
                    f"{task.id}: {field} <= {upper[field]} AND >= {lower[field]}"
                )
    assert not problems, f"Impossible numeric ranges found: {problems}"


def test_age_restricted_flag_compatible_with_content_rating_constraint():
    """
    If a task has age_restricted in policy_flags, its content_rating constraints
    must not make it impossible for an R or NC-17 movie to satisfy them.
    """
    problems = []
    for task in _TASKS:
        if "age_restricted" not in task.policy_flags:
            continue
        for tc in task.constraints:
            c = tc.constraint
            if c.field != "content_rating":
                continue
            if c.op == "in":
                allowed = c.value if isinstance(c.value, list) else [c.value]
                if not _RESTRICTED_RATINGS & set(allowed):
                    problems.append(
                        f"{task.id}: age_restricted flag but content_rating 'in' "
                        f"constraint excludes R and NC-17 (allowed: {allowed})"
                    )
            elif c.op == "==":
                if c.value not in _RESTRICTED_RATINGS:
                    problems.append(
                        f"{task.id}: age_restricted flag but content_rating == {c.value!r} "
                        f"(not R or NC-17)"
                    )
            elif c.op == "!=":
                if c.value in _RESTRICTED_RATINGS:
                    problems.append(
                        f"{task.id}: age_restricted flag but content_rating != {c.value!r} "
                        f"excludes a restricted rating"
                    )
    assert not problems, (
        f"age_restricted flag contradicted by content_rating constraint: {problems}"
    )


def test_no_duplicate_constraints():
    """No task should list the same (field, op, value) constraint more than once."""
    problems = []
    for task in _TASKS:
        seen: set[tuple] = set()
        for tc in task.constraints:
            c = tc.constraint
            key = (c.field, c.op, str(c.value))
            if key in seen:
                problems.append(f"{task.id}: duplicate ({c.field}, {c.op}, {c.value})")
            seen.add(key)
    assert not problems, f"Duplicate constraints found: {problems}"

# ---------------------------------------------------------------------------
# Section 3 — answers.json sync
# ---------------------------------------------------------------------------

def test_answers_json_matches_live_catalog():
    """
    The constraint_solutions in answers.json must exactly match what CatalogValidator
    computes from the current catalog. Detects drift if tasks were edited after the
    answer key was last regenerated.
    """
    validator = CatalogValidator(_CATALOG)
    problems = []
    for task in _TASKS:
        result = validator.validate_task(task)
        live = set(result.solution_ids)
        recorded = set(_ANSWERS.get(task.id, {}).get("constraint_solutions", []))
        if live != recorded:
            extra = live - recorded
            missing = recorded - live
            problems.append(
                f"{task.id}: answers.json out of sync "
                f"(+{len(extra)} in catalog not in answers, "
                f"-{len(missing)} in answers not in catalog)"
            )
    assert not problems, (
        "answers.json has drifted from the current catalog:\n"
        + "\n".join(problems)
        + "\nFix: run `uv run python scripts/regenerate_answers.py` from the repo root."
    )
