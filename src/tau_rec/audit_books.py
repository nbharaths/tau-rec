"""Automated census and empirical triage for a controlled books artifact.

This module deliberately separates objective artifact validity from model
performance.  A model failure can add a family to the review queue, but it can
never make a satisfiable task "impossible"; only the catalog-grounded solver
answers that question.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tau_rec.catalog.validator import CatalogValidator
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import RevealTag, Task
from tau_rec.domain import BOOKS
from tau_rec.leaderboard.hashing import hash_file, hash_task_dir
from tau_rec.metrics.pass_k import pass_at_k
from tau_rec.simulator.user_simulator import BookDisclosureController


AUDIT_VERSION = "books-wikidata-automated-audit-v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.80
DEFAULT_SOLUTION_OVERLAP_THRESHOLD = 0.95


@dataclass(frozen=True)
class Artifact:
    root: Path
    catalog: Catalog
    tasks: dict[str, Task]
    answers: dict[str, dict[str, Any]]
    family_manifest: dict[str, Any]
    task_manifest: dict[str, Any]
    task_to_family: dict[str, str]
    task_to_reveal: dict[str, str]

    @property
    def families(self) -> list[dict[str, Any]]:
        return self.family_manifest["families"]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifact(root: Path) -> Artifact:
    tasks = {
        path.stem: Task.from_json(str(path))
        for path in sorted((root / "tasks").glob("*.json"))
    }
    family_manifest = _load_json(root / "family_manifest.json")
    task_to_family: dict[str, str] = {}
    task_to_reveal: dict[str, str] = {}
    for family in family_manifest["families"]:
        for reveal, task_id in family["task_ids"].items():
            task_to_family[task_id] = family["family_id"]
            task_to_reveal[task_id] = reveal
    return Artifact(
        root=root,
        catalog=Catalog.from_json(str(root / "catalog.json"), BOOKS),
        tasks=tasks,
        answers=_load_json(root / "answers.json"),
        family_manifest=family_manifest,
        task_manifest=_load_json(root / "task_manifest.json"),
        task_to_family=task_to_family,
        task_to_reveal=task_to_reveal,
    )


def _sha256_texts(texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _percentile(values: Iterable[float], q: float) -> float | None:
    rows = list(values)
    if not rows:
        return None
    return float(np.percentile(np.asarray(rows, dtype=float), q * 100))


def distribution_summary(values: Iterable[str]) -> dict[str, Any]:
    counts = Counter(values)
    total = sum(counts.values())
    probabilities = [count / total for count in counts.values()] if total else []
    entropy = -sum(p * math.log2(p) for p in probabilities if p)
    normalized = (
        entropy / math.log2(len(counts)) if len(counts) > 1 else 1.0 if counts else 0.0
    )
    return {
        "total": total,
        "unique": len(counts),
        "entropy_bits": entropy,
        "normalized_entropy": normalized,
        "effective_categories": 2**entropy,
        "counts": dict(sorted(counts.items(), key=lambda row: (-row[1], row[0]))),
    }


def _numeric_summary(values: Iterable[int | float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    if not rows:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "count": len(rows),
        "min": min(rows),
        "p25": _percentile(rows, 0.25),
        "median": _percentile(rows, 0.5),
        "p75": _percentile(rows, 0.75),
        "max": max(rows),
    }


def _constraint_text(task: Task) -> str:
    return "; ".join(
        f"{tagged.constraint.field} {tagged.constraint.op} {tagged.constraint.value}"
        for tagged in task.constraints
    )


def _family_text(task: Task) -> str:
    # Persona and policy diversity are measured exactly in the coverage census.
    # Excluding their repeated boilerplate here makes the semantic view about
    # the selection problem itself rather than shared benchmark phrasing.
    return _constraint_text(task)


def _opening_text(task: Task) -> str:
    return BookDisclosureController(task).respond("").content


def _solution_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 20:
        return "6-20"
    if count <= 100:
        return "21-100"
    if count <= 500:
        return "101-500"
    return "501+"


def _constraint_template(task: Task) -> str:
    return "|".join(
        f"{tagged.constraint.field}:{tagged.constraint.op}"
        for tagged in task.constraints
    )


def _solution_set_diversity(
    artifact: Artifact,
    *,
    threshold: float = DEFAULT_SOLUTION_OVERLAP_THRESHOLD,
) -> dict[str, Any]:
    families = []
    for family in artifact.families:
        task_id = family["task_ids"]["volunteer"]
        solutions = set(artifact.answers[task_id]["constraint_solutions"])
        if solutions:
            families.append((family["family_id"], solutions))
    pairs = []
    similarities = []
    for left in range(len(families)):
        left_id, left_solutions = families[left]
        for right in range(left + 1, len(families)):
            right_id, right_solutions = families[right]
            similarity = len(left_solutions & right_solutions) / len(
                left_solutions | right_solutions
            )
            similarities.append(similarity)
            if similarity >= threshold:
                pairs.append({
                    "left_family_id": left_id,
                    "right_family_id": right_id,
                    "jaccard": similarity,
                    "left_solution_count": len(left_solutions),
                    "right_solution_count": len(right_solutions),
                    "identical": left_solutions == right_solutions,
                })
    pairs.sort(key=lambda row: (-row["jaccard"], row["left_family_id"], row["right_family_id"]))
    return {
        "compared_pairs": len(similarities),
        "threshold": threshold,
        "pairs_above_threshold": len(pairs),
        "identical_solution_set_pairs": sum(row["identical"] for row in pairs),
        "pair_jaccard": _numeric_summary(similarities),
        "flagged_pairs": pairs,
    }


def coverage_report(artifact: Artifact) -> dict[str, Any]:
    representatives = [
        artifact.tasks[family["task_ids"]["volunteer"]]
        for family in artifact.families
    ]
    constraint_fields: list[str] = []
    constraint_values: list[str] = []
    year_decades: list[str] = []
    policy_flags: list[str] = []
    policy_combinations: list[str] = []
    for task in representatives:
        policy_flags.extend(task.policy_flags)
        policy_combinations.append("+".join(sorted(task.policy_flags)))
        for tagged in task.constraints:
            constraint = tagged.constraint
            constraint_fields.append(constraint.field)
            constraint_values.append(
                f"{constraint.field}:{constraint.op}:{constraint.value}"
            )
            if constraint.field == "publication_year":
                year_decades.append(f"{int(constraint.value) // 10 * 10}s")

    solution_counts = {
        family["family_id"]: len(
            artifact.answers[family["task_ids"]["volunteer"]][
                "constraint_solutions"
            ]
        )
        for family in artifact.families
    }
    by_complexity = {
        complexity: _numeric_summary(
            solution_counts[family["family_id"]]
            for family in artifact.families
            if family["complexity"] == complexity
        )
        for complexity in ("simple", "medium", "complex")
    }
    catalog = artifact.catalog.items
    catalog_genres = [genre for item in catalog for genre in item.genres]
    catalog_languages = [language for item in catalog for language in item.languages]
    catalog_countries = [country for item in catalog for country in item.countries]
    catalog_decades = [
        f"{item.publication_year // 10 * 10}s"
        for item in catalog
        if item.publication_year is not None
    ]
    authors = [author for item in catalog for author in item.authors]

    return {
        "family_count": len(artifact.families),
        "task_count": len(artifact.tasks),
        "catalog_count": len(catalog),
        "complexity": distribution_summary(
            family["complexity"] for family in artifact.families
        ),
        "nvr": distribution_summary(
            "nvr" if family["no_valid_recommendation"] else "satisfiable"
            for family in artifact.families
        ),
        "personas": distribution_summary(task.persona for task in representatives),
        "constraint_templates": distribution_summary(
            _constraint_template(task) for task in representatives
        ),
        "constraint_fields": distribution_summary(constraint_fields),
        "constraint_values": distribution_summary(constraint_values),
        "publication_year_decades": distribution_summary(year_decades),
        "policy_flags": distribution_summary(policy_flags),
        "policy_combinations": distribution_summary(policy_combinations),
        "accepted_format_combinations": distribution_summary(
            "+".join(sorted(task.user_formats)) if task.user_formats else "none"
            for task in representatives
        ),
        "solution_counts": {
            "overall": _numeric_summary(solution_counts.values()),
            "by_complexity": by_complexity,
            "buckets": distribution_summary(
                _solution_bucket(count) for count in solution_counts.values()
            ),
        },
        "solution_set_diversity": _solution_set_diversity(artifact),
        "catalog": {
            "genres": distribution_summary(catalog_genres),
            "languages": distribution_summary(catalog_languages),
            "countries": distribution_summary(catalog_countries),
            "publication_year_decades": distribution_summary(catalog_decades),
            "unique_authors": len(set(authors)),
            "items_with_multiple_authors": sum(len(item.authors) > 1 for item in catalog),
            "items_with_multiple_genres": sum(len(item.genres) > 1 for item in catalog),
            "items_missing_country": sum(not item.countries for item in catalog),
        },
    }


def _hidden_leakage_checks(artifact: Artifact) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for family in artifact.families:
        mixed = artifact.tasks[family["task_ids"]["mixed"]]
        hidden = artifact.tasks[family["task_ids"]["hidden"]]
        mixed_opening = _opening_text(mixed)
        hidden_opening = _opening_text(hidden)
        if mixed_opening != hidden_opening:
            issues.append({
                "family_id": family["family_id"],
                "reason": "mixed_hidden_openings_differ",
                "detail": "matched variants do not render the same opening disclosure",
            })
        normalized = hidden_opening.casefold()
        for tagged in hidden.constraints:
            if tagged.reveal != RevealTag.HIDDEN:
                continue
            value = str(tagged.constraint.value).casefold()
            if value and value in normalized:
                issues.append({
                    "family_id": family["family_id"],
                    "reason": "hidden_value_in_opening",
                    "detail": f"hidden value {tagged.constraint.value!r} appears in opening",
                })
    return issues


def deterministic_validation(
    artifact: Artifact,
    *,
    independent_validator: Path | None,
) -> dict[str, Any]:
    validator = CatalogValidator(artifact.catalog, BOOKS)
    errors: list[str] = []
    warnings: list[str] = []
    for task in artifact.tasks.values():
        result = validator.validate_task(task)
        errors.extend(f"{task.id}: {message}" for message in result.errors)
        warnings.extend(f"{task.id}: {message}" for message in result.warnings)
        expected = artifact.answers.get(task.id, {}).get("constraint_solutions")
        if expected != result.solution_ids:
            errors.append(f"{task.id}: answer key differs from CatalogValidator")

    leakage = _hidden_leakage_checks(artifact)
    errors.extend(
        f"{row['family_id']}: {row['reason']}: {row['detail']}" for row in leakage
    )
    independent: dict[str, Any] = {"status": "skipped"}
    if independent_validator is not None:
        command = [
            sys.executable,
            str(independent_validator),
            "--catalog",
            str(artifact.root / "catalog.json"),
            "--artifact",
            str(artifact.root),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        independent = {
            "status": "pass" if completed.returncode == 0 else "fail",
            "returncode": completed.returncode,
            "summary": (completed.stdout or completed.stderr).strip().splitlines()[-1:],
        }
        if completed.returncode:
            errors.append("independent controlled-artifact validator failed")
    return {
        "status": "pass" if not errors else "fail",
        "task_count": len(artifact.tasks),
        "errors": errors,
        "warnings": warnings,
        "hidden_leakage_issues": leakage,
        "independent_validator": independent,
    }


def _normalise_embeddings(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _solution_jaccard(
    left_family: str,
    right_family: str,
    family_solutions: dict[str, set[str]],
) -> float | None:
    left = family_solutions[left_family]
    right = family_solutions[right_family]
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _embedding_pair_report(
    matrix: np.ndarray,
    records: list[dict[str, str]],
    *,
    threshold: float,
    top_pairs: int,
    family_solutions: dict[str, set[str]],
    same_view_only: bool,
) -> dict[str, Any]:
    similarities = matrix @ matrix.T
    eligible: list[tuple[float, int, int]] = []
    nearest = [-1.0] * len(records)
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if records[left]["family_id"] == records[right]["family_id"]:
                continue
            if same_view_only and records[left].get("view") != records[right].get("view"):
                continue
            similarity = float(similarities[left, right])
            eligible.append((similarity, left, right))
            nearest[left] = max(nearest[left], similarity)
            nearest[right] = max(nearest[right], similarity)
    eligible.sort(
        key=lambda row: (
            -row[0], records[row[1]]["id"], records[row[2]]["id"]
        )
    )

    def render(row: tuple[float, int, int]) -> dict[str, Any]:
        similarity, left, right = row
        left_record = records[left]
        right_record = records[right]
        return {
            "left": left_record,
            "right": right_record,
            "cosine_similarity": similarity,
            "solution_jaccard": _solution_jaccard(
                left_record["family_id"],
                right_record["family_id"],
                family_solutions,
            ),
        }

    above = [row for row in eligible if row[0] >= threshold]
    # Connected components summarize dense near-duplicate neighborhoods.
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for _, left, right in above:
        union(left, right)
    components: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        components[find(index)].append(record["id"])
    clusters = sorted(
        (sorted(values) for values in components.values() if len(values) > 1),
        key=lambda values: (-len(values), values),
    )
    valid_nearest = [value for value in nearest if value >= -0.5]
    return {
        "records": len(records),
        "compared_pairs": len(eligible),
        "threshold": threshold,
        "pairs_above_threshold": len(above),
        "pair_similarity": _numeric_summary(row[0] for row in eligible),
        "nearest_neighbor_similarity": _numeric_summary(valid_nearest),
        "top_pairs": [render(row) for row in eligible[:top_pairs]],
        "flagged_pairs": [render(row) for row in above[:100]],
        "flagged_pairs_truncated": max(0, len(above) - 100),
        "clusters": clusters[:50],
        "clusters_truncated": max(0, len(clusters) - 50),
    }


def _center_by_view(
    matrix: np.ndarray,
    records: list[dict[str, str]],
) -> np.ndarray:
    """Remove the common embedding direction separately for each text view."""
    centered = np.zeros_like(matrix, dtype=np.float32)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.get("view", "all")].append(index)
    for indices in groups.values():
        block = matrix[indices]
        centered[indices] = block - block.mean(axis=0, keepdims=True)
    return _normalise_embeddings(centered)


def _exact_text_groups(
    texts: list[str], records: list[dict[str, str]], *, limit: int = 50
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for text, record in zip(texts, records):
        groups[text].append(record["id"])
    duplicates = sorted(
        (sorted(ids) for ids in groups.values() if len(ids) > 1),
        key=lambda ids: (-len(ids), ids),
    )
    return {
        "group_count": len(duplicates),
        "record_count": sum(len(ids) for ids in duplicates),
        "groups": duplicates[:limit],
        "groups_truncated": max(0, len(duplicates) - limit),
    }


def embedding_diversity(
    artifact: Artifact,
    *,
    model_name: str,
    threshold: float,
    top_pairs: int,
) -> dict[str, Any]:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError(
            "embedding audit requires the audit extra: uv sync --extra audit"
        ) from exc

    family_records: list[dict[str, str]] = []
    family_texts: list[str] = []
    opening_records: list[dict[str, str]] = []
    opening_texts: list[str] = []
    family_solutions: dict[str, set[str]] = {}
    for family in artifact.families:
        family_id = family["family_id"]
        volunteer_id = family["task_ids"]["volunteer"]
        volunteer = artifact.tasks[volunteer_id]
        family_records.append({"id": family_id, "family_id": family_id})
        family_texts.append(_family_text(volunteer))
        family_solutions[family_id] = set(
            artifact.answers[volunteer_id]["constraint_solutions"]
        )
        for reveal in ("volunteer", "mixed", "hidden"):
            task_id = family["task_ids"][reveal]
            opening_records.append({
                "id": task_id,
                "family_id": family_id,
                "view": reveal,
            })
            opening_texts.append(_opening_text(artifact.tasks[task_id]))

    texts = family_texts + opening_texts
    model = TextEmbedding(model_name=model_name)
    raw = np.asarray(list(model.embed(texts)), dtype=np.float32)
    family_raw = raw[: len(family_texts)]
    opening_raw = raw[len(family_texts) :]
    family_matrix = _normalise_embeddings(
        family_raw - family_raw.mean(axis=0, keepdims=True)
    )
    opening_matrix = _center_by_view(opening_raw, opening_records)
    matrix = np.vstack((family_matrix, opening_matrix))
    try:
        package_version = version("fastembed")
    except PackageNotFoundError:  # pragma: no cover
        package_version = "unknown"
    return {
        "status": "complete",
        "model": model_name,
        "fastembed_version": package_version,
        "dimension": int(matrix.shape[1]),
        "text_count": len(texts),
        "similarity_space": (
            "mean-centered and L2-normalized; openings centered separately by reveal view"
        ),
        "text_sha256": _sha256_texts(texts),
        "normalized_embedding_sha256": hashlib.sha256(
            matrix.astype("<f4", copy=False).tobytes()
        ).hexdigest(),
        "family_semantics": {
            **_embedding_pair_report(
                family_matrix,
                family_records,
                threshold=threshold,
                top_pairs=top_pairs,
                family_solutions=family_solutions,
                same_view_only=False,
            ),
            "exact_duplicate_text": _exact_text_groups(
                family_texts, family_records
            ),
        },
        "user_openings": {
            **_embedding_pair_report(
                opening_matrix,
                opening_records,
                threshold=threshold,
                top_pairs=top_pairs,
                family_solutions=family_solutions,
                same_view_only=True,
            ),
            "exact_duplicate_text": _exact_text_groups(
                opening_texts, opening_records
            ),
        },
    }


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(path)


def _safe_json(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _trial_trace(run_dir: Path, task_id: str, trial: int) -> dict[str, Any]:
    path = run_dir / "traces" / f"{task_id}_trial{trial}.json"
    return _load_json(path) if path.is_file() else {}


def _trial_observation(
    row: dict[str, Any],
    task: Task,
    answer: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    stop_reason = str(trace.get("stop_reason", "missing"))
    recommendations = trace.get("recommendations") or []
    tool_errors = 0
    empty_searches = 0
    saw_solution = False
    saw_policy_eligible = False
    provider_cost = 0.0
    solutions = set(answer.get("constraint_solutions", []))
    eligible = set(answer.get("policy_eligible_solutions", []))
    for event in trace.get("events", []):
        if event.get("type") != "tool_call":
            continue
        result = _safe_json(event.get("result"))
        if isinstance(result, dict) and "error" in result:
            tool_errors += 1
        if event.get("name") == "search_catalog" and isinstance(result, list):
            if not result:
                empty_searches += 1
            result_ids = {str(item.get("id")) for item in result if isinstance(item, dict)}
            saw_solution = saw_solution or bool(result_ids & solutions)
            saw_policy_eligible = saw_policy_eligible or bool(result_ids & eligible)
    for calls in trace.get("api_calls", {}).values():
        for call in calls:
            cost = call.get("provider_reported_cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                provider_cost += float(cost)
    return {
        "task_id": task.id,
        "trial": int(row.get("trial", 0)),
        "complexity": task.complexity,
        "reveal": task.reveal_difficulty,
        "nvr": task.no_valid_recommendation,
        "policy_flags": list(task.policy_flags),
        "solution_count": len(solutions),
        "solution_bucket": _solution_bucket(len(solutions)),
        "success": float(row.get("primary_reward", 0)) >= 1.0 - 1e-9,
        "constraint_success": float(row.get("constraint_score", 0)) >= 1.0 - 1e-9,
        "policy_success": float(row.get("policy_score", 0)) >= 1.0 - 1e-9,
        "completed": stop_reason in {"recommended", "abstained"},
        "recommended": stop_reason == "recommended",
        "abstained": stop_reason == "abstained",
        "timeout": stop_reason == "timeout",
        "stop_reason": stop_reason,
        "false_nvr_recommendation": task.no_valid_recommendation and bool(recommendations),
        "false_satisfiable_abstention": (
            not task.no_valid_recommendation and stop_reason == "abstained"
        ),
        "tool_errors": tool_errors,
        "empty_searches": empty_searches,
        "saw_constraint_solution": saw_solution,
        "saw_policy_eligible_solution": saw_policy_eligible,
        "provider_reported_cost_usd": provider_cost,
        "runtime_seconds": float(
            row.get("efficiency", {}).get("trial_runtime_s", 0.0) or 0.0
        ),
    }


def _observation_metrics(rows: list[dict[str, Any]], ks: tuple[int, ...] = (1, 4)) -> dict[str, Any]:
    if not rows:
        return {"tasks": 0, "trials": 0}
    by_task: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(int(row["success"]))
    result: dict[str, Any] = {
        "tasks": len(by_task),
        "trials": len(rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "constraint_success_rate": sum(row["constraint_success"] for row in rows) / len(rows),
        "policy_success_rate": sum(row["policy_success"] for row in rows) / len(rows),
        "completion_rate": sum(row["completed"] for row in rows) / len(rows),
        "recommendation_rate": sum(row["recommended"] for row in rows) / len(rows),
        "abstention_rate": sum(row["abstained"] for row in rows) / len(rows),
        "timeout_rate": sum(row["timeout"] for row in rows) / len(rows),
        "false_nvr_recommendations": sum(row["false_nvr_recommendation"] for row in rows),
        "false_satisfiable_abstentions": sum(row["false_satisfiable_abstention"] for row in rows),
        "tool_errors": sum(row["tool_errors"] for row in rows),
        "empty_searches": sum(row["empty_searches"] for row in rows),
        "saw_policy_eligible_rate": sum(row["saw_policy_eligible_solution"] for row in rows) / len(rows),
        "provider_reported_cost_usd": sum(
            row.get("provider_reported_cost_usd", 0.0) for row in rows
        ),
        "runtime_seconds": _numeric_summary(
            row.get("runtime_seconds", 0.0) for row in rows
        ),
    }
    for k in ks:
        eligible = [values for values in by_task.values() if len(values) >= k]
        result[f"pass^{k}"] = (
            sum(pass_at_k(len(values), sum(values), k) for values in eligible) / len(eligible)
            if eligible else None
        )
        result[f"pass^{k}_tasks"] = len(eligible)
    return result


def _group_metrics(
    observations: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        values = row[key] if isinstance(row[key], list) else [row[key]]
        for value in values:
            groups[str(value)].append(row)
    return {
        value: _observation_metrics(rows)
        for value, rows in sorted(groups.items())
    }


def analyze_run(label: str, run_dir: Path, artifact: Artifact) -> dict[str, Any]:
    manifest = _load_json(run_dir / "run_manifest.json")
    if manifest.get("tasks_sha256") != hash_task_dir(artifact.root / "tasks"):
        raise ValueError(f"{label}: run task hash does not match artifact")
    if manifest.get("catalog_sha256") != hash_file(artifact.root / "catalog.json"):
        raise ValueError(f"{label}: run catalog hash does not match artifact")
    result_rows = _load_json(run_dir / "trial_results.json")
    observations: list[dict[str, Any]] = []
    for row in result_rows:
        task_id = row["task_id"]
        if task_id not in artifact.tasks:
            raise ValueError(f"{label}: unknown task {task_id}")
        observations.append(_trial_observation(
            row,
            artifact.tasks[task_id],
            artifact.answers[task_id],
            _trial_trace(run_dir, task_id, int(row.get("trial", 0))),
        ))

    by_task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_task_rows[row["task_id"]].append(row)
    task_summary = {
        task_id: _observation_metrics(rows)
        for task_id, rows in sorted(by_task_rows.items())
    }
    return {
        "label": label,
        "path": str(run_dir),
        "condition": {
            "model": manifest.get("model"),
            "reasoning_effort": manifest.get("reasoning_effort"),
            "retrieval_mode": manifest.get("retrieval_mode"),
            "simulator_mode": manifest.get("simulator_interaction_mode"),
            "catalog_size": manifest.get("catalog_size"),
            "trials_per_task": manifest.get("trials_per_task"),
        },
        "overall": _observation_metrics(observations),
        "by_complexity": _group_metrics(observations, "complexity"),
        "by_reveal": _group_metrics(observations, "reveal"),
        "by_nvr": _group_metrics(observations, "nvr"),
        "by_policy_flag": _group_metrics(observations, "policy_flags"),
        "by_solution_bucket": _group_metrics(observations, "solution_bucket"),
        "task_summary": task_summary,
    }


def _queue_row(
    priority: int,
    source: str,
    reason: str,
    family_id: str = "",
    *,
    related_family_id: str = "",
    task_id: str = "",
    complexity: str = "",
    reveal: str = "",
    detail: str = "",
) -> dict[str, Any]:
    return {
        "priority": priority,
        "source": source,
        "reason": reason,
        "family_id": family_id,
        "related_family_id": related_family_id,
        "task_id": task_id,
        "complexity": complexity,
        "reveal": reveal,
        "detail": detail,
    }


def build_review_queue(
    artifact: Artifact,
    deterministic: dict[str, Any],
    coverage: dict[str, Any],
    embeddings: dict[str, Any],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    family_by_id = {row["family_id"]: row for row in artifact.families}
    for issue in deterministic.get("hidden_leakage_issues", []):
        family = family_by_id[issue["family_id"]]
        queue.append(_queue_row(
            0,
            "deterministic",
            issue["reason"],
            issue["family_id"],
            complexity=family["complexity"],
            detail=issue["detail"],
        ))
    if deterministic["status"] == "fail" and not deterministic.get("hidden_leakage_issues"):
        queue.append(_queue_row(
            0,
            "deterministic",
            "artifact_validator_failed",
            detail="; ".join(deterministic.get("errors", [])[:5]),
        ))

    solution_pairs: set[tuple[str, str]] = set()
    for pair in coverage["solution_set_diversity"]["flagged_pairs"]:
        solution_pairs.add(tuple(sorted((
            pair["left_family_id"], pair["right_family_id"]
        ))))
        family = family_by_id[pair["left_family_id"]]
        queue.append(_queue_row(
            2 if pair["identical"] else 3,
            "solution_sets",
            "identical_solution_set" if pair["identical"] else "high_solution_overlap",
            pair["left_family_id"],
            related_family_id=pair["right_family_id"],
            complexity=family["complexity"],
            detail=(
                f"jaccard={pair['jaccard']:.4f}; "
                f"counts={pair['left_solution_count']}/{pair['right_solution_count']}"
            ),
        ))

    if embeddings.get("status") == "complete":
        seen_pairs: set[tuple[str, str, str]] = set()
        # Opening repetition is expected in the matched disclosure design:
        # mixed/hidden arms intentionally expose only one shared fact.  It is
        # reported, but only full-family semantic redundancy enters the queue.
        for section in ("family_semantics",):
            for pair in embeddings[section]["flagged_pairs"][:25]:
                left = pair["left"]["family_id"]
                right = pair["right"]["family_id"]
                key = (section, *sorted((left, right)))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if tuple(sorted((left, right))) in solution_pairs:
                    continue
                queue.append(_queue_row(
                    3,
                    "embedding",
                    f"near_duplicate_{section}",
                    left,
                    related_family_id=right,
                    complexity=family_by_id[left]["complexity"],
                    detail=(
                        f"cosine={pair['cosine_similarity']:.4f}; "
                        f"solution_jaccard={pair['solution_jaccard']}"
                    ),
                ))

    for run in runs:
        condition = run["condition"]
        by_family: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for task_id, metrics in run["task_summary"].items():
            by_family[artifact.task_to_family[task_id]][
                artifact.task_to_reveal[task_id]
            ] = metrics
            task = artifact.tasks[task_id]
            if metrics.get("tool_errors", 0):
                queue.append(_queue_row(
                    3,
                    f"run:{run['label']}",
                    "agent_tool_error",
                    artifact.task_to_family[task_id],
                    task_id=task_id,
                    complexity=task.complexity,
                    reveal=task.reveal_difficulty,
                    detail=f"{metrics['tool_errors']} tool errors",
                ))
            if (
                task.no_valid_recommendation
                and metrics.get("trials", 0) >= 4
                and metrics.get("false_nvr_recommendations", 0) >= 2
            ):
                queue.append(_queue_row(
                    3,
                    f"run:{run['label']}",
                    "agent_false_nvr_recommendation",
                    artifact.task_to_family[task_id],
                    task_id=task_id,
                    complexity=task.complexity,
                    reveal=task.reveal_difficulty,
                    detail=(
                        f"{metrics['false_nvr_recommendations']} false recommendations"
                    ),
                ))
        for family_id, reveal_metrics in by_family.items():
            family = family_by_id[family_id]
            volunteer = reveal_metrics.get("volunteer")
            hidden = reveal_metrics.get("hidden")
            if (
                condition.get("retrieval_mode") == "structured"
                and not family["no_valid_recommendation"]
                and volunteer
                and volunteer["trials"] >= 4
                and volunteer.get("constraint_success_rate") == 0
            ):
                queue.append(_queue_row(
                    2,
                    f"run:{run['label']}",
                    "structured_volunteer_0_constraint_success",
                    family_id,
                    task_id=family["task_ids"]["volunteer"],
                    complexity=family["complexity"],
                    reveal="volunteer",
                    detail=f"0/{volunteer['trials']} constraint-successful trials",
                ))
            total_trials = sum(row["trials"] for row in reveal_metrics.values())
            total_constraint_successes = sum(
                round(row.get("constraint_success_rate", 0) * row["trials"])
                for row in reveal_metrics.values()
            )
            if total_trials >= 12 and total_constraint_successes == 0:
                queue.append(_queue_row(
                    2,
                    f"run:{run['label']}",
                    "family_0_constraint_success",
                    family_id,
                    complexity=family["complexity"],
                    detail=(
                        f"0/{total_trials} constraint-successful trials across reveal variants"
                    ),
                ))
            if volunteer and hidden and volunteer["trials"] >= 4 and hidden["trials"] >= 4:
                reversal = (
                    hidden.get("constraint_success_rate", 0)
                    - volunteer.get("constraint_success_rate", 0)
                )
                if reversal >= 0.5:
                    queue.append(_queue_row(
                        3,
                        f"run:{run['label']}",
                        "hidden_outperforms_volunteer",
                        family_id,
                        complexity=family["complexity"],
                        detail=f"hidden-minus-volunteer constraint success={reversal:+.3f}",
                    ))

    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in queue:
        key = tuple(row[field] for field in (
            "source", "reason", "family_id", "related_family_id", "task_id"
        ))
        deduped.setdefault(key, row)
    return sorted(
        deduped.values(),
        key=lambda row: (
            row["priority"], row["source"], row["reason"],
            row["family_id"], row["related_family_id"], row["task_id"],
        ),
    )


def _format_rate(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def markdown_report(report: dict[str, Any]) -> str:
    deterministic = report["deterministic_validation"]
    coverage = report["coverage"]
    embeddings = report["embedding_diversity"]
    queue = report["review_queue"]
    lines = [
        "# books automated audit",
        "",
        f"Artifact gate: **{deterministic['status'].upper()}**. "
        f"Review-queue rows: **{len(queue)}**.",
        "",
        "> This report automates structural, coverage, semantic-diversity, and model-run triage. "
        "It does not replace human review of bibliographic truth or naturalness, and model "
        "failure is not evidence that a catalog-grounded task is impossible.",
        "",
        "## Deterministic validity",
        "",
        f"- Tasks checked: {deterministic['task_count']}",
        f"- Errors: {len(deterministic['errors'])}",
        f"- Warnings: {len(deterministic['warnings'])}",
        f"- Independent validator: {deterministic['independent_validator']['status']}",
        "",
        "## Coverage census",
        "",
        f"- Catalog: {coverage['catalog_count']:,} works",
        f"- Families/tasks: {coverage['family_count']} / {coverage['task_count']}",
        f"- Personas: {coverage['personas']['unique']} unique "
        f"(effective {coverage['personas']['effective_categories']:.1f})",
        f"- Constraint templates: {coverage['constraint_templates']['unique']} unique "
        f"(effective {coverage['constraint_templates']['effective_categories']:.1f})",
        f"- Constraint values: {coverage['constraint_values']['unique']} unique",
        f"- Highly overlapping solution-set pairs: "
        f"{coverage['solution_set_diversity']['pairs_above_threshold']} "
        f"(identical: {coverage['solution_set_diversity']['identical_solution_set_pairs']})",
        f"- Catalog genres/languages/countries: "
        f"{coverage['catalog']['genres']['unique']} / "
        f"{coverage['catalog']['languages']['unique']} / "
        f"{coverage['catalog']['countries']['unique']}",
        "",
    ]
    if embeddings.get("status") == "complete":
        family = embeddings["family_semantics"]
        openings = embeddings["user_openings"]
        lines.extend([
            "## Semantic diversity",
            "",
            f"Embedding model: `{embeddings['model']}` ({embeddings['dimension']} dimensions; "
            "mean-centered cosine space).",
            "",
            "| View | Compared pairs | Above threshold | Median nearest neighbor | Maximum |",
            "|---|---:|---:|---:|---:|",
            f"| Family semantics | {family['compared_pairs']} | "
            f"{family['pairs_above_threshold']} | "
            f"{_format_rate(family['nearest_neighbor_similarity']['median'])} | "
            f"{_format_rate(family['nearest_neighbor_similarity']['max'])} |",
            f"| User openings | {openings['compared_pairs']} | "
            f"{openings['pairs_above_threshold']} | "
            f"{_format_rate(openings['nearest_neighbor_similarity']['median'])} | "
            f"{_format_rate(openings['nearest_neighbor_similarity']['max'])} |",
            "",
            "Opening similarity is descriptive: mixed and hidden arms intentionally "
            "begin with one shared disclosed fact, so opening pairs are not added to "
            "the review queue.",
            "",
            "Highest-similarity family pairs:",
            "",
            "| Left | Right | Cosine | Solution Jaccard |",
            "|---|---|---:|---:|",
        ])
        for pair in family["top_pairs"][:10]:
            jaccard = pair["solution_jaccard"]
            lines.append(
                f"| {pair['left']['family_id']} | {pair['right']['family_id']} | "
                f"{pair['cosine_similarity']:.4f} | "
                f"{'—' if jaccard is None else f'{jaccard:.4f}'} |"
            )
        lines.append("")
    else:
        lines.extend(["## Semantic diversity", "", "Embedding audit skipped.", ""])

    lines.extend(["## Empirical run triage", ""])
    if not report["runs"]:
        lines.extend([
            "No run was supplied. Pass `--run LABEL=/path/to/run` after a locked "
            "K=4 sweep to add completion and outcome diagnostics.",
            "",
        ])
    else:
        lines.extend([
            "| Run | Model | Retrieval | Trials | Success | Completion | pass^4 |",
            "|---|---|---|---:|---:|---:|---:|",
        ])
        for run in report["runs"]:
            condition, overall = run["condition"], run["overall"]
            lines.append(
                f"| {run['label']} | {condition['model']} | "
                f"{condition['retrieval_mode']} | {overall['trials']} | "
                f"{overall['success_rate']:.3f} | {overall['completion_rate']:.3f} | "
                f"{_format_rate(overall.get('pass^4'))} |"
            )
        lines.append("")
        for run in report["runs"]:
            lines.extend([
                f"### {run['label']} strata",
                "",
                "| Stratum | Tasks | Trials | Success | Completion | pass^4 |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for group_name, groups in (
                ("complexity", run["by_complexity"]),
                ("reveal", run["by_reveal"]),
            ):
                for value, metric in groups.items():
                    lines.append(
                        f"| {group_name}={value} | {metric['tasks']} | "
                        f"{metric['trials']} | {metric['success_rate']:.3f} | "
                        f"{metric['completion_rate']:.3f} | "
                        f"{_format_rate(metric.get('pass^4'))} |"
                    )
            lines.append("")

    lines.extend([
        "## Ranked review queue",
        "",
        "| Priority | Source | Reason | Family | Related/task | Detail |",
        "|---:|---|---|---|---|---|",
    ])
    for row in queue[:50]:
        related = row["related_family_id"] or row["task_id"]
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(
            f"| {row['priority']} | {row['source']} | {row['reason']} | "
            f"{row['family_id']} | {related} | {detail} |"
        )
    if len(queue) > 50:
        lines.append(f"\n{len(queue) - 50} additional rows are in the CSV queue.")
    lines.append("")
    return "\n".join(lines)


def write_review_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "priority", "source", "reason", "family_id", "related_family_id",
        "task_id", "complexity", "reveal", "detail",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    artifact: Artifact,
    *,
    independent_validator: Path | None,
    embedding_model: str | None,
    near_duplicate_threshold: float,
    top_pairs: int,
    run_paths: list[tuple[str, Path]],
) -> dict[str, Any]:
    deterministic = deterministic_validation(
        artifact, independent_validator=independent_validator
    )
    coverage = coverage_report(artifact)
    embeddings = (
        embedding_diversity(
            artifact,
            model_name=embedding_model,
            threshold=near_duplicate_threshold,
            top_pairs=top_pairs,
        )
        if embedding_model
        else {"status": "skipped"}
    )
    runs = [analyze_run(label, path, artifact) for label, path in run_paths]
    queue = build_review_queue(
        artifact, deterministic, coverage, embeddings, runs
    )
    return {
        "audit_version": AUDIT_VERSION,
        "artifact": {
            "track_id": _load_json(artifact.root / "track.json")["id"],
            "catalog_sha256": hash_file(artifact.root / "catalog.json"),
            "tasks_sha256": hash_task_dir(artifact.root / "tasks"),
            "family_manifest_sha256": hash_file(artifact.root / "family_manifest.json"),
            "task_manifest_sha256": hash_file(artifact.root / "task_manifest.json"),
        },
        "interpretation": {
            "deterministic_failures_are_release_blocking": True,
            "embedding_flags_require_review_not_automatic_deletion": True,
            "model_failures_are_empirical_triage_not_impossibility_proofs": True,
            "human_bibliographic_and_naturalness_review_still_required": True,
        },
        "parameters": {
            "embedding_model": embedding_model,
            "near_duplicate_threshold": near_duplicate_threshold,
            "top_pairs": top_pairs,
        },
        "deterministic_validation": deterministic,
        "coverage": coverage,
        "embedding_diversity": embeddings,
        "runs": runs,
        "review_queue": queue,
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    default_artifact = root / "data" / "tracks" / "books-wikidata-v2"
    default_output = root / "analysis" / "books-wikidata-v2" / "automated_audit.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=default_artifact)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--review-queue", type=Path, default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    )
    parser.add_argument("--top-pairs", type=int, default=25)
    parser.add_argument("--run", action="append", type=_named_path, default=[])
    parser.add_argument("--skip-independent-validator", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.near_duplicate_threshold <= 1:
        parser.error("--near-duplicate-threshold must be between 0 and 1")
    if args.top_pairs < 1:
        parser.error("--top-pairs must be positive")

    artifact = load_artifact(args.artifact)
    independent = None if args.skip_independent_validator else (
        root / "scripts" / "validate_books_controlled.py"
    )
    try:
        report = build_report(
            artifact,
            independent_validator=independent,
            embedding_model=None if args.skip_embeddings else args.embedding_model,
            near_duplicate_threshold=args.near_duplicate_threshold,
            top_pairs=args.top_pairs,
            run_paths=args.run,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    queue_path = args.review_queue or args.output.with_name("automated_review_queue.csv")
    write_review_queue(queue_path, report["review_queue"])
    print(
        f"Automated audit {report['deterministic_validation']['status'].upper()}: "
        f"{report['coverage']['family_count']} families, "
        f"{len(report['review_queue'])} review-queue rows"
    )
    print(f"Wrote {args.output}, {args.output.with_suffix('.md')}, and {queue_path}")
    return 0 if report["deterministic_validation"]["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
