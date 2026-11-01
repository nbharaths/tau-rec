#!/usr/bin/env python3
"""Validate the controlled books artifact with an independent answer path."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from tau_rec.catalog.books_tasks import (
    IMPLIED_FACT_FIELDS,
    IMPLIED_FACT_MIN_LIFT,
    IMPLIED_FACT_MIN_PROBABILITY,
    IMPLIED_FACT_MIN_SUPPORT,
    MATCHED_DESIGN_VERSION,
    MATCHED_REVEAL_ASSIGNMENT_VERSION,
    RETRIEVER_INDEPENDENT_SAMPLING,
    REVEAL_GUARD_VERSION,
    REVEALS,
)
from tau_rec.catalog.taxonomy import BOOK_TAXONOMY_VERSION, contains_label
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import RevealTag, Task
from tau_rec.domain import BOOKS
from tau_rec.environment.tools import ToolKit
from tau_rec.leaderboard.hashing import hash_file

PUBLIC_PROVENANCE_FIELDS = {
    "source_name", "source_url", "source_version", "retrieved_at", "license_id",
    "license_url", "attribution", "transformation_script", "output_catalog_sha256",
    "redistribution_allowed",
}


def _matches(item, field: str, op: str, value) -> bool:
    """Independent, deliberately direct implementation of predicate truth."""
    actual = getattr(item, field, None)
    if actual is None:
        return False
    if op == "<=":
        return actual <= value
    if op == ">=":
        return actual >= value
    if op == "==":
        return actual == value
    if op == "!=":
        return actual != value
    if op == "contains":
        if field in {"languages", "countries"}:
            return contains_label(field, actual, str(value))
        return value in actual
    if op == "contains_any":
        if field in {"languages", "countries"}:
            return any(contains_label(field, actual, str(element)) for element in value)
        return any(element in actual for element in value)
    if op == "not_contains":
        if field in {"languages", "countries"}:
            return not contains_label(field, actual, str(value))
        return value not in actual
    if op == "in":
        return actual in value
    raise ValueError(f"unknown operator {op!r}")


def _implied_fact_leaks(task: Task, catalog: Catalog) -> list[str]:
    """Independently find revealed facts that imply a hidden fact."""
    leaks = []
    constraints = [tagged.constraint for tagged in task.constraints]
    for target_tagged in task.constraints:
        target = target_tagged.constraint
        if (
            target_tagged.reveal != RevealTag.HIDDEN
            or target.op != "contains"
            or target.field not in IMPLIED_FACT_FIELDS
        ):
            continue
        carrying = [item for item in catalog.items if getattr(item, target.field, None)]
        if not carrying:
            continue
        prior = sum(
            _matches(item, target.field, target.op, target.value) for item in carrying
        ) / len(carrying)
        for source_tagged, source in zip(task.constraints, constraints):
            if (
                source_tagged.reveal == RevealTag.HIDDEN
                or source.op != "contains"
                or source.field not in IMPLIED_FACT_FIELDS
            ):
                continue
            support = [
                item for item in carrying
                if _matches(item, source.field, source.op, source.value)
            ]
            if len(support) < IMPLIED_FACT_MIN_SUPPORT:
                continue
            probability = sum(
                _matches(item, target.field, target.op, target.value) for item in support
            ) / len(support)
            if (
                probability >= IMPLIED_FACT_MIN_PROBABILITY
                and probability >= IMPLIED_FACT_MIN_LIFT * prior
            ):
                leaks.append(
                    f"revealed {source.field}={source.value} implies hidden "
                    f"{target.field}={target.value} (P={probability:.2f}, n={len(support)})"
                )
    return leaks


def _solutions(task: Task, catalog: Catalog, *, omit: int | None = None) -> list[str]:
    return [
        item.id
        for item in catalog.items
        if all(
            _matches(
                item,
                tagged.constraint.field,
                tagged.constraint.op,
                tagged.constraint.value,
            )
            for index, tagged in enumerate(task.constraints)
            if index != omit
        )
    ]


def _constraint_signature(task: Task) -> str:
    return "|".join(
        f"{tagged.constraint.field}:{tagged.constraint.op}:{tagged.constraint.value}"
        for tagged in task.constraints
    )


def _structured_arguments(task: Task) -> dict:
    arguments: dict = {}
    list_fields = {"genres", "languages", "countries"}
    bound_fields = {
        ("publication_year", ">="): "min_publication_year",
        ("publication_year", "<="): "max_publication_year",
    }
    for tagged in task.constraints:
        constraint = tagged.constraint
        if constraint.field in list_fields and constraint.op == "contains":
            arguments.setdefault(constraint.field, []).append(str(constraint.value))
        elif (constraint.field, constraint.op) in bound_fields:
            arguments[bound_fields[(constraint.field, constraint.op)]] = constraint.value
        else:
            raise ValueError(
                f"{task.id}: no structured-tool translation for "
                f"{constraint.field} {constraint.op}"
            )
    if task.user_formats:
        arguments["formats"] = list(task.user_formats)
    return arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()

    catalog = Catalog.from_json(str(args.catalog), BOOKS)
    artifact = args.artifact
    manifest = json.loads((artifact / "task_manifest.json").read_text())
    family_manifest_path = artifact / "family_manifest.json"
    family_manifest = json.loads(family_manifest_path.read_text())
    answers = json.loads((artifact / "answers.json").read_text())
    tasks = {
        path.stem: Task.from_json(str(path))
        for path in sorted((artifact / "tasks").glob("*.json"))
    }
    errors: list[str] = []

    if manifest.get("sampling_mode") != RETRIEVER_INDEPENDENT_SAMPLING:
        errors.append("artifact does not declare retriever-independent sampling")
    if manifest.get("design_version") != MATCHED_DESIGN_VERSION:
        errors.append("artifact does not declare the matched reveal design")
    if manifest.get("reveal_assignment_version") != MATCHED_REVEAL_ASSIGNMENT_VERSION:
        errors.append("artifact does not declare the counterbalanced reveal assignment")
    if "posthoc outcome only" not in manifest.get("retrieval_role", ""):
        errors.append("retrieval role is not explicitly posthoc-only")
    if hash_file(args.catalog) != manifest.get("catalog_sha256"):
        errors.append("catalog SHA-256 drifted")
    if hashlib.sha256(family_manifest_path.read_bytes()).hexdigest() != manifest.get(
        "family_manifest_sha256"
    ):
        errors.append("family manifest SHA-256 drifted")
    if len(tasks) != manifest.get("n_tasks"):
        errors.append(f"loaded {len(tasks)} tasks, manifest says {manifest.get('n_tasks')}")
    if len(family_manifest.get("families", [])) != manifest.get("n_families"):
        errors.append("family count disagrees across manifests")
    if manifest.get("source_status") == "public-redistributable":
        provenance: dict = {}
        provenance_path = artifact / "source_provenance.json"
        if not provenance_path.is_file():
            errors.append("public artifact has no source_provenance.json")
        else:
            provenance = json.loads(provenance_path.read_text())
            if not isinstance(provenance, dict):
                errors.append("public source provenance is not a JSON object")
                provenance = {}
            missing = sorted(PUBLIC_PROVENANCE_FIELDS - set(provenance))
            if missing:
                errors.append(f"public source provenance is missing fields: {missing}")
            provenance_hash = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
            if provenance_hash != manifest.get("source_provenance_sha256"):
                errors.append("public source provenance SHA-256 drifted")
            if provenance.get("redistribution_allowed") is not True:
                errors.append("public source does not affirm redistribution permission")
            if provenance.get("output_catalog_sha256") != manifest.get("catalog_sha256"):
                errors.append("public provenance names a different output catalog")
        catalog_manifest_path = artifact / "catalog_manifest.json"
        selection_path = artifact / "source_selection.jsonl"
        if not catalog_manifest_path.is_file():
            errors.append("public artifact has no catalog_manifest.json")
        elif not selection_path.is_file():
            errors.append("public artifact has no source_selection.jsonl")
        else:
            catalog_manifest = json.loads(catalog_manifest_path.read_text())
            selection_sha = hash_file(selection_path)
            if catalog_manifest.get("catalog_sha256") != hash_file(args.catalog):
                errors.append("catalog manifest names a different catalog hash")
            if catalog_manifest.get("source_selection_sha256") != selection_sha:
                errors.append("catalog manifest source-selection hash drifted")
            if provenance and provenance.get("source_selection_sha256") != selection_sha:
                errors.append("public provenance source-selection hash drifted")
            if catalog_manifest.get("license", {}).get("id") != "CC0-1.0":
                errors.append("public Wikidata catalog does not declare CC0-1.0")
            query_records = catalog_manifest.get("source", {}).get(
                "query_response_records", []
            )
            if not query_records or any(
                len(record.get(key, "")) != 64
                for record in query_records
                for key in ("request_sha256", "response_sha256")
            ):
                errors.append("catalog manifest does not pin every query response")
            snapshot_path = artifact / "source_snapshot_asset.json"
            if not snapshot_path.is_file():
                errors.append("public artifact has no source-snapshot identity record")
            else:
                snapshot = json.loads(snapshot_path.read_text())
                track_config = json.loads((artifact / "track.json").read_text())
                track_id = track_config.get("id")
                if snapshot.get("schema_version") != (
                    "tau-rec-wikidata-source-snapshot-v1"
                ):
                    errors.append("source-snapshot identity has an unknown schema")
                if snapshot.get("track_id") != track_id:
                    errors.append("source-snapshot identity names a different track")
                if snapshot.get("catalog_sha256") != hash_file(args.catalog):
                    errors.append("source-snapshot identity names a different catalog")
                if snapshot.get("query_response_file_count") != len(query_records):
                    errors.append("source-snapshot response count disagrees with catalog")
                if snapshot.get("uncompressed_response_bytes") != sum(
                    record.get("bytes", 0) for record in query_records
                ):
                    errors.append("source-snapshot byte count disagrees with catalog")
                for key in ("archive_sha256", "snapshot_manifest_sha256"):
                    if not isinstance(snapshot.get(key), str) or len(snapshot[key]) != 64:
                        errors.append(f"source-snapshot identity has invalid {key}")
                if not isinstance(snapshot.get("archive_bytes"), int) or (
                    snapshot["archive_bytes"] <= 0
                ):
                    errors.append("source-snapshot identity has invalid archive size")
                if track_config.get("release_status") == "public" and (
                    snapshot.get("status") != "published" or not snapshot.get("url")
                ):
                    errors.append("public track has no published source-snapshot URL")
            selection_rows = [
                json.loads(line) for line in selection_path.read_text().splitlines()
                if line.strip()
            ]
            selection_ids = [f"wd_{row.get('wikidata_id', '')}" for row in selection_rows]
            if selection_ids != [item.id for item in catalog.items]:
                errors.append("source-selection IDs do not exactly match catalog order")
            synthetic_version = catalog_manifest.get(
                "synthetic_environment_state", {}
            ).get("version")
            if any(
                getattr(item, "formats_source", None) != synthetic_version
                for item in catalog.items
            ):
                errors.append("catalog format-state version disagrees with its manifest")
            if manifest.get("task_profile") != "wikidata-v2":
                errors.append("task manifest names an unknown task profile")
            if manifest.get("constraint_fields") != [
                "genres", "languages", "countries", "publication_year",
            ]:
                errors.append("Wikidata task profile uses an unexpected constraint field")
            if any(item.rating is not None or item.vote_count is not None for item in catalog.items):
                errors.append("Wikidata catalog fabricates rating or vote-count values")

    catalog_sizes = manifest.get("catalog_sizes", [])
    if catalog_sizes != sorted(set(catalog_sizes)) or not catalog_sizes:
        errors.append(f"catalog sizes are not a non-empty nested sequence: {catalog_sizes}")
    for size in catalog_sizes:
        if not 0 < size <= len(catalog):
            errors.append(f"invalid catalog size {size}")
            continue
        actual = hashlib.sha256(
            "\n".join(item.id for item in catalog.items[:size]).encode()
        ).hexdigest()
        if manifest.get("catalog_prefix_id_sha256", {}).get(str(size)) != actual:
            errors.append(f"catalog prefix {size:,} ID hash drifted")
    core_size = min(catalog_sizes, default=len(catalog))
    core_ids = {item.id for item in catalog.items[:core_size]}

    diversity = manifest.get("selection", {}).get("solution_set_diversity", {})
    diversity_limit = diversity.get("maximum_pairwise_jaccard_exclusive")
    if not isinstance(diversity_limit, (int, float)) or not 0 < diversity_limit <= 1:
        errors.append("task manifest has no valid solution-set diversity threshold")

    if manifest.get("taxonomy_version") != BOOK_TAXONOMY_VERSION:
        errors.append("Wikidata task manifest has an unknown taxonomy version")
    representatives = [
        tasks[row["task_ids"]["volunteer"]]
        for row in family_manifest.get("families", [])
    ]
    all_format_counts = Counter(
        value for task in tasks.values() for value in task.user_formats
    )
    if dict(sorted(all_format_counts.items())) != manifest.get(
        "policy_format_counts"
    ):
        errors.append("policy-format counts drifted from the task manifest")
    for complexity in ("simple", "medium", "complex"):
        format_counts = Counter(
            value
            for task in representatives if task.complexity == complexity
            for value in task.user_formats
        )
        if set(format_counts) != {"audiobook", "ebook", "hardcover", "paperback"}:
            errors.append(f"{complexity}: format-policy coverage is incomplete")
        elif len(set(format_counts.values())) != 1:
            errors.append(f"{complexity}: format-policy coverage is not balanced")
    for size in catalog_sizes:
        prefix = Catalog(items=catalog.items[:size])
        toolkit = ToolKit(
            prefix, None, {}, spec=BOOKS, retrieval_mode="structured",
            structured_profile="books-wikidata-core-v3",
        )
        for task in representatives:
            result = json.loads(toolkit.call(
                "search_catalog", _structured_arguments(task)
            ))
            if isinstance(result, dict) and "error" in result:
                errors.append(
                    f"{task.id}@{size}: structured tool returned {result['error']}"
                )
                continue
            if task.no_valid_recommendation:
                if result:
                    errors.append(
                        f"{task.id}@{size}: structured tool found a result for NVR"
                    )
                continue
            eligible = set(answers[task.id]["policy_eligible_solutions"])
            if not any(row.get("id") in eligible for row in result):
                errors.append(
                    f"{task.id}@{size}: structured top 20 has no policy-eligible answer"
                )

    seen_task_ids: set[str] = set()
    satisfiable_solution_sets: list[tuple[str, set[str]]] = []
    reveal_field_counts: Counter = Counter()
    for family in family_manifest.get("families", []):
        family_id = family.get("family_id", "<missing>")
        task_ids = family.get("task_ids", {})
        if set(task_ids) != set(REVEALS):
            errors.append(f"{family_id}: task map does not cover {REVEALS}")
            continue
        try:
            variants = [tasks[task_ids[reveal]] for reveal in REVEALS]
        except KeyError as exc:
            errors.append(f"{family_id}: missing task {exc.args[0]}")
            continue
        seen_task_ids.update(task.id for task in variants)

        constraint_rows = [
            [tagged.constraint.model_dump(mode="json") for tagged in task.constraints]
            for task in variants
        ]
        if any(row != constraint_rows[0] for row in constraint_rows[1:]):
            errors.append(f"{family_id}: constraints differ across reveal variants")
        if _constraint_signature(variants[0]) != family.get("constraint_signature"):
            errors.append(f"{family_id}: frozen constraint signature drifted")

        shared_states = []
        for reveal, task in zip(REVEALS, variants):
            if task.reveal_difficulty != reveal:
                errors.append(f"{family_id}: {task.id} has reveal label {task.reveal_difficulty}")
            for tagged in task.constraints:
                reveal_field_counts[(reveal, tagged.constraint.field, tagged.reveal.value)] += 1
            state = task.model_dump(mode="json")
            state.pop("id")
            state.pop("constraints")
            state.pop("reveal_difficulty")
            shared_states.append(state)
        if any(state != shared_states[0] for state in shared_states[1:]):
            errors.append(f"{family_id}: non-reveal state differs across variants")
        reveal_vectors = {
            reveal: [tagged.reveal for tagged in task.constraints]
            for reveal, task in zip(REVEALS, variants)
        }
        if any(tag != RevealTag.VOLUNTEER for tag in reveal_vectors["volunteer"]):
            errors.append(f"{family_id}: volunteer arm does not reveal every predicate")
        mixed_volunteer = {
            index for index, tag in enumerate(reveal_vectors["mixed"])
            if tag == RevealTag.VOLUNTEER
        }
        hidden_volunteer = {
            index for index, tag in enumerate(reveal_vectors["hidden"])
            if tag == RevealTag.VOLUNTEER
        }
        if len(mixed_volunteer) != 1 or any(
            tag not in {RevealTag.VOLUNTEER, RevealTag.ON_ASK}
            for tag in reveal_vectors["mixed"]
        ):
            errors.append(f"{family_id}: invalid mixed reveal assignment")
        if (
            hidden_volunteer != mixed_volunteer
            or RevealTag.HIDDEN not in reveal_vectors["hidden"]
        ):
            errors.append(f"{family_id}: invalid hidden reveal assignment")

        independently_computed = _solutions(variants[0], catalog)
        if independently_computed:
            satisfiable_solution_sets.append(
                (family_id, set(independently_computed))
            )
        expected_answer_hash = hashlib.sha256(
            "\n".join(independently_computed).encode()
        ).hexdigest()
        if expected_answer_hash != family.get("solution_ids_sha256"):
            errors.append(f"{family_id}: independent answer hash disagrees")
        if not family.get("no_valid_recommendation") and not (
            set(independently_computed) & core_ids
        ):
            errors.append(f"{family_id}: no answer exists in the smallest catalog prefix")
        expansion = manifest.get("minimum_predicate_expansion")
        if not family.get("no_valid_recommendation") and expansion:
            required_growth = max(
                int(expansion.get("absolute_items", 1)),
                math.ceil(
                    len(independently_computed) * float(expansion.get("relative", 0))
                ),
            )
            leave_one_counts = [
                len(_solutions(variants[0], catalog, omit=index))
                for index in range(len(variants[0].constraints))
            ]
            if any(
                count < len(independently_computed) + required_growth
                for count in leave_one_counts
            ):
                errors.append(
                    f"{family_id}: a satisfiable predicate misses the declared "
                    f"leave-one expansion of {required_growth} items"
                )

        for task in variants:
            task_path = artifact / "tasks" / f"{task.id}.json"
            actual_hash = hashlib.sha256(task_path.read_bytes()).hexdigest()
            if manifest.get("task_sha256", {}).get(task.id) != actual_hash:
                errors.append(f"{task.id}: task SHA-256 drifted")
            answer = answers.get(task.id)
            if answer is None:
                errors.append(f"{task.id}: answer is missing")
                continue
            if answer.get("family_id") != family_id:
                errors.append(f"{task.id}: answer points at the wrong family")
            if answer.get("constraint_solutions") != independently_computed:
                errors.append(f"{task.id}: independently recomputed answers disagree")
            eligible = set(answer.get("policy_eligible_solutions", []))
            if not eligible <= set(independently_computed):
                errors.append(f"{task.id}: policy answer contains a non-solution")
            if not task.no_valid_recommendation and not (eligible & core_ids):
                errors.append(
                    f"{task.id}: no policy-eligible answer exists in the smallest prefix"
                )
            diagnostic = answer.get("posthoc_retrieval_diagnostic")
            if diagnostic and diagnostic.get("used_for_task_selection") is not False:
                errors.append(f"{task.id}: retrieval diagnostic is not marked posthoc")

        if family.get("no_valid_recommendation"):
            leave_one = [
                len(_solutions(variants[0], catalog, omit=index))
                for index in range(len(variants[0].constraints))
            ]
            if independently_computed or min(leave_one, default=0) < 3:
                errors.append(
                    f"{family_id}: NVR task is not minimal-unsat with >=3 leave-one matches"
                )
            for task in variants:
                if answers[task.id].get("minimal_unsat_leave_one_counts") != leave_one:
                    errors.append(f"{task.id}: leave-one answer counts drifted")

    if seen_task_ids != set(tasks):
        missing = sorted(set(tasks) - seen_task_ids)
        extra = sorted(seen_task_ids - set(tasks))
        errors.append(f"family coverage differs from task files; missing={missing}, extra={extra}")

    observed_maximum = 0.0
    if isinstance(diversity_limit, (int, float)):
        for index, (left_id, left) in enumerate(satisfiable_solution_sets):
            for right_id, right in satisfiable_solution_sets[index + 1:]:
                jaccard = len(left & right) / len(left | right)
                observed_maximum = max(observed_maximum, jaccard)
                if jaccard >= diversity_limit:
                    errors.append(
                        f"{left_id}/{right_id}: solution-set Jaccard {jaccard:.4f} "
                        f"is not below {diversity_limit}"
                    )
        recorded_maximum = diversity.get("observed_maximum_pairwise_jaccard")
        if not isinstance(recorded_maximum, (int, float)) or not math.isclose(
            observed_maximum, recorded_maximum, rel_tol=0, abs_tol=1e-12
        ):
            errors.append("recorded maximum solution-set Jaccard drifted")

    if manifest.get("reveal_guard_version") != REVEAL_GUARD_VERSION:
        errors.append("artifact does not declare the implied-fact reveal guard")
    for family in family_manifest.get("families", []):
        hidden_task = tasks.get(family.get("task_ids", {}).get("hidden", ""))
        if hidden_task is not None:
            errors.extend(
                f"{family.get('family_id')}: {leak}"
                for leak in _implied_fact_leaks(hidden_task, catalog)
            )

    for field in manifest.get("constraint_fields", []):
        mixed_roles = {
            role for reveal, current_field, role in reveal_field_counts
            if reveal == "mixed" and current_field == field
            and reveal_field_counts[(reveal, current_field, role)] > 0
        }
        hidden_roles = {
            role for reveal, current_field, role in reveal_field_counts
            if reveal == "hidden" and current_field == field
            and reveal_field_counts[(reveal, current_field, role)] > 0
        }
        if mixed_roles != {"volunteer", "on_ask"}:
            errors.append(f"{field}: mixed assignments are not counterbalanced: {mixed_roles}")
        if not {"volunteer", "hidden"} <= hidden_roles:
            errors.append(f"{field}: hidden assignments are not counterbalanced: {hidden_roles}")
    if manifest.get("task_profile") == "wikidata-v2":
        from tau_rec.catalog.books_v2_checks import v2_rule_errors

        errors.extend(v2_rule_errors(
            catalog.items, tasks, family_manifest.get("families", []), core_size=catalog_sizes[0],
        ))
    live_reveal_counts = {
        f"{reveal}/{field}/{tag}": count
        for (reveal, field, tag), count in sorted(reveal_field_counts.items())
    }
    if live_reveal_counts != manifest.get("reveal_field_counts"):
        errors.append("recorded reveal-by-field counts drifted")

    if errors:
        print(f"controlled books artifact FAILED ({len(errors)} errors)")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(
        f"controlled books artifact PASS: {len(catalog):,} books, "
        f"{manifest['n_families']} families, {len(tasks)} matched tasks, "
        f"catalog sizes={catalog_sizes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
