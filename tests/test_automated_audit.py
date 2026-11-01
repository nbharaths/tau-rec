from types import SimpleNamespace

import numpy as np

from tau_rec.audit_books import (
    _embedding_pair_report,
    _normalise_embeddings,
    _observation_metrics,
    build_review_queue,
    distribution_summary,
)


def test_distribution_summary_reports_effective_coverage():
    summary = distribution_summary(["a", "a", "b", "c"])
    assert summary["total"] == 4
    assert summary["unique"] == 3
    assert summary["counts"] == {"a": 2, "b": 1, "c": 1}
    assert 2 < summary["effective_categories"] < 3
    assert 0 < summary["normalized_entropy"] <= 1


def test_embedding_pairs_exclude_matched_variants_from_same_family():
    matrix = _normalise_embeddings(np.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.99, 0.10],
    ]))
    records = [
        {"id": "task_a_v", "family_id": "family_a", "view": "volunteer"},
        {"id": "task_a_m", "family_id": "family_a", "view": "mixed"},
        {"id": "task_b_v", "family_id": "family_b", "view": "volunteer"},
    ]
    report = _embedding_pair_report(
        matrix,
        records,
        threshold=0.9,
        top_pairs=10,
        family_solutions={"family_a": {"x", "y"}, "family_b": {"y", "z"}},
        same_view_only=False,
    )
    assert report["compared_pairs"] == 2
    assert report["pairs_above_threshold"] == 2
    assert all(
        pair["left"]["family_id"] != pair["right"]["family_id"]
        for pair in report["top_pairs"]
    )
    assert report["top_pairs"][0]["solution_jaccard"] == 1 / 3


def test_observation_metrics_compute_completion_and_unbiased_pass4():
    rows = []
    for task_id, successes in (("task_a", 4), ("task_b", 3)):
        for trial in range(4):
            success = trial < successes
            rows.append({
                "task_id": task_id,
                "success": success,
                "constraint_success": success,
                "policy_success": True,
                "completed": trial != 3 or task_id == "task_a",
                "recommended": True,
                "abstained": False,
                "timeout": trial == 3 and task_id == "task_b",
                "false_nvr_recommendation": False,
                "false_satisfiable_abstention": False,
                "tool_errors": 0,
                "empty_searches": 0,
                "saw_policy_eligible_solution": True,
            })
    metrics = _observation_metrics(rows)
    assert metrics["success_rate"] == 7 / 8
    assert metrics["completion_rate"] == 7 / 8
    assert metrics["pass^4"] == 0.5
    assert metrics["pass^4_tasks"] == 2


def test_review_queue_treats_model_zeroes_as_triage_not_artifact_failure():
    family = {
        "family_id": "family_a",
        "complexity": "simple",
        "no_valid_recommendation": False,
        "task_ids": {
            "volunteer": "task_v",
            "mixed": "task_m",
            "hidden": "task_h",
        },
    }
    artifact = SimpleNamespace(
        families=[family],
        tasks={
            task_id: SimpleNamespace(
                no_valid_recommendation=False,
                complexity="simple",
                reveal_difficulty=reveal,
            )
            for reveal, task_id in family["task_ids"].items()
        },
        task_to_family={task_id: "family_a" for task_id in family["task_ids"].values()},
        task_to_reveal={task_id: reveal for reveal, task_id in family["task_ids"].items()},
    )
    zero = {
        "tasks": 1,
        "trials": 4,
        "success_rate": 0.0,
        "constraint_success_rate": 0.0,
        "tool_errors": 0,
        "false_nvr_recommendations": 0,
    }
    run = {
        "label": "ds_structured",
        "condition": {"retrieval_mode": "structured"},
        "task_summary": {task_id: dict(zero) for task_id in artifact.tasks},
    }
    queue = build_review_queue(
        artifact,
        {"status": "pass", "hidden_leakage_issues": []},
        {"solution_set_diversity": {"flagged_pairs": []}},
        {"status": "skipped"},
        [run],
    )
    reasons = {row["reason"] for row in queue}
    assert "structured_volunteer_0_constraint_success" in reasons
    assert "family_0_constraint_success" in reasons
    assert all(row["source"] == "run:ds_structured" for row in queue)
