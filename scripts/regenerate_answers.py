#!/usr/bin/env python3
"""Regenerate data/answers.json from the current catalog and task constraints."""
import json
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.catalog.validator import CatalogValidator

REPO = Path(__file__).parent.parent
CATALOG_PATH = REPO / "data" / "catalog.json"
TASKS_GLOB = str(REPO / "data" / "tasks" / "*.json")
ANSWERS_PATH = REPO / "data" / "answers.json"

catalog = Catalog.from_json(str(CATALOG_PATH))


def with_titles(ids):
    return [{"id": mid, "title": catalog.get(mid).title} for mid in ids]


tasks = [Task.from_json(p) for p in sorted(glob.glob(TASKS_GLOB))]
validator = CatalogValidator(catalog)

answers = {}
for task in tasks:
    result = validator.validate_task(task)
    solution_ids = result.solution_ids

    user_services = set(task.user_services)
    reachable = [
        mid for mid in solution_ids
        if set(catalog.get(mid).streaming_services) & user_services
    ] if user_services else []

    answers[task.id] = {
        "no_valid_recommendation": task.no_valid_recommendation,
        "user_services": task.user_services,
        "constraint_solutions": solution_ids,
        "constraint_solutions_with_titles": with_titles(solution_ids),
        "reachable_solutions": reachable,
        "reachable_solutions_with_titles": with_titles(reachable),
    }

ANSWERS_PATH.write_text(json.dumps(answers, indent=2, sort_keys=True) + "\n")
print(f"Wrote {len(answers)} entries to {ANSWERS_PATH}")
