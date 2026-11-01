#!/usr/bin/env python3
"""A deterministic no-LLM floor: recommend without asking anything.

For each task, take only the facts the opening turn states, filter the 20k
catalog on them, drop read works (and, when the opening names formats, works
lacking an accepted one), and recommend the most popular remaining work in the
structured tool's popularity order; abstain when none remains. Scoring follows
the benchmark: a satisfiable task succeeds when the item meets every predicate
and is unread (and is in an accepted format if formats were named); an NVR task
succeeds only by abstaining. It bounds what committing without elicitation earns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import RevealTag, Task
from tau_rec.domain import BOOKS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = Catalog.from_json(str(args.artifact / "catalog.json"), BOOKS)

    def popularity(item) -> tuple:
        votes = item.vote_count if item.vote_count is not None else item.sitelink_count
        rating = item.rating if item.rating is not None else float("-inf")
        return (-votes, -rating, item.id)

    ordered = sorted(catalog.items, key=popularity)
    answers = json.loads((args.artifact / "answers.json").read_text())
    panel = set(json.loads(args.panels.read_text())["panels"]["primary"]["task_ids"])
    rows = []
    for path in sorted((args.artifact / "tasks").glob("*.json")):
        task = Task.from_json(str(path))
        key = answers[task.id]
        opening = [tagged.constraint for tagged in task.constraints if tagged.reveal == RevealTag.VOLUNTEER]
        formats_named = task.reveal_difficulty == "volunteer" and bool(key["user_formats"])
        read = set(key["read_ids"])
        choice = next((
            item for item in ordered
            if item.id not in read and all(constraint.evaluate(item) for constraint in opening)
            and (not formats_named or set(item.formats or []) & set(key["user_formats"]))
        ), None)
        if key["no_valid_recommendation"]:
            success = choice is None
        else:
            success = choice is not None and choice.id in set(key["constraint_solutions"]) and (
                not formats_named or choice.id in set(key["policy_eligible_solutions"]))
        rows.append({"task_id": task.id, "reveal": task.reveal_difficulty, "panel": task.id in panel,
                     "nvr": key["no_valid_recommendation"], "success": success})
    summary = {}
    for reveal in ("volunteer", "mixed", "hidden"):
        subset = [row for row in rows if row["panel"] and row["reveal"] == reveal]
        summary[reveal] = sum(row["success"] for row in subset) / len(subset)
    on_panel = [row for row in rows if row["panel"]]
    summary["all"] = sum(row["success"] for row in on_panel) / len(on_panel)
    args.output.write_text(json.dumps({"summary_primary_panel": summary, "rows": rows}, indent=2) + "\n")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
