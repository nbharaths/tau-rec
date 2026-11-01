#!/usr/bin/env python3
"""Bound how much mixed-condition success rests on recogniser misreads.

A disclosed fact is a false positive when the blind label says the agent's
turn did not ask about its field and no generic request covers it (a generic
request advances one fact by design). A mixed success with any false-positive
disclosure may not have been earned; counting all of them as failures gives a
lower bound on mixed success. Rates are per trial within each agent and
interface on the primary panel, and model-averaged over agents.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

STAGES = ("primary_remainder", "reliability")


def false_positive(entry: dict, label: dict) -> bool:
    """A disclosed fact whose field the turn did not ask about and no generic request covers.

    A generic request ("tell me more about what you're in the mood for")
    advances one fact by design, also when the same turn names fields, so it
    covers one unasked fact (``disclosed_fields`` lists one field per fact).
    """
    asked = set(label.get("fields", []))
    unasked = [f for f in entry["disclosed_fields"] if f not in asked]
    return len(unasked) > (1 if label.get("generic", False) else 0)


def score(key: dict, labels: dict, mixed: dict[tuple[str, str], dict[str, bool]]) -> dict:
    flagged = defaultdict(set)
    for uid, entry in key.items():
        if false_positive(entry, labels[uid]):
            flagged[(entry["agent"], entry["interface"])].add(entry["source"].split("#")[0])
    by_cell = {}
    for (agent, interface), trials in sorted(mixed.items()):
        wins = {t for t, ok in trials.items() if ok}
        fp_wins = wins & flagged[(agent, interface)]
        n = len(trials)
        by_cell[f"{agent}/{interface}"] = {
            "mixed_trials": n, "mixed_successes": len(wins), "successes_with_fp_disclosure": len(fp_wins),
            "share_fp": round(len(fp_wins) / len(wins), 4) if wins else None,
            "pass1": round(len(wins) / n, 4) if n else None,
            "pass1_lower_bound": round((len(wins) - len(fp_wins)) / n, 4) if n else None,
        }
    agents = sorted({k.split("/")[0] for k in by_cell})
    averaged = {}
    for agent in agents:
        rows = [v for k, v in by_cell.items() if k.startswith(agent + "/")]
        averaged[agent] = {
            "share_fp": round(sum(r["successes_with_fp_disclosure"] for r in rows)
                              / max(1, sum(r["mixed_successes"] for r in rows)), 4),
            "pass1_drop": round(sum(r["pass1"] - r["pass1_lower_bound"] for r in rows) / len(rows), 4),
        }
    rows = list(by_cell.values())
    return {
        "by_cell": by_cell,
        "by_agent": averaged,
        "model_averaged": {
            "share_fp": round(sum(r["successes_with_fp_disclosure"] for r in rows)
                              / max(1, sum(r["mixed_successes"] for r in rows)), 4),
            "pass1_drop": round(sum(r["pass1"] - r["pass1_lower_bound"] for r in rows) / max(1, len(rows)), 4),
        },
        "false_positive_disclosures": sum(false_positive(e, labels[u]) for u, e in key.items()),
        "disclosures": len(key),
    }


def mixed_outcomes(root: Path, panel_task_ids: set[str]) -> dict[tuple[str, str], dict[str, bool]]:
    out = defaultdict(dict)
    for stage in STAGES:
        for path in sorted((root / stage).glob("*/*/20000/*/traces/*.json")):
            trace = json.loads(path.read_text())
            if trace["task_id"] in panel_task_ids and trace["task_id"].endswith("_m"):
                out[(path.parts[-6], path.parts[-5])][str(path.relative_to(root))] = bool(
                    trace["evaluation"].get("primary_reward"))
    return dict(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("out/books-wikidata-v4"))
    parser.add_argument("--plan", type=Path, default=Path("analysis/books-wikidata-v2/experiment_plan_v4.json"))
    parser.add_argument("--key", type=Path,
                        default=Path("analysis/books-wikidata-v2/audit_work/disclosure_misreads/key.json"))
    parser.add_argument("--labels", type=Path, default=Path("analysis/books-wikidata-v2/disclosure_misread_labels.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis/books-wikidata-v2/disclosure_misread_bound.json"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    tasks = {t for s in plan["stages"] if s["name"] in STAGES for t in s["task_ids"]}
    key = json.loads(args.key.read_text())
    labels = json.loads(args.labels.read_text())["labels"]
    missing = sorted(set(key) - set(labels), key=int)
    if missing:
        raise SystemExit(f"{len(missing)} disclosure turns have no label (first: {missing[:5]})")
    result = score(key, labels, mixed_outcomes(args.root, tasks))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["model_averaged"]), f"{result['false_positive_disclosures']}/{result['disclosures']} FP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
