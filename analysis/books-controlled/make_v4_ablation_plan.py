#!/usr/bin/env python3
"""Freeze the exploratory v4 field-list ablation of the interface contrast.

The structured arm's tool line lists the fields it filters on; the keyword
arm's does not. This run gives the keyword arm the same field list
(prompt variant ``keyword-field-list-v1``, tool unchanged) for DeepSeek and
Qwen on the 18-family reliability block at 20k, so the v4 interface contrast
can be split into the field list and retrieval. Built from the frozen v4 plan:
the same artifact, models, runtime, harness profile and stall rule. Each cell
is compared with the matching v4 reliability cells, which ran earlier, so the
contrasts are exploratory and not time-matched. A smoke stage on one family
runs first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

VARIANT = "keyword-field-list-v1"
AGENTS = ["deepseek_v41_flash_high", "qwen38_27b_high"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(name: str, role: str, task_ids: list[str], trials: int) -> dict:
    return {
        "name": name,
        "role": role,
        "agent_labels": AGENTS,
        "agent_prompt_variant": VARIANT,
        "simulator_mode": "symbolic",
        "simulator_labels": [],
        "retrieval_modes": ["keyword"],
        "catalog_sizes": [20000],
        "task_ids": task_ids,
        "tasks_filter": ",".join(task_ids),
        "task_count": len(task_ids),
        "trials_per_task": trials,
        "planned_trials": len(AGENTS) * len(task_ids) * trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=Path("analysis/books-wikidata-v2/experiment_plan_v4.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis/books-wikidata-v2/experiment_plan_v4_ablation.json"))
    parser.add_argument("--ceiling", type=float, default=20.0)
    args = parser.parse_args()
    base = json.loads(args.base.read_text())
    reliability = next(stage for stage in base["stages"] if stage["name"] == "reliability")
    tasks = reliability["task_ids"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/tau_rec", "data/tracks/books-wikidata-v2"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    stages = [
        _stage("smoke_keyword_field_list", "wiring check for the keyword field-list variant; excluded from estimates",
               tasks[:3], 1),
        _stage("keyword_field_list", "reliability block, keyword arm with the structured arm's field list",
               tasks, reliability["trials_per_task"]),
    ]
    plan = {
        "version": "books-wikidata-v4-ablation-v1",
        "status": "frozen",
        "base_plan_sha256": _sha256(args.base),
        "artifact": base["artifact"],
        "models": base["models"],
        "api_parameters": base["api_parameters"],
        "fixed_runtime": base["fixed_runtime"],
        "harness": {**base["harness"], "commit_at_freeze": {"sha": head, "dirty": bool(dirty)}},
        "stall_retry_rule": base["stall_retry_rule"],
        "budgets_usd": {"v4_hard_ceiling": args.ceiling},
        "launch_rules": {"order": "the smoke stage first, then keyword_field_list"},
        "stages": stages,
        "trial_accounting": {
            "smoke_trials": stages[0]["planned_trials"],
            "ablation_trials": stages[1]["planned_trials"],
        },
        "analysis": {
            "status": "exploratory; specified before the ablation ran; unadjusted",
            "field_list": (
                "per agent, keyword with the field list minus the v4 reliability keyword cell (same agent, 20k, "
                "K=8): pass^1 overall, on volunteer, mixed and hidden tasks, and the share of mixed trials with "
                "every on_ask fact disclosed"
            ),
            "retrieval_with_field_list": (
                "per agent, the v4 reliability structured cell minus keyword with the field list: the same "
                "measures, with the field list in both prompts"
            ),
            "inference": "paired family bootstrap, B = 10,000, seed 20261014",
            "comparison_cells": "from the v4 matrix, which ran earlier; not time-matched",
        },
    }
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: {plan['trial_accounting']} trials, frozen at {head[:12]} (dirty={bool(dirty)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
