#!/usr/bin/env python3
"""Freeze the exploratory v4 addendum: an agent prompt variant, and GPT-6.1 Sol with LLM users.

Built from the frozen v4 plan: the same artifact, models, runtime, harness
profile and stall rule, on the 18-family reliability block at 20k with the
structured interface. Each addendum cell is compared with the matching v4
reliability cell, which ran earlier, so the contrasts are exploratory and not
time-matched. A smoke stage on one family runs first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

VARIANT = "ask-requirements-v1"
GPT, DEEPSEEK = "gpt61_sol_high", "deepseek_v41_flash_high"
USERS = ["sim_glm53_flash", "sim_qwen38_27b"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(name: str, role: str, agents: list[str], task_ids: list[str], trials: int, *,
           users: list[str] | None = None, variant: str | None = None) -> dict:
    stage = {
        "name": name,
        "role": role,
        "agent_labels": agents,
        "simulator_mode": "llm" if users else "symbolic",
        "simulator_labels": users or [],
        "retrieval_modes": ["structured"],
        "catalog_sizes": [20000],
        "task_ids": task_ids,
        "tasks_filter": ",".join(task_ids),
        "task_count": len(task_ids),
        "trials_per_task": trials,
        "planned_trials": len(agents) * max(1, len(users or [])) * len(task_ids) * trials,
    }
    if variant:
        stage["agent_prompt_variant"] = variant
    return stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=Path("analysis/books-wikidata-v2/experiment_plan_v4.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis/books-wikidata-v2/experiment_plan_v4_addendum.json"))
    parser.add_argument("--ceiling", type=float, default=20.0)
    args = parser.parse_args()
    base = json.loads(args.base.read_text())
    reliability = next(stage for stage in base["stages"] if stage["name"] == "reliability")
    tasks = reliability["task_ids"]
    smoke = tasks[:3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/tau_rec", "data/tracks/books-wikidata-v2"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    stages = [
        _stage("smoke_prompt_variant", "wiring check for the prompt variant; excluded from estimates",
               [GPT, DEEPSEEK], smoke, 1, variant=VARIANT),
        _stage("smoke_gpt_llm_users", "wiring check for GPT-6.1 Sol with LLM users; excluded from estimates",
               [GPT], smoke, 1, users=USERS),
        _stage("prompt_variant", "reliability block with the ask-requirements prompt variant",
               [GPT, DEEPSEEK], tasks, reliability["trials_per_task"], variant=VARIANT),
        _stage("gpt_llm_users", "GPT-6.1 Sol with the two LLM users on the reliability block",
               [GPT], tasks, 4, users=USERS),
    ]
    plan = {
        "version": "books-wikidata-v4-addendum-v1",
        "status": "frozen",
        "base_plan_sha256": _sha256(args.base),
        "artifact": base["artifact"],
        "models": base["models"],
        "api_parameters": base["api_parameters"],
        "fixed_runtime": base["fixed_runtime"],
        "harness": {**base["harness"], "commit_at_freeze": {"sha": head, "dirty": bool(dirty)}},
        "stall_retry_rule": base["stall_retry_rule"],
        "budgets_usd": {"v4_hard_ceiling": args.ceiling},
        "launch_rules": {"order": "the two smoke stages first, then prompt_variant and gpt_llm_users together"},
        "stages": stages,
        "trial_accounting": {
            "smoke_trials": sum(s["planned_trials"] for s in stages if s["name"].startswith("smoke")),
            "addendum_trials": sum(s["planned_trials"] for s in stages if not s["name"].startswith("smoke")),
        },
        "analysis": {
            "status": "exploratory; specified before the addendum ran; unadjusted",
            "prompt_variant": (
                "per agent, variant minus the v4 reliability cell (same agent, structured, 20k, K=8): pass^1 overall "
                "and on mixed tasks, and the share of mixed trials with every on_ask fact disclosed; paired family "
                "bootstrap, B = 10,000, seed 20261014"
            ),
            "gpt_llm_users": (
                "per LLM user, GPT-6.1 Sol with that user minus GPT-6.1 Sol with the symbolic user (v4 reliability "
                "cell): pass^1 overall and on mixed tasks; compared descriptively with the DeepSeek contrasts"
            ),
            "comparison_cells": "from the v4 matrix, which ran earlier; not time-matched",
        },
    }
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: {plan['trial_accounting']} trials, frozen at {head[:12]} (dirty={bool(dirty)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
