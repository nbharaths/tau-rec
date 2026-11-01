#!/usr/bin/env python3
"""Write the exploratory follow-up plans that run after the v4 matrix.

Each is built from the frozen v4 plan: the same artifact, runtime, harness
profile and stall rule, on the 18-family reliability block at 20k with the
structured interface and the rule-based user. Each is compared with matching v4
reliability cells, which ran earlier, so every contrast is exploratory and not
time-matched.

- Prompt variant: the ask-requirements prompt for GLM 5.3 Flash and Qwen 3.8
  27B at K=8, matching the addendum's GPT-6.1 Sol and DeepSeek cells. $20.
- Verify variant: the ask-requirements prompt plus an instruction to propose
  the book with its candidate marker and wait for the verdict before calling
  recommend, for all four v4 agents at K=8. $25.
- Protocol variant: the ask-and-verify prompt plus what the rule-based user's
  replies mean (a declined field holds a requirement; a rejected book fails at
  least one), for all four v4 agents at K=8 on the verify plan's tasks. $25.

Separate plans give separate ledgers: the launcher splits one plan's ceiling
equally across its running cells. Without --freeze the plans are drafts;
--only writes (and freezes) a subset, leaving the other plan files untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

VARIANT = "ask-requirements-v1"
VERIFY_VARIANT = "ask-and-verify-v1"
PROTOCOL_VARIANT = "ask-verify-protocol-v1"
GLM, QWEN = "glm53_flash_high", "qwen38_27b_high"
V4_AGENTS = ["deepseek_v41_flash_high", GLM, QWEN, "gpt61_sol_high"]
FROZEN_INPUTS = ("src/tau_rec", "data/tracks/books-wikidata-v2")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dirty_inputs() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "--", *FROZEN_INPUTS],
                         capture_output=True, text=True, check=True, cwd=ROOT).stdout
    return [line for line in out.splitlines() if line.strip()]


def _stage(name: str, role: str, agents: list[str], task_ids: list[str], trials: int,
           variant: str | None = None) -> dict:
    stage = {
        "name": name, "role": role, "agent_labels": agents,
        "simulator_mode": "symbolic", "simulator_labels": [],
        "retrieval_modes": ["structured"], "catalog_sizes": [20000],
        "task_ids": task_ids, "tasks_filter": ",".join(task_ids), "task_count": len(task_ids),
        "trials_per_task": trials, "planned_trials": len(agents) * len(task_ids) * trials,
    }
    if variant:
        stage["agent_prompt_variant"] = variant
    return stage


def _plan(base: dict, base_path: Path, config: dict, config_path: Path, *, version: str, ceiling: float,
          stages: list[dict], launch: str, analysis: dict, freeze: bool, head: str) -> dict:
    harness = dict(base["harness"])
    harness["commit_at_freeze"] = {"sha": head, "dirty": False} if freeze else None
    return {
        "version": version,
        "status": "frozen" if freeze else "draft",
        "base_plan_sha256": _sha256(base_path),
        "artifact": {**base["artifact"], "model_config_sha256": _sha256(config_path)},
        "models": {**base["models"], "agents": config["agent_models"]},
        "api_parameters": base["api_parameters"],
        "fixed_runtime": base["fixed_runtime"],
        "harness": harness,
        "stall_retry_rule": base["stall_retry_rule"],
        "budgets_usd": {"v4_hard_ceiling": ceiling},
        "launch_rules": {"order": launch},
        "stages": stages,
        "trial_accounting": {
            "smoke_trials": sum(s["planned_trials"] for s in stages if s["name"].startswith("smoke")),
            "followup_trials": sum(s["planned_trials"] for s in stages if not s["name"].startswith("smoke")),
        },
        "analysis": {"status": "exploratory; specified before the follow-up ran; unadjusted", **analysis,
                     "comparison_cells": "from the v4 matrix, which ran earlier; not time-matched"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=ROOT / "analysis/books-wikidata-v2/experiment_plan_v4.json")
    parser.add_argument("--model-config", type=Path,
                        default=ROOT / "analysis/books-wikidata-v2/model_config_v4_followup.json")
    parser.add_argument("--prompt-output", type=Path,
                        default=ROOT / "analysis/books-wikidata-v2/experiment_plan_v4_prompt_open.json")
    parser.add_argument("--verify-output", type=Path,
                        default=ROOT / "analysis/books-wikidata-v2/experiment_plan_v4_verify.json")
    parser.add_argument("--protocol-output", type=Path,
                        default=ROOT / "analysis/books-wikidata-v2/experiment_plan_v4_protocol.json")
    parser.add_argument("--only", nargs="+", choices=["prompt", "verify", "protocol"],
                        default=["prompt", "verify", "protocol"], help="plans to write")
    parser.add_argument("--freeze", action="store_true", help="mark frozen at HEAD; refuses uncommitted inputs")
    args = parser.parse_args(argv)
    if args.freeze and (dirty := _dirty_inputs()):
        raise SystemExit(f"refusing to freeze with uncommitted inputs: {dirty}")
    base = json.loads(args.base.read_text())
    config = json.loads(args.model_config.read_text())
    tasks = next(s for s in base["stages"] if s["name"] == "reliability")["task_ids"]
    smoke = tasks[:3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                          cwd=ROOT).stdout.strip() if args.freeze else ""
    prompt = _plan(
        base, args.base, config, args.model_config, version="books-wikidata-v4-prompt-open-v1", ceiling=20.0,
        stages=[
            _stage("smoke_prompt_variant", "wiring check for the prompt variant; excluded from estimates",
                   [GLM, QWEN], smoke, 1, VARIANT),
            _stage("prompt_variant", "reliability block with the ask-requirements prompt variant",
                   [GLM, QWEN], tasks, 8, VARIANT),
        ],
        launch="smoke_prompt_variant first, then prompt_variant",
        analysis={
            "prompt_variant": (
                "per agent, variant minus the v4 reliability cell (same agent, structured, 20k, K=8): pass^1 "
                "overall and by condition, and the share of mixed trials with every on_ask fact disclosed; "
                "paired family bootstrap, B = 10,000, seed 20261014. Reported as reference numbers for the "
                "recommended second condition and for hidden tasks; no gap-closing claim."),
        },
        freeze=args.freeze, head=head,
    )
    verify = _plan(
        base, args.base, config, args.model_config, version="books-wikidata-v4-verify-v1", ceiling=25.0,
        stages=[
            _stage("smoke_verify_variant", "wiring check for the verify variant; excluded from estimates",
                   V4_AGENTS, smoke, 1, VERIFY_VARIANT),
            _stage("verify_variant", "reliability block with the ask-and-verify prompt variant",
                   V4_AGENTS, tasks, 8, VERIFY_VARIANT),
        ],
        launch="smoke_verify_variant first, then verify_variant",
        analysis={
            "verify_variant": (
                "per agent, task-averaged pass^1 on volunteer and mixed tasks, hidden success on satisfiable "
                "tasks with and without a cap of three rejected proposals, and hidden success on NVR tasks, "
                "beside the same agent's base and ask-requirements cells (prompt_headroom.py); the share of "
                "hidden failures whose final book was never proposed (hidden_failures.py); paired family "
                "bootstrap of verify minus ask-requirements on capped hidden success, B = 10,000, seed "
                "20261014. Descriptive reference numbers; no claim that the hidden condition is solved."),
        },
        freeze=args.freeze, head=head,
    )
    protocol = _plan(
        base, args.base, config, args.model_config, version="books-wikidata-v4-protocol-v1", ceiling=25.0,
        stages=[
            _stage("smoke_protocol_variant", "wiring check for the protocol variant; excluded from estimates",
                   V4_AGENTS, smoke, 1, PROTOCOL_VARIANT),
            _stage("protocol_variant", "reliability block with the ask-verify-protocol prompt variant",
                   V4_AGENTS, tasks, 8, PROTOCOL_VARIANT),
        ],
        launch="smoke_protocol_variant first, then protocol_variant",
        analysis={
            "protocol_variant": (
                "per agent, the same measures as the verify follow-up (prompt_headroom.py: task-averaged pass^1 "
                "on volunteer and mixed tasks, hidden success on satisfiable tasks with and without a cap of three "
                "rejected proposals, hidden success on NVR tasks; hidden_failures.py: the untested share of "
                "hidden failures), beside the same agent's base, ask-requirements and ask-and-verify cells and the "
                "popularity and eliminating references on the reliability block; paired family bootstrap of "
                "protocol minus ask-and-verify on hidden satisfiable success, B = 10,000, seed 20261014. Reads "
                "how much of the gap to the eliminating agent closes when agents are told what declines and "
                "rejections mean; descriptive, no claim that the hidden condition is solved."),
        },
        freeze=args.freeze, head=head,
    )
    outputs = {"prompt": (args.prompt_output, prompt), "verify": (args.verify_output, verify),
               "protocol": (args.protocol_output, protocol)}
    for path, plan in (outputs[name] for name in args.only):
        path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}: {plan['trial_accounting']} ({plan['status']}, HEAD {head[:12] or 'not recorded'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
