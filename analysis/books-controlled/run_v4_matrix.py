#!/usr/bin/env python3
"""Plan, launch, resume, and validate the frozen v4 books matrix.

Built on ``matrix_stages.py`` (stage plans, waves, the spend ledger and
resumes). All selected stages run in one process, and every pending cell of
every stage launches at once. One ledger sums provider spend over every v4 stage under ``--output``,
and each cell's ceiling is an equal share of what remains across all running
cells. Stalls: a trial stopped at the agent-time cap (stop reason
``wall_cap``) whose time outside successful agent calls reaches the plan's
threshold is quarantined and re-run once.

``--execute`` refuses a draft plan, a ceiling above the plan's, uncommitted
inputs, or a harness and track at HEAD that differ from the commit the plan
was frozen on.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_spec = importlib.util.spec_from_file_location("matrix_stages", Path(__file__).with_name("matrix_stages.py"))
stages = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stages)

VERSION = "books-wikidata-v4-launcher-v1"
stages.VERSION = VERSION
stages.STATE_SUFFIX = "_v4_state.json"
stages.COMMITTED_PATHS = (
    "src/tau_rec",
    "data/tracks/books-wikidata-v2",
    "analysis/books-controlled/matrix_stages.py",
    "analysis/books-controlled/run_v4_matrix.py",
    "analysis/books-wikidata-v2/experiment_plan_v4.json",
    "analysis/books-wikidata-v2/model_config_v4.json",
    "analysis/books-wikidata-v2/panels.json",
)
# What every v4 trial must run on: the commit recorded when the plan was frozen.
FROZEN_PATHS = ("src/tau_rec", "data/tracks/books-wikidata-v2")


def stall_seconds(trace: dict, cap_seconds: float) -> float | None:
    """Agent time outside successful agent calls, or ``None`` if not wall-capped.

    Under the books-v4 harness the cap counts only the agent's time, so time
    the user simulator spent is not a stall and does not appear here.
    """
    if trace.get("stop_reason") != "wall_cap":
        return None
    calls = trace.get("api_calls", {}).get("agent", [])
    successful = sum(float(call.get("latency_seconds") or 0.0) for call in calls if call.get("status") == "ok")
    return max(0.0, cap_seconds - successful)


def stalled_traces(run_dir: Path, cap_seconds: float, threshold_seconds: float) -> list[Path]:
    stalled = []
    for path in sorted((run_dir / "traces").glob("*.json")):
        seconds = stall_seconds(json.loads(path.read_text()), cap_seconds)
        if seconds is not None and seconds >= threshold_seconds:
            stalled.append(path)
    return stalled


_stage_validate = stages.validate


def validate(run_dir: Path, cell: dict, *, stage: dict, plan: dict, runtime: dict) -> dict:
    """The stage checks, plus the harness profile and a clean harness commit."""
    result = _stage_validate(run_dir, cell, stage=stage, plan=plan, runtime=runtime)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return result
    manifest = json.loads(manifest_path.read_text())
    errors = []
    if manifest.get("agent_prompt_variant") != stage.get("agent_prompt_variant"):
        errors.append(
            f"agent_prompt_variant: expected {stage.get('agent_prompt_variant')!r}, "
            f"found {manifest.get('agent_prompt_variant')!r}"
        )
    profile = (manifest.get("harness_profile") or {}).get("id")
    if profile != plan["harness"]["profile"]["id"]:
        errors.append(f"harness_profile: expected {plan['harness']['profile']['id']!r}, found {profile!r}")
    commit = manifest.get("harness_commit") or {}
    freeze = (plan["harness"].get("commit_at_freeze") or {}).get("sha")
    warnings = []
    if not commit.get("sha") or not freeze or not harness_same(freeze, commit["sha"]):
        errors.append(f"harness_commit: expected the frozen harness ({(freeze or '?')[:12]}), found {commit!r}")
    elif commit.get("dirty"):
        # harness_commit flags any modified tracked file; --execute refuses to launch or resume
        # unless the harness and track paths are committed and match the freeze.
        warnings.append("the working tree had modified tracked files outside the guarded harness paths")
    if errors:
        result = {**result, "status": "partial", "errors": result["errors"] + errors}
    return {**result, "warnings": warnings}


_stage_run_command = stages.run_command


def run_command(args, stage: dict, cell: dict, *, concurrency: int, ceiling: float) -> list[str]:
    """The stage command, plus the stage's agent prompt variant when it names one."""
    command = _stage_run_command(args, stage, cell, concurrency=concurrency, ceiling=ceiling)
    if stage.get("agent_prompt_variant"):
        command += ["--agent-prompt-variant", stage["agent_prompt_variant"]]
    return command


stages.stall_seconds, stages.stalled_traces, stages.validate = stall_seconds, stalled_traces, validate
stages.run_command = run_command


def uncommitted_inputs(plan_path: Path | None = None) -> list[str]:
    """Uncommitted v4 inputs, including the plan being launched.

    Passed explicitly: the stage module's default holds its own paths, and an
    addendum plan is not among the fixed v4 paths.
    """
    paths = stages.COMMITTED_PATHS + ((str(plan_path),) if plan_path else ())
    return stages.uncommitted(paths)


def harness_matches_freeze(freeze_sha: str, paths=FROZEN_PATHS) -> bool:
    """Whether the harness and track at HEAD are identical to the freeze commit."""
    return subprocess.run(["git", "diff", "--quiet", freeze_sha, "HEAD", "--", *paths]).returncode == 0


def harness_same(freeze_sha: str, sha: str, paths=FROZEN_PATHS) -> bool:
    """Whether the harness and track at ``sha`` are identical to the freeze commit."""
    return subprocess.run(["git", "diff", "--quiet", freeze_sha, sha, "--", *paths]).returncode == 0


def wave_concurrency(total: int, wave_pending: int, all_pending: int) -> int:
    """A wave's share of the global concurrency, in proportion to its pending cells."""
    return max(1, round(total * wave_pending / max(1, all_pending)))


def select_waves(stages: list[dict], loaded: dict, agents: list[str] | None) -> list[tuple]:
    """One wave per agent per stage; with ``agents``, only those agents' waves."""
    known = {label for stage in stages for label in stage["agent_labels"]}
    unknown = sorted(set(agents or []) - known)
    if unknown:
        raise SystemExit(f"unknown agent label(s) {unknown}; the selected stages have {sorted(known)}")
    waves = []
    for stage in stages:
        state, state_path = loaded[stage["name"]]
        grouped: dict[str, list[dict]] = {}
        for cell in state["cells"]:
            grouped.setdefault(cell["wave"], []).append(cell)
        waves += [
            (stage, cells, state, state_path) for label, cells in grouped.items()
            if agents is None or label in agents
        ]
    return waves


def _load_state(args, plan_path: Path, plan: dict, stage: dict) -> tuple[dict, Path]:
    state_path = args.output / f"{stage['name']}{stages.STATE_SUFFIX}"
    planned = stages.build_cells(plan, stage)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["plan_sha256"] != stages._sha256(plan_path):
            if not args.replan:
                raise SystemExit(f"{stage['name']}: existing state belongs to a different plan; pass --replan")
            state["cells"] = stages.replan_cells(
                state, planned, superseded_root=args.output / "superseded" / stage["name"],
            )
            state.setdefault("plan_history", []).append(state["plan_sha256"])
            state["plan_sha256"] = stages._sha256(plan_path)
        if {cell["key"] for cell in state["cells"]} != {cell["key"] for cell in planned}:
            raise SystemExit(f"{stage['name']}: existing state has a different cell set")
    else:
        state = {
            "version": VERSION, "stage": stage["name"], "status": "planned",
            "plan_sha256": stages._sha256(plan_path), "cells": planned,
        }
    return state, state_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", required=True, help="comma-separated stage names, run together")
    parser.add_argument("--agents", help="comma-separated agent labels; only their cells run")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-provider-cost-usd", type=float)
    parser.add_argument("--total-concurrency", type=int, default=48)
    parser.add_argument("--replan", action="store_true",
                        help="adopt a changed plan: keep unchanged cells, supersede changed ones")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    checks = {
        "model_config_sha256": stages._sha256(args.model_config),
        "panels_sha256": stages._sha256(args.panels),
        "catalog_sha256": stages._sha256(args.artifact / "catalog.json"),
        "task_manifest_sha256": stages._sha256(args.artifact / "task_manifest.json"),
        "policy_sha256": stages._sha256(args.artifact / "policy.md"),
    }
    drifted = [key for key, value in checks.items() if plan["artifact"][key] != value]
    if drifted:
        raise SystemExit(f"plan does not match frozen inputs: {drifted}")
    by_name = {row["name"]: row for row in plan["stages"]}
    names = args.stages.split(",")
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}")
    selected = [by_name[name] for name in names]
    plan_path = args.plan
    loaded = {stage["name"]: _load_state(args, plan_path, plan, stage) for stage in selected}
    # The stage wave runner reads the plan and the runtime from args.
    args.plan, args.runtime = plan, plan["fixed_runtime"]
    for stage in selected:
        cells = loaded[stage["name"]][0]["cells"]
        print(f"{stage['name']}: {len(cells)} cells, {sum(c['expected_trials'] for c in cells):,} trials")
    if not args.execute:
        for state, state_path in loaded.values():
            stages._write(state_path, state)
        print("State written; no provider calls made.")
        return 0

    if plan.get("status") == "draft-not-frozen":
        raise SystemExit("the plan is a draft; freeze it with the authors' ceiling first")
    ceiling = plan["budgets_usd"].get("v4_hard_ceiling")
    if not isinstance(ceiling, (int, float)):
        raise SystemExit("the plan has no v4 hard ceiling")
    if args.max_provider_cost_usd is None or not 0 < args.max_provider_cost_usd <= ceiling:
        raise SystemExit(f"--execute needs --max-provider-cost-usd in (0, {ceiling}]")
    dirty = uncommitted_inputs(plan_path)
    if dirty:
        raise SystemExit(f"commit the frozen inputs first; uncommitted: {dirty[:5]}")
    freeze = plan["harness"].get("commit_at_freeze") or {}
    if not freeze.get("sha") or freeze.get("dirty"):
        raise SystemExit("the plan was not frozen on a clean commit")
    if not harness_matches_freeze(freeze["sha"]):
        raise SystemExit(f"harness or track differ from the freeze commit {freeze['sha'][:12]}")

    stages.refuse_under_tests()
    head = stages._git_head()
    for state, state_path in loaded.values():
        state.update({"git_commit": head, "status": "running"})
        stages._write(state_path, state)
    env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    waves = select_waves(selected, loaded, args.agents.split(",") if args.agents else None)
    all_pending = sum(1 for _, cells, _, _ in waves for cell in cells if cell["status"] != "complete")
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, len(waves))) as pool:
        futures = []
        for stage, cells, state, state_path in waves:
            pending = sum(1 for cell in cells if cell["status"] != "complete")
            wave_args = argparse.Namespace(**vars(args))
            wave_args.wave_concurrency = wave_concurrency(args.total_concurrency, pending, all_pending)
            futures.append(pool.submit(
                stages._run_wave, wave_args, stage, cells, state, state_path, env, max(1, all_pending),
            ))
        for future in futures:
            try:
                future.result()
            except SystemExit as error:
                failures.append(str(error))
    for stage in selected:
        state, state_path = loaded[stage["name"]]
        if all(cell["status"] == "complete" for cell in state["cells"]):
            state["status"] = "complete"
        state["provider_spend_usd"] = sum(float(c.get("provider_spend_usd") or 0.0) for c in state["cells"])
        stages._write(state_path, state)
        print(f"{stage['name']}: {state['status']}, ${state['provider_spend_usd']:.4f} provider spend")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
