#!/usr/bin/env python3
"""Stage machinery for the books matrix launcher (``run_v4_matrix.py``).

Plans each stage's cells, launches and resumes them, and validates their run
directories.  Time matching: every cell of one agent in a stage (both
retrieval arms, and all catalog sizes for the scale stage) runs as one
concurrent wave, so arm and size contrasts never span different hours.
Budget: one global ledger sums provider spend over every stage under
``--output``; each concurrent cell receives an equal share of what remains.
Stalls: a wall-capped trial whose uncovered plus failed-call time reaches the
plan's threshold is quarantined and re-run once.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path

# The launcher sets these for its plan.
VERSION = "books-matrix-stages-v1"
STATE_SUFFIX = "_state.json"
COMMITTED_PATHS = ("src/tau_rec",)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _usage_spend(run_dir: Path) -> tuple[float, int, int]:
    spend = 0.0
    legacy_usage_total = 0.0
    cumulative_budget_totals: list[float] = []
    usage_paths = sorted(run_dir.glob("usage*.json"))
    for path in usage_paths:
        payload = json.loads(path.read_text())
        provider = payload.get("provider_reported_cost_usd", {})
        legacy_usage_total += float(provider.get("total_usd", 0.0) or 0.0)
        cumulative = payload.get("provider_budget", {}).get("total_usd")
        if isinstance(cumulative, (int, float)) and not isinstance(cumulative, bool):
            cumulative_budget_totals.append(float(cumulative))
    if usage_paths:
        spend = (
            max(cumulative_budget_totals)
            if cumulative_budget_totals else legacy_usage_total
        )

    # Trace-level coverage spans interrupted and resumed processes. It also
    # supplies spend when a killed first process never wrote usage.json.
    trace_spend = 0.0
    reported = 0
    successful = 0
    for trace_path in sorted((run_dir / "traces").glob("*.json")):
        trace = json.loads(trace_path.read_text())
        for calls in trace.get("api_calls", {}).values():
            for call in calls:
                if call.get("status") != "ok":
                    continue
                successful += 1
                reported_cost = call.get("provider_reported_cost_usd")
                if isinstance(reported_cost, (int, float)) and not isinstance(
                    reported_cost, bool
                ):
                    trace_spend += float(reported_cost)
                    reported += 1
    if not usage_paths:
        spend = trace_spend
    return spend, reported, successful


def _trace_pairs(trace_dir: Path) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for path in trace_dir.glob("*_trial*.json"):
        task_id, _, raw_trial = path.stem.rpartition("_trial")
        if task_id and raw_trial.isdigit():
            pairs.add((task_id, int(raw_trial)))
    return pairs


def stall_seconds(trace: dict, cap_seconds: float) -> float | None:
    """Infrastructure time in a wall-capped trial, or ``None`` if not capped."""
    runtime = float(trace.get("trial_runtime_s") or 0.0)
    if runtime < cap_seconds - 1:
        return None
    calls = [call for rows in trace.get("api_calls", {}).values() for call in rows]
    timed = [
        (datetime.fromisoformat(call["started_at"]).timestamp(), call)
        for call in calls if call.get("started_at")
    ]
    if not timed:
        return runtime
    start = min(stamp for stamp, _ in timed)
    covered = max(stamp + float(call.get("latency_seconds") or 0.0) for stamp, call in timed) - start
    failed = sum(
        float(call.get("latency_seconds") or 0.0) for _, call in timed if call.get("status") != "ok"
    )
    return max(0.0, runtime - covered) + failed


def stalled_traces(run_dir: Path, cap_seconds: float, threshold_seconds: float) -> list[Path]:
    stalled = []
    for path in sorted((run_dir / "traces").glob("*.json")):
        seconds = stall_seconds(json.loads(path.read_text()), cap_seconds)
        if seconds is not None and seconds >= threshold_seconds:
            stalled.append(path)
    return stalled


def build_cells(plan: dict, stage: dict) -> list[dict]:
    agents = {row["label"]: row for row in plan["models"]["agents"]}
    simulators = {row["label"]: row for row in plan["models"]["llm_simulators"]}
    cells = []
    for label in stage["agent_labels"]:
        agent = agents[label]
        for simulator in stage["simulator_labels"] or [None]:
            for retrieval in stage["retrieval_modes"]:
                for size in stage["catalog_sizes"]:
                    sim = simulators.get(simulator) if simulator else None
                    cells.append({
                        "key": "__".join(filter(None, [label, simulator, retrieval, str(size)])),
                        "wave": label,
                        "agent_label": label,
                        "model": agent["model"],
                        "reasoning_effort": agent.get("reasoning_effort"),
                        "provider_order": agent.get("provider_order"),
                        "simulator_mode": stage["simulator_mode"],
                        "simulator_label": simulator,
                        "simulator_model": sim["model"] if sim else None,
                        "simulator_provider_order": sim.get("provider_order") if sim else None,
                        "retrieval_mode": retrieval,
                        "catalog_size": size,
                        "trials_per_task": stage["trials_per_task"],
                        "expected_trials": stage["task_count"] * stage["trials_per_task"],
                        "status": "planned",
                        "run_dir": None,
                        "stall_retries": 0,
                        "quarantined_traces": [],
                    })
    return cells


def cell_output(root: Path, stage: str, cell: dict) -> Path:
    parts = [stage, cell["agent_label"]]
    if cell["simulator_label"]:
        parts.append(cell["simulator_label"])
    return root.joinpath(*parts, cell["retrieval_mode"], str(cell["catalog_size"]))


def run_command(args, stage: dict, cell: dict, *, concurrency: int, ceiling: float) -> list[str]:
    runtime = args.runtime
    command = [
        "uv", "run", "tau-rec", "run",
        "--track", _stable_path(args.artifact),
        "--model", cell["model"],
        "--output", _stable_path(cell_output(args.output, stage["name"], cell)),
        "--simulator-mode", cell["simulator_mode"],
        "--retrieval-mode", cell["retrieval_mode"],
        "--catalog-size", str(cell["catalog_size"]),
        "--tasks-filter", stage["tasks_filter"],
        "--trials", str(cell["trials_per_task"]),
        "--max-turns", str(runtime["max_turns"]),
        "--trial-timeout-seconds", str(runtime["trial_timeout_seconds"]),
        "--concurrency", str(concurrency),
        "--max-provider-cost-usd", f"{ceiling:.8f}",
    ]
    if cell["reasoning_effort"]:
        command += ["--reasoning-effort", cell["reasoning_effort"]]
    if cell["provider_order"]:
        command += ["--provider-order", ",".join(cell["provider_order"])]
    if cell["simulator_model"]:
        command += ["--simulator-model", cell["simulator_model"]]
    if cell["simulator_provider_order"]:
        command += ["--simulator-provider-order", ",".join(cell["simulator_provider_order"])]
    return command


def resume_command(run_dir: Path, *, concurrency: int, ceiling: float) -> list[str]:
    return [
        "uv", "run", "tau-rec", "run", "--resume", _stable_path(run_dir),
        "--concurrency", str(concurrency), "--max-provider-cost-usd", f"{ceiling:.8f}",
    ]


def validate(run_dir: Path, cell: dict, *, stage: dict, plan: dict, runtime: dict) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return {"status": "missing", "errors": ["missing run_manifest.json"]}
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "track": plan["artifact"]["track_id"],
        "model": cell["model"],
        "reasoning_effort": cell["reasoning_effort"],
        "provider_order": cell["provider_order"],
        "simulator_interaction_mode": cell["simulator_mode"],
        "simulator_provider_order": cell["simulator_provider_order"],
        "retrieval_mode": cell["retrieval_mode"],
        "catalog_size": cell["catalog_size"],
        "trials_per_task": cell["trials_per_task"],
        "tasks_filter": stage["tasks_filter"],
        "max_turns": runtime["max_turns"],
        "trial_timeout_seconds": float(runtime["trial_timeout_seconds"]),
        "catalog_sha256": plan["artifact"]["catalog_sha256"],
        "tasks_sha256": plan["artifact"]["tasks_sha256"],
    }
    if cell["simulator_model"]:
        expected["simulator_model"] = cell["simulator_model"]
    errors = [
        f"{field}: expected {value!r}, found {manifest.get(field)!r}"
        for field, value in expected.items() if manifest.get(field) != value
    ]
    expected_pairs = {
        (task_id, trial) for task_id in stage["task_ids"] for trial in range(cell["trials_per_task"])
    }
    actual_pairs = _trace_pairs(run_dir / "traces")
    if actual_pairs - expected_pairs:
        errors.append(f"unexpected trace pairs: {sorted(actual_pairs - expected_pairs)[:5]}")
    spend, reported, successful = _usage_spend(run_dir)
    if successful and reported != successful:
        errors.append(f"provider cost coverage: {reported}/{successful} successful calls")
    complete = actual_pairs == expected_pairs and not errors and all(
        (run_dir / name).is_file() for name in ("trial_results.json", "task_results.json")
    )
    return {
        "status": "complete" if complete else "partial",
        "trace_pairs": len(actual_pairs),
        "missing_trace_pairs": len(expected_pairs - actual_pairs),
        "provider_spend_usd": spend,
        "provider_cost_coverage": reported / successful if successful else None,
        "errors": errors,
    }


def global_spend(root: Path) -> float:
    """Every dollar under ``root``: live cells, superseded cells, and aborted attempts."""
    total = 0.0
    for state_path in root.glob(f"*{STATE_SUFFIX}"):
        state = json.loads(state_path.read_text())
        for cell in state["cells"]:
            if cell.get("run_dir") and Path(cell["run_dir"]).exists():
                total += _usage_spend(Path(cell["run_dir"]))[0]
        total += sum(float(row.get("provider_spend_usd") or 0.0) for row in state.get("superseded", []))
    for note in root.glob("aborted/*/ABORTED.json"):
        total += float(json.loads(note.read_text()).get("provider_spend_usd") or 0.0)
    return total


DEFINING_FIELDS = (
    "model", "provider_order", "simulator_model", "simulator_provider_order",
    "reasoning_effort", "trials_per_task", "retrieval_mode", "catalog_size",
)


def replan_cells(state: dict, planned: list[dict], *, superseded_root: Path) -> list[dict]:
    """Keep cells whose definition is unchanged; archive and reset the rest.

    A superseded cell's run directory moves under ``superseded_root`` and its
    spend stays on the ledger through ``state["superseded"]``.
    """
    existing = {cell["key"]: cell for cell in state["cells"]}
    merged = []
    for cell in planned:
        old = existing.get(cell["key"])
        if old and all(old.get(field) == cell.get(field) for field in DEFINING_FIELDS):
            merged.append(old)
            continue
        if old and old.get("run_dir") and Path(old["run_dir"]).exists():
            spend = old.get("provider_spend_usd")
            if spend is None:
                spend = _usage_spend(Path(old["run_dir"]))[0]
            destination = superseded_root / old["key"]
            destination.mkdir(parents=True, exist_ok=True)
            moved = destination / Path(old["run_dir"]).name
            shutil.move(old["run_dir"], moved)
            state.setdefault("superseded", []).append({
                "key": old["key"], "run_dir": _stable_path(moved),
                "provider_spend_usd": float(spend), "definition": {f: old.get(f) for f in DEFINING_FIELDS},
            })
        merged.append(cell)
    return merged


def uncommitted(paths=COMMITTED_PATHS) -> list[str]:
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths], capture_output=True, text=True, check=True,
    ).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()


_STATE_LOCK = threading.RLock()


def _write(path: Path, state: dict) -> None:
    with _STATE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def cell_share(remaining: float, slots: int, prior_spend: float) -> float:
    """A cell's cumulative ceiling: an equal share of what remains plus what it spent."""
    return remaining / slots + prior_spend


MISSING_TRIAL_RESUMES = 3


def needs_another_pass(cell: dict, *, stalled: list, max_stall_retries: int) -> str | None:
    """Why a finished pass must be followed by a resume, or ``None``.

    Stalled traces are quarantined and re-run under the frozen stall rule.
    A trial that failed on a transient provider error leaves no trace; its
    cell is resumed a bounded number of times so those trials are re-run.
    """
    if stalled and cell["stall_retries"] < max_stall_retries:
        return "stall"
    if cell["status"] != "complete" and cell.get("missing_resumes", 0) < MISSING_TRIAL_RESUMES:
        return "missing"
    return None


def refuse_under_tests() -> None:
    """No test may start a provider call."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise SystemExit("refusing to launch provider calls from a test")


def _run_wave(args, stage, cells, state, state_path, env, slots: int) -> None:
    """Start or resume every unfinished cell of one agent at once, then resume as needed.

    ``slots`` is the number of cells running anywhere at once, so concurrent
    waves together never exceed the global remaining budget.
    """
    pending = [cell for cell in cells if cell["status"] != "complete"]
    if not pending:
        return
    refuse_under_tests()
    remaining = args.max_provider_cost_usd - global_spend(args.output)
    if remaining <= 0:
        raise SystemExit("global provider ceiling exhausted")
    concurrency = max(1, args.wave_concurrency // len(pending))
    max_stall = args.plan["stall_retry_rule"]["max_retries"]
    label = "RUN"
    while pending:
        processes = []
        for cell in pending:
            prior = _usage_spend(Path(cell["run_dir"]))[0] if cell["run_dir"] else 0.0
            share = cell_share(remaining, slots, prior)
            if cell["run_dir"]:
                command = resume_command(Path(cell["run_dir"]), concurrency=concurrency, ceiling=share)
                before = None
            else:
                out = cell_output(args.output, stage["name"], cell)
                before = set(out.glob("*/run_manifest.json"))
                command = run_command(args, stage, cell, concurrency=concurrency, ceiling=share)
            print(f"{label} {cell['key']} (share ${share:.2f}, concurrency {concurrency})")
            processes.append((cell, before, subprocess.Popen(command, env=env)))
        for cell, before, process in processes:
            code = process.wait()
            if before is not None:
                created = sorted(set(cell_output(args.output, stage["name"], cell).glob("*/run_manifest.json")) - before)
                if len(created) != 1:
                    raise SystemExit(f"{cell['key']}: expected one new run, found {len(created)}")
                cell["run_dir"] = _stable_path(created[0].parent)
            result = validate(Path(cell["run_dir"]), cell, stage=stage, plan=args.plan, runtime=args.runtime)
            with _STATE_LOCK:
                cell.update(result)
                cell["last_exit_code"] = code
        _write(state_path, state)
        if any(cell["last_exit_code"] == 2 for cell in pending):
            raise SystemExit("provider-cost ceiling stopped the matrix")
        retry = []
        for cell in pending:
            run_dir = Path(cell["run_dir"])
            stalled = (
                stalled_traces(
                    run_dir, float(args.runtime["trial_timeout_seconds"]),
                    float(args.plan["stall_retry_rule"]["threshold_seconds"]),
                )
                if cell["stall_retries"] < max_stall else []
            )
            reason = needs_another_pass(cell, stalled=stalled, max_stall_retries=max_stall)
            if reason == "stall":
                quarantine = run_dir / "quarantine" / f"stall_attempt{cell['stall_retries'] + 1}"
                quarantine.mkdir(parents=True, exist_ok=True)
                for path in stalled:
                    shutil.move(str(path), quarantine / path.name)
                with _STATE_LOCK:
                    cell["quarantined_traces"] += [path.name for path in stalled]
                    cell["stall_retries"] += 1
            elif reason == "missing":
                with _STATE_LOCK:
                    cell["missing_resumes"] = cell.get("missing_resumes", 0) + 1
            if reason:
                retry.append(cell)
        _write(state_path, state)
        pending, label = retry, "RESUME"
        remaining = args.max_provider_cost_usd - global_spend(args.output)
    for cell in cells:
        if cell["status"] != "complete":
            raise SystemExit(f"{cell['key']}: incomplete after this wave; rerun the launcher to resume")
