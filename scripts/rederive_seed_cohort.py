"""Re-derive {n, c} for every archived run and diff against stored task_results.json.

Run directories live under results/final-traces/<run>/<timestamp>/, which is
gitignored but present on the author's machine. Usage:

    uv run python scripts/rederive_seed_cohort.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.conversation import ConversationTrace
from tau_rec.data_model.task import Task
from tau_rec.evaluator.evaluator import CombinedEvaluator
from tau_rec.metrics.pass_k import aggregate_pass_k

ROOT = Path(__file__).resolve().parent.parent
TRACE_ROOT = ROOT / "results" / "final-traces"


def load_tasks() -> dict[str, Task]:
    return {
        p.stem: Task.model_validate_json(p.read_text())
        for p in sorted((ROOT / "data" / "tasks").glob("*.json"))
    }


def rederive(run_dir: Path, tasks: dict[str, Task], ev: CombinedEvaluator) -> dict[str, dict]:
    counts: dict[str, dict] = {}
    for tp in sorted((run_dir / "traces").glob("*.json")):
        trace = ConversationTrace.model_validate_json(tp.read_text())
        result = ev.evaluate(task=tasks[trace.task_id], trace=trace)
        entry = counts.setdefault(trace.task_id, {"n": 0, "c": 0})
        entry["n"] += 1
        entry["c"] += 1 if result.primary_reward == 1.0 else 0
    return counts


def main() -> int:
    if not TRACE_ROOT.is_dir():
        print(f"no trace archive at {TRACE_ROOT}", file=sys.stderr)
        return 1

    catalog = Catalog.from_json(str(ROOT / "data" / "catalog.json"))
    tasks = load_tasks()
    ev = CombinedEvaluator(catalog)

    total_deltas = 0
    header = f"{'config':<34}{'stored p4':>10}{'rederived p4':>14}  changed"
    print(header)
    print("-" * len(header))

    for run in sorted(TRACE_ROOT.iterdir()):
        if not run.is_dir():
            continue
        for ts in sorted(run.iterdir()):
            if not (ts / "traces").is_dir() or not (ts / "task_results.json").exists():
                continue

            stored = json.loads((ts / "task_results.json").read_text())
            fresh = rederive(ts, tasks, ev)

            deltas = {
                tid: f"c {stored[tid]['c']} -> {fresh[tid]['c']}"
                for tid in sorted(fresh)
                if tid in stored and stored[tid]["c"] != fresh[tid]["c"]
            }
            total_deltas += len(deltas)
            print(
                f"{run.name:<34}"
                f"{aggregate_pass_k(stored, 4):>10.4f}"
                f"{aggregate_pass_k(fresh, 4):>14.4f}"
                f"  {deltas or ''}"
            )

    print(f"\n{total_deltas} task(s) changed across all configs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
