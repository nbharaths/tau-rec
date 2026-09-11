"""Generate the g0 leaderboard entries from the archived paper runs.

g0 is the nine configurations behind the RecSys paper. Counts are re-derived
from traces with the current evaluator rather than copied from the stored
task_results.json, because re-derived is canonical.

    uv run python scripts/seed_leaderboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.conversation import ConversationTrace
from tau_rec.data_model.task import Task
from tau_rec.evaluator.evaluator import CombinedEvaluator
from tau_rec.leaderboard.entry import LeaderboardEntry

ROOT = Path(__file__).resolve().parent.parent
TRACE_ROOT = ROOT / "results" / "final-traces"
OUT_DIR = ROOT / "leaderboard" / "entries"

# Where the cohort's traces are published. Pinned to a tag, never to
# /releases/latest/, for the same reason model_id may not end in `-latest`.
# Bump alongside scripts/make_g1_entries.py when a release supersedes it.
TRACES_URL = (
    "https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g0-traces.tar.gz"
)

# run directory -> (submission_id, display name, reasoning effort, run date)
# llama33-paper is deliberately absent: 68 trials across 60 tasks, an aborted
# run rather than a ninth configuration.
COHORT = {
    "[final]run-gpt54-nothink": ("g0-gpt54", "GPT-5.4", None, "2026-05-03"),
    "[final]run-gpt54-medthink": ("g0-gpt54-medium", "GPT-5.4 (medium thinking)", "medium", "2026-05-03"),
    "[final]run-sonnet46-nothink": ("g0-sonnet46", "Claude Sonnet 4.6", None, "2026-05-03"),
    "[final]run-gemini25flash": ("g0-gemini25-flash", "Gemini 2.5 Flash", None, "2026-05-03"),
    "ds-v4-flash-4t-gpt5mini": ("g0-dsv4-flash", "DeepSeek V4 Flash", None, "2026-05-02"),
    "ds-v4-flash-thinking-4t": ("g0-dsv4-flash-high", "DeepSeek V4 Flash (high thinking)", "high", "2026-05-02"),
    "ds-v4-flash-thinking-max": ("g0-dsv4-flash-max", "DeepSeek V4 Flash (max thinking)", "xhigh", "2026-05-03"),
    "qwen3-32b-paper": ("g0-qwen3-32b", "Qwen3-32B", None, "2026-05-02"),
    "gpt5mini-paper": ("g0-gpt5-mini", "GPT-5 mini", None, "2026-05-02"),
}


def main() -> int:
    catalog = Catalog.from_json(str(ROOT / "data" / "catalog.json"))
    tasks = {
        p.stem: Task.model_validate_json(p.read_text())
        for p in sorted((ROOT / "data" / "tasks").glob("*.json"))
    }
    ev = CombinedEvaluator(catalog)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for run_name, (sub_id, display, effort, run_date) in COHORT.items():
        run_dir = TRACE_ROOT / run_name
        ts_dirs = [d for d in sorted(run_dir.iterdir()) if (d / "traces").is_dir()]
        ts = ts_dirs[0]

        per_task: dict[str, dict[str, int]] = {}
        model_ids: set[str] = set()
        for tp in sorted((ts / "traces").glob("*.json")):
            trace = ConversationTrace.model_validate_json(tp.read_text())
            model_ids.add(trace.model)
            result = ev.evaluate(task=tasks[trace.task_id], trace=trace)
            row = per_task.setdefault(trace.task_id, {"n": 0, "c": 0})
            row["n"] += 1
            row["c"] += 1 if result.primary_reward == 1.0 else 0

        assert len(model_ids) == 1, f"{run_name}: mixed model ids {model_ids}"

        entry = LeaderboardEntry(
            submission_id=sub_id,
            display_name=display,
            submitted_by="tau-rec authors",
            model_id=model_ids.pop(),
            pinned=False,          # the paper used floating aliases
            run_date=run_date,
            reasoning_effort=effort,
            simulator_model="gpt-5-mini",
            harness_generation="g0",
            tau_rec_version="0.1.0",
            trials_per_task=4,
            # g0 predates digest recording, and its content genuinely differs
            # from HEAD (8-policy prompt; preference_respect still live).
            policy_sha256=None,
            tasks_sha256=None,
            catalog_sha256=None,
            per_task=per_task,
            traces_url=TRACES_URL,
        )
        path = OUT_DIR / f"{sub_id}.json"
        path.write_text(json.dumps(json.loads(entry.model_dump_json()), indent=2) + "\n")
        total_n = sum(v["n"] for v in per_task.values())
        print(f"{sub_id:<24} {entry.model_id:<38} {len(per_task):>3} tasks  {total_n:>4} trials")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
