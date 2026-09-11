"""Build the g1 leaderboard entries from the 2026-09-10 run sweep.

Thin wrapper over `tau-rec leaderboard make-entry`, one invocation per run, so
the entries are produced by the same code path an external submitter uses
rather than by a second bespoke scorer.

The earliest runs predate manifest recording, so reasoning effort is passed by
flag; `make-entry` prefers the manifest wherever one exists. The values below
are the flags the runs were actually launched with: every model except luna ran
with no `--reasoning-effort` at all, and luna requires one (the API rejects
tool-using requests without it) and was given `medium`.

    uv run python scripts/make_g1_entries.py
"""
from __future__ import annotations

import collections
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# run dir -> (submission_id, display name, reasoning effort)
RUNS = {
    "dsflash": ("g1-deepseek-flash", "DeepSeek Flash", None),
    "glm": ("g1-glm-53-flash", "GLM-5.3 Flash", None),
    "luna": ("g1-gpt56-luna-medium", "GPT-5.6 Luna (medium thinking)", "medium"),
    "mistral": ("g1-mistral-small-3", "Mistral Small 3", None),
    "minimax": ("g1-minimax-m3", "MiniMax M3", None),
    "kimi": ("g1-kimi-k25", "Kimi K2.5", None),
    "grok": ("g1-grok-43", "Grok 4.3", None),
    # Cheaper g0 configurations re-run under g1, so the paper cohort is not
    # stranded in a table nothing new can be compared against.
    "qwen3-32b": ("g1-qwen3-32b", "Qwen3-32B", None),
    "gemini25flash": ("g1-gemini25-flash", "Gemini 2.5 Flash", None),
    "gpt5mini": ("g1-gpt5-mini", "GPT-5 mini", None),
    "dsv4flash": ("g1-dsv4-flash", "DeepSeek V4 Flash", None),
    "dsv4flash-high": ("g1-dsv4-flash-high", "DeepSeek V4 Flash (high thinking)", "high"),
}

RUN_DATE = "2026-09-10"

# See the note in scripts/seed_leaderboard.py: pinned to a tag, bump together.
TRACES_URL = (
    "https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz"
)


def main() -> int:
    failures: list[str] = []
    for run_name, (sub_id, display, effort) in RUNS.items():
        run_dir = ROOT / "out" / "g1" / run_name
        if not any(run_dir.rglob("traces")):
            print(f"SKIP {sub_id}: no traces under {run_dir}")
            failures.append(run_name)
            continue

        # pass^4 needs n >= 4, and aggregate_pass_k scores a short task as zero
        # rather than excluding it. An incomplete run would therefore land on
        # the board looking merely bad instead of unfinished.
        # Dedup by filename first, matching make-entry: a top-up pass writes
        # the same trial into a new timestamp dir and supersedes it.
        traces = {p.name for d in run_dir.rglob("traces") for p in d.glob("*.json")}
        counts = collections.Counter(name.rsplit("_trial", 1)[0] for name in traces)
        short = [t for t, c in counts.items() if c < 4]
        if len(counts) < 60 or short:
            print(
                f"SKIP {sub_id}: {len(counts)}/60 tasks present, "
                f"{len(short)} with fewer than 4 trials"
            )
            failures.append(run_name)
            continue
        if any(c > 4 for c in counts.values()):
            over = sorted(t for t, c in counts.items() if c > 4)
            print(f"SKIP {sub_id}: {len(over)} task(s) have more than 4 trials: {over[:5]}")
            failures.append(run_name)
            continue
        cmd = [
            "uv", "run", "tau-rec", "leaderboard", "make-entry",
            "--run-dir", str(run_dir),
            "--submission-id", sub_id,
            "--display-name", display,
            "--submitted-by", "tau-rec authors",
            "--generation", "g1",
            "--run-date", RUN_DATE,
            "--traces-url", TRACES_URL,
        ]
        if effort:
            cmd += ["--reasoning-effort", effort]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAIL {sub_id}: {result.stderr.strip() or result.stdout.strip()}")
            failures.append(run_name)
        else:
            print(result.stdout.strip())

    if failures:
        print(f"\n{len(failures)} run(s) did not produce an entry: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
