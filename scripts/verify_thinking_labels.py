"""Recompute the per-step latency evidence behind the g0 thinking-mode labels.

Three `g0` leaderboard rows share the model string `deepseek/deepseek-v4-flash`
and differ only in thinking mode. No trace records the mode, so the labels rest
on per-step agent latency: more thinking means a slower LLM call, and the three
runs should separate in the order the labels predict. This script recomputes
that separation from the traces so the claim in `leaderboard/GENERATIONS.md` is
checkable rather than asserted.

Step latency is the right measure here, not `trial_runtime_s`. Trial runtime is
dominated by how many tool calls a model happens to make, which varies several-
fold between trials on identical input; per-step latency isolates the cost of a
single LLM call.

`GENERATIONS.md` quotes the one-sided p-values, since the hypothesis is
directional. Both are printed below, because scipy defaults to two-sided and
that gives exactly double -- a difference worth seeing rather than tripping over.

Usage:
    uv run python scripts/verify_thinking_labels.py
    uv run python scripts/verify_thinking_labels.py non-think=out/a think-max=out/b

Trace archives are not in the repo (`out/` is gitignored). Download the tarball
from the release whose `traces_url` the entries point at, then pass the run
directories as `label=path` arguments in the order the labels predict.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parent.parent

# The archives behind the published claim, in predicted-latency order.
DEFAULT_RUNS = [
    ("non-think", "out/ds-v4-flash-4t-gpt5mini/20260502_193511"),
    ("think-high", "out/ds-v4-flash-thinking-4t/20260502_194508"),
    ("think-max", "out/ds-v4-flash-thinking-max/20260503_081900"),
]


def step_latencies(run_dir: Path) -> list[float]:
    """Every agent step latency across every trace in a run directory."""
    traces = run_dir / "traces"
    if not traces.is_dir():
        raise SystemExit(
            f"No traces/ under {run_dir}.\n"
            "Pass run directories explicitly: label=path label=path ..."
        )
    out: list[float] = []
    for path in sorted(traces.glob("*.json")):
        trace = json.loads(path.read_text())
        out.extend(trace.get("agent_step_latencies_s") or [])
    if not out:
        raise SystemExit(f"{run_dir} has traces but no agent_step_latencies_s.")
    return out


def parse_args(argv: list[str]) -> list[tuple[str, str]]:
    if not argv:
        return DEFAULT_RUNS
    runs = []
    for arg in argv:
        label, sep, path = arg.partition("=")
        if not sep:
            raise SystemExit(f"Expected label=path, got {arg!r}")
        runs.append((label, path))
    if len(runs) < 2:
        raise SystemExit("Need at least two runs to compare.")
    return runs


def main(argv: list[str]) -> int:
    runs = parse_args(argv)
    latencies = {}
    for label, path in runs:
        run_dir = Path(path)
        if not run_dir.is_absolute():
            run_dir = REPO / run_dir
        latencies[label] = step_latencies(run_dir)

    print(f"{'run':12} {'steps':>6} {'median':>9} {'IQR':>18}")
    for label, _ in runs:
        vals = latencies[label]
        q1, q3 = statistics.quantiles(vals, n=4)[0], statistics.quantiles(vals, n=4)[2]
        print(f"{label:12} {len(vals):6} {statistics.median(vals):8.2f}s "
              f"{q1:8.2f}-{q3:.2f}s")

    print()
    ordered = True
    labels = [label for label, _ in runs]
    for a, b in zip(labels, labels[1:]):
        _, p_one = mannwhitneyu(latencies[a], latencies[b], alternative="less")
        _, p_two = mannwhitneyu(latencies[a], latencies[b], alternative="two-sided")
        faster = statistics.median(latencies[a]) < statistics.median(latencies[b])
        ordered = ordered and faster
        print(f"{a} < {b}: p={p_one:.2e} one-sided, {p_two:.2e} two-sided"
              f"{'' if faster else '   <-- ORDER VIOLATED'}")

    print()
    if ordered:
        print("Medians separate in the order the labels predict.")
        return 0
    print("Medians do NOT separate in label order; the labels are unsupported.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
