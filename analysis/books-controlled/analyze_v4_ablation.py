#!/usr/bin/env python3
"""Analyze the exploratory v4 field-list ablation as experiment_plan_v4_ablation.json specifies it.

Per agent (DeepSeek, Qwen), on the 18-family reliability block at 20k:

- field list: keyword with the structured arm's field list in its tool line
  minus the v4 reliability keyword cell (same agent, K=8);
- retrieval with the field list: the v4 reliability structured cell minus
  keyword with the field list;

each on pass^1 overall and on volunteer, mixed and hidden tasks, with the
share of mixed trials in which the symbolic user disclosed every askable fact
and, added after the run as a description, the success rate of trials the
agent-time cap did not stop.
Intervals reuse analyze_v4's paired family bootstrap (B = 10,000, seed
20261014); everything here is exploratory and unadjusted, and the v4 cells ran
earlier, so the contrasts are not time-matched.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location("analyze_v4", Path(__file__).with_name("analyze_v4.py"))
v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4)

AGENTS = ("deepseek_v41_flash_high", "qwen38_27b_high")


def _cell(root: Path, stage: str, agent: str, arm: str) -> dict[str, list[dict]]:
    state = json.loads((root / f"{stage}_v4_state.json").read_text())
    if any(cell["status"] != "complete" for cell in state["cells"]):
        raise SystemExit(f"{stage}: not every cell is complete")
    rows = [cell for cell in state["cells"] if cell["agent_label"] == agent and not cell.get("simulator_label")
            and cell["retrieval_mode"] == arm and cell["catalog_size"] == 20000]
    if len(rows) != 1:
        raise SystemExit(f"{stage}/{agent}/{arm}: expected one cell, found {len(rows)}")
    return v4._trials(Path(rows[0]["run_dir"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    families = json.loads((args.artifact / "family_manifest.json").read_text())["families"]
    family_of, template_of = {}, {}
    for row in families:
        template_of[row["family_id"]] = "|".join(
            sorted({":".join(part.split(":")[:2]) for part in row["constraint_signature"].split("|")}))
        for task_id in row["task_ids"].values():
            family_of[task_id] = row["family_id"]
    tasks = {path.stem: json.loads(path.read_text()) for path in (args.artifact / "tasks").glob("*.json")}
    contrasts, levels = {}, {}

    def contrast(name: str, treated: dict, control: dict, letter: str | None = None) -> None:
        values = {task: v4._pass(treated[task], 1) - v4._pass(control[task], 1)
                  for task in treated if task in control and (letter is None or task.endswith(f"_{letter}"))}
        row = v4.interval(name, v4._family_mean(values, family_of), template_of)
        row["tasks"] = len(values)
        contrasts[name] = row

    def describe(name: str, by_task: dict) -> None:
        mixed = [(task, trace) for task, traces in by_task.items() if task.endswith("_m") for trace in traces]
        levels[name] = {
            "pass^1": v4._mean(v4._pass(t, 1) for t in by_task.values()),
            **{f"pass^1/{reveal}": v4._mean(v4._pass(t, 1) for task, t in by_task.items()
                                            if task.endswith(f"_{letter}"))
               for letter, reveal in v4.REVEALS.items()},
            "mixed_complete_elicitation": v4._mean(
                (lambda pair: pair[0] == pair[1])(v4._decided_on_ask(trace, tasks[task])) for task, trace in mixed),
            "searches_per_trial": v4._mean(v4._searches(trace) for t in by_task.values() for trace in t),
            "success_uncapped": v4._mean(trace["evaluation"]["primary_reward"] >= 1 - 1e-9
                                         for t in by_task.values() for trace in t
                                         if trace["stop_reason"] != "wall_cap"),
            "stop_classes": dict(Counter(trace["stop_reason"] for t in by_task.values() for trace in t)),
            "trials": sum(len(t) for t in by_task.values()),
        }

    for agent in AGENTS:
        keyword = _cell(args.main_root, "reliability", agent, "keyword")
        structured = _cell(args.main_root, "reliability", agent, "structured")
        field_list = _cell(args.ablation_root, "keyword_field_list", agent, "keyword")
        describe(f"ablation/{agent}/keyword", keyword)
        describe(f"ablation/{agent}/keyword_field_list", field_list)
        describe(f"ablation/{agent}/structured", structured)
        for letter, label in ((None, "all"), *v4.REVEALS.items()):
            contrast(f"ablation/{agent}/field_list/{label}/pass^1", field_list, keyword, letter)
            contrast(f"ablation/{agent}/retrieval_with_field_list/{label}/pass^1", structured, field_list, letter)

    state = json.loads((args.ablation_root / "keyword_field_list_v4_state.json").read_text())
    result = {
        "plan_version": "books-wikidata-v4-ablation-v1",
        "seed": v4.SEED,
        "iterations": v4.ITERATIONS,
        "status": "exploratory; unadjusted; not time-matched",
        "run_accounting": {"trials": sum(cell["trace_pairs"] for cell in state["cells"]),
                           "provider_spend_usd": state["provider_spend_usd"],
                           "git_commit": state["git_commit"], "plan_sha256": state["plan_sha256"]},
        "contrasts": contrasts,
        "levels": levels,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, row in contrasts.items():
        lo, hi = row["family_ci95"]
        print(f"{name} {row['estimate']:+.3f} [{lo:+.3f}, {hi:+.3f}] p={row['family_p']:.4f}")
    for name, row in levels.items():
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
