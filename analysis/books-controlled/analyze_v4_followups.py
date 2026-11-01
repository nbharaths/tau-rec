#!/usr/bin/env python3
"""Analyze the GLM/Qwen prompt-variant follow-up as its frozen plan specifies.

Per agent (experiment_plan_v4_prompt_open.json): the ask-requirements variant
minus the agent's v4 reliability cell, overall and by condition. Levels add
success among trials that did not reach the agent-time cap and the share of
agent calls rate-limited by the provider.

Intervals reuse analyze_v4's paired family bootstrap (B = 10,000, seed
20261014); all of it is exploratory and unadjusted, and the comparison cells
ran earlier, so no contrast is time-matched. A cell with no traces, a reliability task without traces,
or fewer traces per task than planned is refused.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("analyze_v4_addendum", Path(__file__).with_name("analyze_v4_addendum.py"))
addendum = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(addendum)
v4 = addendum.v4

PROMPT_AGENTS = ("glm53_flash_high", "qwen38_27b_high")


def require_complete(name: str, by_task: dict[str, list], trials_per_task: int, expected: set[str]) -> None:
    """Refuse a cell with no traces, a planned task without traces, or a task with too few."""
    if not by_task:
        raise SystemExit(f"{name}: no traces found (is the root right, relative to the working directory?)")
    missing = sorted(expected - set(by_task))
    if missing:
        raise SystemExit(f"{name}: no traces for {len(missing)} planned tasks, e.g. {missing[:5]}")
    short = sorted(task for task, traces in by_task.items() if len(traces) < trials_per_task)
    if short:
        raise SystemExit(f"{name}: fewer than {trials_per_task} traces for {short[:5]}")


def uncapped_success(traces: list[dict]) -> float | None:
    """Success among trials that did not reach the agent-time cap (a serving-speed sensitivity)."""
    return v4._mean(t["evaluation"]["primary_reward"] >= 1 - 1e-9 for t in traces if t["stop_reason"] != "wall_cap")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "analysis/books-wikidata-v2/experiment_plan_v4.json",
                        help="the v4 plan, for the reliability block's task list")
    args = parser.parse_args()
    block = set(next(s for s in json.loads(args.plan.read_text())["stages"] if s["name"] == "reliability")["task_ids"])
    families = json.loads((args.artifact / "family_manifest.json").read_text())["families"]
    family_of, template_of = {}, {}
    for row in families:
        template_of[row["family_id"]] = "|".join(
            sorted({":".join(part.split(":")[:2]) for part in row["constraint_signature"].split("|")}))
        for task_id in row["task_ids"].values():
            family_of[task_id] = row["family_id"]
    tasks = {path.stem: json.loads(path.read_text()) for path in (args.artifact / "tasks").glob("*.json")}
    contrasts, levels = {}, {}

    def interval(name: str, values: dict[str, float]) -> dict:
        row = v4.interval(name, v4._family_mean(values, family_of), template_of)
        row["tasks"] = len(values)
        contrasts[name] = row
        return row

    def cell(root: Path, stage: str, agent: str, k: int) -> dict:
        by_task = addendum._cell(root, stage, agent)
        require_complete(f"{stage}/{agent}", by_task, k, block)
        return by_task

    def describe(name: str, by_task: dict) -> None:
        mixed = [(task, trace) for task, traces in by_task.items() if task.endswith("_m") for trace in traces]
        asked = [{f for d in trace.get("simulator_decisions", []) if d.get("intent") == "field"
                  for f in d.get("requested_fields", [])} for _, trace in mixed]
        wins = [trace for task, traces in by_task.items() if task.endswith("_h") for trace in traces
                if trace["evaluation"]["primary_reward"] >= 1 - 1e-9]
        levels[name] = {
            "pass^1": v4._mean(v4._pass(t, 1) for t in by_task.values()),
            "pass^4": v4._mean(v4._pass(t, 4) for t in by_task.values()),
            **{f"pass^1/{reveal}": v4._mean(v4._pass(t, 1) for task, t in by_task.items() if task.endswith(f"_{letter}"))
               for letter, reveal in v4.REVEALS.items()},
            "mixed_no_elicitation": v4._mean(v4._decided_on_ask(tr, tasks[t])[1] == 0 for t, tr in mixed),
            "mixed_complete_elicitation": v4._mean(
                (lambda pair: pair[0] == pair[1])(v4._decided_on_ask(tr, tasks[t])) for t, tr in mixed),
            "mixed_asked_year_country_or_language": v4._mean(bool(f & addendum.REQUIREMENT_FIELDS) for f in asked),
            "hidden_rejections_per_success": v4._mean(
                sum(d.get("action") == "candidate_verdict" and d.get("verdict") == "rejected"
                    for d in trace.get("simulator_decisions", [])) for trace in wins),
            "success_uncapped": uncapped_success([trace for t in by_task.values() for trace in t]),
            **{f"success_uncapped/{reveal}": uncapped_success(
                [trace for task, t in by_task.items() if task.endswith(f"_{letter}") for trace in t])
               for letter, reveal in v4.REVEALS.items()},
            "provider_rate_limited_call_share": v4._mean(
                "RateLimit" in (call.get("error_type") or "")
                for t in by_task.values() for trace in t for call in trace["api_calls"]["agent"]),
            "stop_classes": dict(Counter(trace["stop_reason"] for t in by_task.values() for trace in t)),
            "trials": sum(len(t) for t in by_task.values()),
        }

    reliability = {agent: cell(args.main_root, "reliability", agent, 8) for agent in PROMPT_AGENTS}

    prompt_ran = (args.prompt_root / "prompt_variant_v4_state.json").exists()
    if prompt_ran:
        for agent in PROMPT_AGENTS:
            variant = cell(args.prompt_root, "prompt_variant", agent, 8)
            default = reliability[agent]
            interval(f"followup/prompt_variant/{agent}/pass^1",
                     {t: v4._pass(variant[t], 1) - v4._pass(default[t], 1) for t in variant})
            for letter, reveal in v4.REVEALS.items():
                interval(f"followup/prompt_variant/{agent}/{reveal}/pass^1",
                         {t: v4._pass(variant[t], 1) - v4._pass(default[t], 1) for t in variant if t.endswith(f"_{letter}")})
            describe(f"followup/prompt_variant/{agent}/default", default)
            describe(f"followup/prompt_variant/{agent}/variant", variant)

    result = {
        "plans": ["analysis/books-wikidata-v2/experiment_plan_v4_prompt_open.json"],
        "status": "exploratory; unadjusted; comparison cells from the v4 matrix (not time-matched)",
        "ran": {"prompt_variant": prompt_ran},
        "contrasts": contrasts,
        "levels": levels,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, row in contrasts.items():
        low, high = row["family_ci95"]
        print(f"{name}: {row['estimate']:+.3f} [{low:+.3f}, {high:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
