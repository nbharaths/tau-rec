#!/usr/bin/env python3
"""Analyze the exploratory v4 addendum as experiment_plan_v4_addendum.json specifies it.

- Prompt variant: per agent, the ask-requirements variant minus the matching v4
  reliability cell (same agent, structured, 20k, K=8), on pass^1 overall and on
  mixed tasks, with the share of mixed trials that elicit every askable fact.
- GPT-6.1 Sol with LLM users: per LLM user, GPT with that user minus GPT with the
  symbolic user (v4 reliability cell), on pass^1 overall and on mixed tasks, with
  DeepSeek's levels under the same users (v4 sensitivity cells) for the ranking.

Intervals reuse analyze_v4's paired family bootstrap (B = 10,000, seed
20261014); everything here is exploratory and unadjusted, and the comparison
cells ran earlier, so the contrasts are not time-matched.
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

GPT, DEEPSEEK = "gpt61_sol_high", "deepseek_v41_flash_high"
USERS = ("sim_glm53_flash", "sim_qwen38_27b")
REQUIREMENT_FIELDS = {"publication_year", "countries", "languages"}


def _cell(root: Path, stage: str, agent: str, simulator: str | None = None) -> dict[str, list[dict]]:
    state = json.loads((root / f"{stage}_v4_state.json").read_text())
    if any(cell["status"] != "complete" for cell in state["cells"]):
        raise SystemExit(f"{stage}: not every cell is complete")
    rows = [cell for cell in state["cells"] if cell["agent_label"] == agent
            and cell.get("simulator_label") == simulator and cell["retrieval_mode"] == "structured"
            and cell["catalog_size"] == 20000]
    if len(rows) != 1:
        raise SystemExit(f"{stage}/{agent}/{simulator}: expected one cell, found {len(rows)}")
    return v4._trials(Path(rows[0]["run_dir"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--addendum-root", type=Path, required=True)
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

    def contrast(name: str, treated: dict, control: dict, *, mixed_only: bool = False) -> None:
        values = {task: v4._pass(treated[task], 1) - v4._pass(control[task], 1)
                  for task in treated if task in control and (not mixed_only or task.endswith("_m"))}
        row = v4.interval(name, v4._family_mean(values, family_of), template_of)
        row["tasks"] = len(values)
        contrasts[name] = row

    def describe(name: str, by_task: dict, *, symbolic: bool = True) -> None:
        """Controller-decision measures exist only for the symbolic user (None otherwise)."""
        mixed = [(task, trace) for task, traces in by_task.items() if task.endswith("_m") for trace in traces]
        asked = [{field for decision in trace.get("simulator_decisions", []) if decision.get("intent") == "field"
                  for field in decision.get("requested_fields", [])} for _, trace in mixed]
        levels[name] = {
            "pass^1": v4._mean(v4._pass(t, 1) for t in by_task.values()),
            **{f"pass^1/{reveal}": v4._mean(v4._pass(t, 1) for task, t in by_task.items() if task.endswith(f"_{letter}"))
               for letter, reveal in v4.REVEALS.items()},
            "mixed_no_elicitation": v4._mean(
                v4._decided_on_ask(trace, tasks[task])[1] == 0 for task, trace in mixed) if symbolic else None,
            "stop_classes": dict(Counter(trace["stop_reason"] for t in by_task.values() for trace in t)),
            "mixed_complete_elicitation": v4._mean(
                (lambda pair: pair[0] == pair[1])(v4._decided_on_ask(trace, tasks[task]))
                for task, trace in mixed) if symbolic else None,
            "mixed_asked_year_country_or_language": v4._mean(
                bool(fields & REQUIREMENT_FIELDS) for fields in asked) if symbolic else None,
            "mixed_all_askable_stated_tolerant": v4._mean(
                all(v4._stated(" ".join(v4._user_messages(trace)), value, tolerant=True)
                    for value in v4._values(tasks[task], "on_ask")) for task, trace in mixed),
            "trials": sum(len(t) for t in by_task.values()),
        }

    for agent in (GPT, DEEPSEEK):
        default = _cell(args.main_root, "reliability", agent)
        variant = _cell(args.addendum_root, "prompt_variant", agent)
        contrast(f"addendum/prompt_variant/{agent}/pass^1", variant, default)
        contrast(f"addendum/prompt_variant/{agent}/mixed/pass^1", variant, default, mixed_only=True)
        describe(f"addendum/prompt_variant/{agent}/default", default)
        describe(f"addendum/prompt_variant/{agent}/variant", variant)
    symbolic = _cell(args.main_root, "reliability", GPT)
    describe("addendum/gpt_users/symbolic", symbolic)
    for user in USERS:
        with_user = _cell(args.addendum_root, "gpt_llm_users", GPT, simulator=user)
        contrast(f"addendum/gpt_users/{user}_minus_symbolic/pass^1", with_user, symbolic)
        contrast(f"addendum/gpt_users/{user}_minus_symbolic/mixed/pass^1", with_user, symbolic, mixed_only=True)
        describe(f"addendum/gpt_users/{user}", with_user, symbolic=False)

    # Do the two agents rank the same way under each user? (reliability block, structured, 20k)
    deepseek_users = {"symbolic": _cell(args.main_root, "reliability", DEEPSEEK)}
    deepseek_users.update({user: _cell(args.main_root, "simulator_sensitivity", DEEPSEEK, simulator=user) for user in USERS})
    for user, by_task in deepseek_users.items():
        describe(f"addendum/users/{user}/{DEEPSEEK}", by_task, symbolic=user == "symbolic")

    result = {
        "plan": "analysis/books-wikidata-v2/experiment_plan_v4_addendum.json",
        "status": "exploratory; unadjusted; comparison cells from the v4 matrix (not time-matched)",
        "contrasts": contrasts,
        "levels": levels,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, row in contrasts.items():
        low, high = row["family_ci95"]
        print(f"{name:70} {row['estimate']:+.3f} [{low:+.3f}, {high:+.3f}] p={row['family_p']:.4f}")
    for name, row in levels.items():
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
