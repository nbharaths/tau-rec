#!/usr/bin/env python3
"""Report one books run with the measures of the reportable protocol.

For a run directory written by ``tau-rec run`` on the books track (one agent,
interface, and catalog size), prints:

- pass^1 overall and by disclosure condition, with hidden satisfiable and
  hidden no-valid-answer tasks separately, and hidden satisfiable success
  counting only successes after at most three rejected proposals, each with a
  paired family-bootstrap 95% interval (B = 10,000, seed 20261014);
- pass^4 when every task has at least four trials;
- mixed-task elicitation (share of mixed trials in which the symbolic user
  disclosed every askable fact; symbolic user only);
- mixed-task exposure (share of satisfiable mixed trials whose search or
  metadata results held a policy-eligible book);
- stop classes.

    uv run python scripts/report_books_run.py --run-dir out/<model>/<timestamp>

Reuses the analysis helpers in analysis/books-controlled/analyze_v4.py and
posthoc_v4.py. No LLM is called.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "analyze_v4", ROOT / "analysis" / "books-controlled" / "analyze_v4.py")
v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4)
_spec = importlib.util.spec_from_file_location(
    "posthoc_v4", ROOT / "analysis" / "books-controlled" / "posthoc_v4.py")
posthoc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(posthoc)

DEFAULT_TRACK = ROOT / "data" / "tracks" / "books-wikidata-v2"


def _interval(name: str, task_values: dict[str, float], family_of: dict, template_of: dict) -> dict | None:
    if not task_values:
        return None
    row = v4.interval(name, v4._family_mean(task_values, family_of), template_of)
    return {"estimate": row["estimate"], "family_ci95": row["family_ci95"], "families": row["families"],
            "tasks": len(task_values)}


def report(run_dir: Path, track: Path = DEFAULT_TRACK) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    families = json.loads((track / "family_manifest.json").read_text())["families"]
    family_of, template_of, nvr = {}, {}, {}
    for row in families:
        template_of[row["family_id"]] = "|".join(
            sorted({":".join(part.split(":")[:2]) for part in row["constraint_signature"].split("|")}))
        nvr[row["family_id"]] = row["no_valid_recommendation"]
        for task_id in row["task_ids"].values():
            family_of[task_id] = row["family_id"]
    tasks = {path.stem: json.loads(path.read_text()) for path in (track / "tasks").glob("*.json")}
    answers = json.loads((track / "answers.json").read_text())
    by_task = v4._trials(run_dir)
    pass1 = {task: v4._pass(traces, 1) for task, traces in by_task.items()}

    def subset(letter: str | None = None, status: str | None = None) -> dict[str, float]:
        return {task: value for task, value in pass1.items()
                if (letter is None or task.endswith(f"_{letter}"))
                and (status is None or nvr[family_of[task]] == (status == "nvr"))}

    intervals = {"all": _interval("report/all", subset(), family_of, template_of)}
    for letter, reveal in v4.REVEALS.items():
        intervals[reveal] = _interval(f"report/{reveal}", subset(letter), family_of, template_of)
    for status in ("satisfiable", "nvr"):
        intervals[f"hidden/{status}"] = _interval(f"report/hidden/{status}", subset("h", status), family_of, template_of)
    capped = {task: posthoc.capped_success(by_task[task], 3) for task in subset("h", "satisfiable")}
    intervals["hidden/satisfiable/within_3_rejections"] = _interval(
        "report/hidden/satisfiable/within_3_rejections", capped, family_of, template_of)
    pass4 = None
    if by_task and all(len(traces) >= 4 for traces in by_task.values()):
        pass4 = v4._mean(v4._pass(traces, 4) for traces in by_task.values())

    mixed = [(task, trace) for task, traces in by_task.items() if task.endswith("_m") for trace in traces]
    symbolic = manifest.get("simulator_interaction_mode") == "symbolic"
    elicitation = None
    if symbolic and mixed:
        elicitation = v4._mean(
            (lambda pair: pair[0] == pair[1])(v4._decided_on_ask(trace, tasks[task])) for task, trace in mixed)
    exposure = v4._mean(
        bool(v4._exposed(trace) & set(answers[task]["policy_eligible_solutions"]))
        for task, trace in mixed if not nvr[family_of[task]]
    )
    return {
        "run": {key: manifest.get(key) for key in (
            "model", "retrieval_mode", "catalog_size", "simulator_interaction_mode", "agent_prompt_variant")},
        "tasks": len(by_task),
        "trials": sum(len(traces) for traces in by_task.values()),
        "pass^1": intervals,
        "pass^4": pass4,
        "mixed_complete_elicitation": elicitation,
        "mixed_valid_exposure": exposure,
        "stop_classes": dict(sorted(Counter(
            trace["stop_reason"] for traces in by_task.values() for trace in traces).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, action="append")
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--output", type=Path, help="write the JSON report here as well")
    args = parser.parse_args()
    reports = {str(run_dir): report(run_dir, args.track) for run_dir in args.run_dir}
    text = json.dumps(reports, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
