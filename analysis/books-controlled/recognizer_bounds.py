#!/usr/bin/env python3
"""A label-free bound on questions the recognizer missed in mixed trials (post hoc).

A missed question can only lower mixed success, and only in a failed trial that
left an askable fact undisclosed and contains a question the recognizer mapped
to no field and no generic request.  Each failed mixed trial on the primary
panel (20k, both interfaces, NVR families included, as in the misread bound) is
classified as: every askable fact disclosed; no question at all; a question the
recognizer mapped to nothing; or only recognized questions.  Counting every
trial of the third kind as a success gives the largest gain recognizer misses of
whole questions could explain.  A question it maps to the wrong field or to
only some of the fields asked about counts as recognized, so partial misses
fall outside the bound.

Questions are sentences with a question mark outside candidate proposals, in
agent turns the user answered, after the harness's seeded greeting.  No LLM or
label is used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tau_rec.simulator.user_simulator import classify_book_question

_spec = importlib.util.spec_from_file_location("hidden_failures", Path(__file__).with_name("hidden_failures.py"))
_failures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_failures)

STAGES = ("primary_remainder", "reliability")
KINDS = ("all_disclosed", "no_question", "recognized_only", "unrecognized_question")


def question_sentences(text: str) -> list[str]:
    """Sentences with a question mark, outside candidate proposals."""
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text)
            if "?" in s and "[[candidate_id:" not in s]


def _turns(trace: dict) -> list[str]:
    """Answered agent turns after the harness's seeded greeting (the first agent message)."""
    turns = _failures._answered_agent_turns(trace)
    seeded = bool(trace["events"]) and trace["events"][0].get("role") == "agent"
    return turns[1 if seeded else 0:]


def _questions(trace: dict) -> list[str]:
    return [s for turn in _turns(trace) for s in question_sentences(turn)]


def _undisclosed_fields(trace: dict, task: dict) -> set[str]:
    disclosed = {i for d in trace.get("simulator_decisions", []) for i in d.get("disclosed_constraint_indices", [])}
    return {c["constraint"]["field"] for i, c in enumerate(task["constraints"])
            if c["reveal"] == "on_ask" and i not in disclosed}


def labelled_miss(trace: dict, task: dict, labels: dict[str, dict]) -> bool:
    """Whether an answered question asks, by its label, about a field whose askable fact stayed undisclosed.

    A field left undisclosed was never recognized in any turn, so such a question
    was missed.  A labelled generic request counts when the recognizer saw no
    generic request in its turn (a recognized one releases a fact by design).
    """
    missing = _undisclosed_fields(trace, task)
    if not missing:
        return False
    for turn in _turns(trace):
        turn_generic = classify_book_question(turn).generic
        for sentence in question_sentences(turn):
            label = labels[sentence]
            if missing & set(label["fields"]) or (label["generic"] and not turn_generic):
                return True
    return False


def classify_mixed_failure(trace: dict, task: dict) -> str:
    askable = {i for i, c in enumerate(task["constraints"]) if c["reveal"] == "on_ask"}
    disclosed = {i for d in trace.get("simulator_decisions", []) for i in d.get("disclosed_constraint_indices", [])}
    if askable <= disclosed:
        return "all_disclosed"
    questions = _questions(trace)
    if not questions:
        return "no_question"
    if any(classify_book_question(s).kind == "none" for s in questions):
        return "unrecognized_question"
    return "recognized_only"


def missed_question_bound(trials: list[tuple[dict, dict]], labels: dict[str, dict] | None = None) -> dict:
    """Failed mixed trials by kind, and the most success missed questions could add.

    Without labels, every failure holding a question the recognizer mapped to
    nothing counts; with labels, every failure holding a labelled miss does.
    """
    kinds = Counter({kind: 0 for kind in KINDS})
    failures = misses = 0
    for trace, task in trials:
        if trace["evaluation"]["primary_reward"] >= 1 - 1e-9:
            continue
        failures += 1
        kind = classify_mixed_failure(trace, task)
        kinds[kind] += 1
        if labels is not None and kind in ("recognized_only", "unrecognized_question"):
            misses += labelled_miss(trace, task, labels)
    out = {"trials": len(trials), "failures": failures, **dict(kinds),
           "max_gain": kinds["unrecognized_question"] / len(trials) if trials else 0.0}
    if labels is not None:
        out |= {"labelled_misses": misses, "max_gain_labelled": misses / len(trials) if trials else 0.0}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="LLM labels of the question sentences (missed_question_labels.json)")
    args = parser.parse_args()
    labels = ({row["sentence"]: row for row in json.loads(args.labels.read_text())["labels"]}
              if args.labels else None)
    plan = json.loads(args.plan.read_text())
    panel = {t for stage in plan["stages"] if stage["name"] in STAGES for t in stage["task_ids"]}
    tasks = {path.stem: json.loads(path.read_text()) for path in (args.artifact / "tasks").glob("*.json")}
    mixed: dict[str, list] = defaultdict(list)
    for stage in STAGES:
        for path in sorted((args.root / stage).glob("*/*/20000/*/traces/*.json")):
            trace = json.loads(path.read_text())
            task = trace["task_id"]
            if task not in panel:
                continue
            cell = f"{path.parts[-6]}/{path.parts[-5]}"
            if task.endswith("_m"):
                mixed[cell].append((trace, tasks[task]))
    by_cell = {cell: missed_question_bound(trials, labels) for cell, trials in sorted(mixed.items())}
    gains = ("max_gain", "max_gain_labelled") if labels is not None else ("max_gain",)
    by_agent: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cell, row in by_cell.items():
        for gain in gains:
            by_agent[cell.split("/")[0]][gain].append(row[gain])
    agent_gain = {agent: {gain: sum(values) / len(values) for gain, values in rows.items()}
                  for agent, rows in sorted(by_agent.items())}
    out = {
        "status": "post hoc",
        "by_cell": by_cell,
        "by_agent": agent_gain,
        "model_averaged": {gain: sum(row[gain] for row in agent_gain.values()) / len(agent_gain) for gain in gains},
        "pooled_failures": {k: sum(row[k] for row in by_cell.values())
                            for k in (*KINDS, "failures", *(("labelled_misses",) if labels is not None else ()))},
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "by_cell"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
