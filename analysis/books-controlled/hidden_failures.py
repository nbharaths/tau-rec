#!/usr/bin/env python3
"""How hidden satisfiable trials on the primary panel end (post hoc).

Each trial is classified by its final recommendation: a success; no
recommendation; a book the agent never proposed for a verdict (untested); a
book the user had rejected; or a book the user had accepted that still failed
(a protocol check). Counts are per agent and pooled over both interfaces.
Untested commitments are split by whether an earlier agent turn named the book
without getting a verdict (no marker, or several), and trials without a
recommendation by how they stopped.  Untested commitments are also split by
what the user had judged before them (nothing, only accepted books, at least one
rejection); trials by whether they put any book to the user; and failures by
whether an answered turn named, without a tag, a book the user would have
accepted; and trials by whether the user ever declined a field question, which
signals a hidden requirement.  With --scripted, every panel hidden task
(satisfiable and NVR) is also counted by who ever solved it: the scripted
agent, only some agent trial, or nobody.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

STAGES = ("primary_remainder", "reliability")


def classify(trace: dict) -> str:
    if trace["evaluation"]["primary_reward"] >= 1 - 1e-9:
        return "success"
    recommended = [e for e in trace["events"] if e.get("type") == "tool_call" and e.get("name") == "recommend"]
    final = recommended[-1]["arguments"].get("item_id") if recommended else None
    if not final:
        return "no_recommendation"
    verdicts = {d.get("candidate_id"): d.get("verdict") for d in trace.get("simulator_decisions", [])
                if d.get("action") == "candidate_verdict"}
    return {None: "untested", "rejected": "rejected", "accepted": "accepted"}.get(verdicts.get(final), "untested")


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _answered_agent_turns(trace: dict) -> list[str]:
    """Agent messages the user answered: those before the last user message ahead of the recommend call, if any."""
    events = trace["events"]
    commit = max((i for i, e in enumerate(events) if e.get("type") == "tool_call" and e.get("name") == "recommend"),
                 default=len(events))
    users = [i for i, e in enumerate(events[:commit]) if e.get("type") == "message" and e.get("role") == "user"]
    return [event.get("content") or "" for event in events[:users[-1] if users else 0]
            if event.get("type") == "message" and event.get("role") == "agent"]


def _names(text: str, item: str, titles: dict[str, str]) -> bool:
    """A turn names a book by its identifier or its title (titles under four characters only by identifier)."""
    title = _words(titles.get(item, ""))
    return bool(re.search(rf"(?<!\w){re.escape(item)}(?!\w)", text)) or (
        len(title) >= 4 and f" {title} " in f" {_words(text)} ")


def named_earlier(trace: dict, titles: dict[str, str]) -> bool:
    """Whether an agent turn before the committing one names the recommended book.

    The committing turn is everything after the last user message before the
    recommend call, so the user never answered it.  A title shorter than four
    characters counts only through the identifier, since short words recur in
    ordinary questions.
    """
    recommended = [e for e in trace["events"] if e.get("type") == "tool_call" and e.get("name") == "recommend"]
    item = recommended[-1]["arguments"]["item_id"]
    return any(_names(text, item, titles) for text in _answered_agent_turns(trace))


def commitment_context(trace: dict) -> str:
    """What the user had judged before the commit: nothing, only accepted books, or at least one rejected book."""
    verdicts = [d.get("verdict") for d in trace.get("simulator_decisions", []) if d.get("action") == "candidate_verdict"]
    return "no_proposal" if not verdicts else "after_rejection" if "rejected" in verdicts else "no_rejection"


def refusal_received(trace: dict) -> bool:
    """Whether the user ever declined a field question, which signals that a requirement is hidden."""
    return any(d.get("withheld_fields") for d in trace.get("simulator_decisions", []))


def untagged_acceptable(trace: dict, acceptable: set[str], titles: dict[str, str]) -> bool:
    """Whether an answered agent turn names, without getting a verdict, a book the user would have accepted.

    ``acceptable`` holds the task's solutions the user has not read; a format
    the user names later could still rule one out, so the count is an upper bound.
    """
    judged = {d.get("candidate_id") for d in trace.get("simulator_decisions", []) if d.get("action") == "candidate_verdict"}
    turns = _answered_agent_turns(trace)
    return any(_names(text, item, titles) for item in acceptable - judged for text in turns)


def feasibility(trials: dict[str, list[bool]], scripted: dict[str, bool], nvr: dict[str, bool]) -> dict:
    """Hidden tasks per kind: solved by the scripted agent, else by some agent trial, else unsolved."""
    out = {kind: {"tasks": 0, "scripted": 0, "agents_only": 0, "unsolved": 0} for kind in ("satisfiable", "nvr")}
    for task, successes in trials.items():
        row = out["nvr" if nvr[task] else "satisfiable"]
        row["tasks"] += 1
        row["scripted" if scripted[task] else "agents_only" if any(successes) else "unsolved"] += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scripted", type=Path, help="run_scripted_reference.py output")
    args = parser.parse_args()
    families = json.loads((args.artifact / "family_manifest.json").read_text())["families"]
    nvr = {task: row["no_valid_recommendation"] for row in families for task in row["task_ids"].values()}
    plan = json.loads(args.plan.read_text())
    panel = {t for stage in plan["stages"] if stage["name"] in STAGES for t in stage["task_ids"]}
    titles = {row["id"]: row["title"] for row in json.loads((args.artifact / "catalog.json").read_text())}
    answers = json.loads((args.artifact / "answers.json").read_text())
    acceptable = {task: set(row["constraint_solutions"]) - set(row["read_ids"]) for task, row in answers.items()}
    by_agent: dict[str, Counter] = {}
    untested: dict[str, Counter] = {}
    contexts: dict[str, Counter] = {}
    proposals: dict[str, Counter] = {}
    mentions: dict[str, Counter] = {}
    refusals: dict[str, Counter] = {}
    stops: Counter = Counter()
    hidden_trials: dict[str, list[bool]] = {}
    for stage in STAGES:
        for path in sorted((args.root / stage).glob("*/*/20000/*/traces/*.json")):
            trace = json.loads(path.read_text())
            task = trace["task_id"]
            if task.endswith("_h") and task in panel:
                hidden_trials.setdefault(task, []).append(trace["evaluation"]["primary_reward"] >= 1 - 1e-9)
            if task.endswith("_h") and not nvr[task] and task in panel:
                agent = path.parts[-6]
                kind = classify(trace)
                by_agent.setdefault(agent, Counter())[kind] += 1
                proposed = commitment_context(trace) != "no_proposal"
                proposals.setdefault(agent, Counter()).update(
                    {"trials": 1, "with_proposal": proposed,
                     "successes_with_proposal": proposed and kind == "success",
                     "successes_without_proposal": not proposed and kind == "success"})
                refusals.setdefault(agent, Counter())[
                    f"{'refused' if refusal_received(trace) else 'never_refused'}_{kind}"] += 1
                if kind != "success":
                    mentions.setdefault(agent, Counter()).update(
                        {"failures": 1, "untagged_acceptable": untagged_acceptable(trace, acceptable[task], titles)})
                if kind == "untested":
                    named = "named_earlier" if named_earlier(trace, titles) else "not_named_earlier"
                    untested.setdefault(agent, Counter())[named] += 1
                    contexts.setdefault(agent, Counter())[commitment_context(trace)] += 1
                elif kind == "no_recommendation":
                    stops[trace.get("stop_reason") or "unknown"] += 1
    pooled = sum(by_agent.values(), Counter())
    failures = sum(n for kind, n in pooled.items() if kind != "success")
    out = {"status": "post hoc", "by_agent": {a: dict(sorted(c.items())) for a, c in sorted(by_agent.items())},
           "pooled": dict(sorted(pooled.items())), "failures": failures,
           "share_of_failures": {kind: round(n / failures, 4) for kind, n in sorted(pooled.items()) if kind != "success"},
           "untested_by_agent": {a: dict(sorted(c.items())) for a, c in sorted(untested.items())},
           "untested_pooled": dict(sorted(sum(untested.values(), Counter()).items())),
           "no_recommendation_stops": dict(sorted(stops.items())),
           "untested_context_by_agent": {a: dict(sorted(c.items())) for a, c in sorted(contexts.items())},
           "untested_context_pooled": dict(sorted(sum(contexts.values(), Counter()).items())),
           "proposals_by_agent": {a: dict(sorted(c.items())) for a, c in sorted(proposals.items())},
           "proposals_pooled": dict(sorted(sum(proposals.values(), Counter()).items())),
           "untagged_acceptable_by_agent": {a: dict(sorted(c.items())) for a, c in sorted(mentions.items())},
           "untagged_acceptable_pooled": dict(sorted(sum(mentions.values(), Counter()).items())),
           "refusal_by_agent": {a: dict(sorted(c.items())) for a, c in sorted(refusals.items())},
           "refusal_pooled": dict(sorted(sum(refusals.values(), Counter()).items()))}
    if args.scripted:
        scripted = {row["task_id"]: row["success"] for row in json.loads(args.scripted.read_text())["rows"]}
        out["feasibility"] = feasibility(hidden_trials, scripted, nvr)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
