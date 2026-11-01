#!/usr/bin/env python3
"""Verify archived books traces against this code: tools, scores, and the symbolic user.

For every run directory under each ``--root`` (traces and quarantined first
attempts alike) this checks three things, without calling any LLM:

- every recorded tool call, re-executed on the same catalog prefix, retrieval
  mode, and profile, returns the recorded result (calls answered with a
  tool-budget error were never executed and are skipped);
- every trace re-scores to its stored evaluation;
- for symbolic-user runs, a fresh symbolic user fed the recorded conversation
  reproduces every recorded user message and the decision log.

    uv run python scripts/verify_books_archive.py --root out/books-wikidata-v4

The exit status is non-zero if anything differs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACK = ROOT / "data" / "tracks" / "books-wikidata-v2"
BUDGET = "Tool-call budget for this turn is used up"
STATELESS = {"search_catalog", "get_metadata", "check_availability", "check_content_preference", "recommend"}

_state: dict = {}


def _setup(track_dir: str) -> None:
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import Task
    from tau_rec.track import load_track

    track = load_track(Path(track_dir))
    _state.update(
        track=track,
        full=Catalog.from_json(str(Path(track_dir) / "catalog.json"), track.spec),
        tasks={task.id: task for task in (
            Task.from_json(str(path)) for path in sorted((Path(track_dir) / "tasks").glob("*.json")))},
        catalogs={}, searches={}, cache={},
    )


async def _replay_user(simulator, events: list[dict]) -> tuple[int, int]:
    from tau_rec.data_model.conversation import Message, Role

    conversation, same, diff = [], 0, 0
    for event in events:
        if event.get("type") != "message":
            continue
        message = Message(role=Role(event["role"]), content=event["content"])
        if message.role == Role.USER:
            reply = await simulator.respond(conversation)
            same, diff = (same + 1, diff) if reply.content == message.content else (same, diff + 1)
        conversation.append(message)
    return same, diff


def verify_run(manifest_path: str) -> tuple[Counter, list[str]]:
    """Counts of matching and differing checks for one run directory, with examples."""
    from tau_rec.catalog.search import CatalogSearch
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.conversation import ConversationTrace
    from tau_rec.environment.tools import ToolKit
    from tau_rec.evaluator.evaluator import CombinedEvaluator
    from tau_rec.simulator.user_simulator import UserSimulator

    spec = _state["track"].spec
    manifest = json.loads(Path(manifest_path).read_text())
    size, retrieval = manifest["catalog_size"], manifest["retrieval_mode"]
    profile, mode = manifest["structured_profile"], manifest.get("mode", "tools")
    catalog = _state["catalogs"].setdefault(size, Catalog(items=_state["full"].items[:size]))
    search = None
    if retrieval == "keyword":
        if size not in _state["searches"]:
            _state["searches"][size] = CatalogSearch(catalog, spec, profile=manifest["search_profile"])
        search = _state["searches"][size]
    evaluator = CombinedEvaluator(catalog, spec)
    symbolic = manifest.get("simulator_interaction_mode") == "symbolic"
    counts, problems = Counter(), []
    run_dir = Path(manifest_path).parent
    for trace_path in sorted(run_dir.glob("traces/*.json")) + sorted(run_dir.glob("quarantine/*.json")):
        text = trace_path.read_text()
        doc = json.loads(text)
        task = _state["tasks"][doc["task_id"]]
        counts["traces"] += 1
        kit = ToolKit(catalog=catalog, search=search, user_histories=task.user_history or {}, spec=spec,
                      mode=mode, retrieval_mode=retrieval, structured_profile=profile)
        for event in doc["events"]:
            if event.get("type") != "tool_call":
                continue
            if BUDGET in (event.get("result") or ""):
                counts["tool_budget_answers"] += 1
                continue
            key = (size, retrieval, profile, event["name"], json.dumps(event["arguments"], sort_keys=True),
                   None if event["name"] in STATELESS else task.id)
            if key not in _state["cache"]:
                _state["cache"][key] = kit.call(event["name"], event["arguments"])
            same = _state["cache"][key] == event["result"]
            counts["tool_calls_same" if same else "tool_calls_differ"] += 1
            if not same:
                problems.append(f"{trace_path}: tool {event['name']} {event['arguments']}")
        if "evaluation" in doc:
            result = evaluator.evaluate(task, ConversationTrace.model_validate_json(text))
            same = json.loads(json.dumps(result.model_dump(mode="json"), default=str)) == doc["evaluation"]
            counts["scores_same" if same else "scores_differ"] += 1
            if not same:
                problems.append(f"{trace_path}: score")
        if symbolic:
            simulator = UserSimulator("unused", task, spec, catalog, interaction_mode="symbolic")
            same, diff = asyncio.run(_replay_user(simulator, doc["events"]))
            counts["user_replies_same"] += same
            counts["user_replies_differ"] += diff
            if diff:
                problems.append(f"{trace_path}: {diff} user replies")
            same = simulator.decision_log == doc.get("simulator_decisions", [])
            counts["decision_logs_same" if same else "decision_logs_differ"] += 1
            if not same:
                problems.append(f"{trace_path}: decision log")
    return counts, problems[:5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifests = sorted(str(path) for root in args.root for path in root.rglob("run_manifest.json"))
    if not manifests:
        raise SystemExit("no run_manifest.json under the given roots")
    total, problems = Counter(), []
    with ProcessPoolExecutor(args.workers, initializer=_setup, initargs=(str(args.track.resolve()),)) as pool:
        for counts, found in pool.map(verify_run, manifests):
            total.update(counts)
            problems.extend(found)
    print(f"{len(manifests)} runs: {json.dumps(dict(sorted(total.items())))}")
    for problem in problems[:20]:
        print("DIFFERS", problem)
    differ = sum(count for key, count in total.items() if key.endswith("_differ"))
    print("every archived tool call, score, and symbolic-user reply reproduces" if not differ
          else f"{differ} checks differ")
    return 1 if differ else 0


if __name__ == "__main__":
    raise SystemExit(main())
