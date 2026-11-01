#!/usr/bin/env python3
"""Extract the agent turns that made the symbolic user disclose a fact, for blind labelling.

Covers every ``disclose`` decision with at least one disclosed fact in the
mixed-condition traces of the primary panel (primary_remainder and reliability
stages, 20k, both interfaces, all four agents). One user reply can log more
than one decision (a candidate verdict and a disclosure), so a decision's
``turn`` field counts decisions, not dialogue turns. Each trace is therefore
replayed through the symbolic user, which reproduces its recorded decision log
exactly, and each decision is assigned to the agent message whose reply logged
it; a trace whose decisions do not replay is refused. Packets carry only a
shuffled uid and the agent's text, as in the recogniser evaluation; the key,
which holds the recogniser's reading and the trial outcome, stays in
git-ignored audit_work/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

STAGES = ("primary_remainder", "reliability")
SEED = 20261026
DEFAULT_TRACK = Path("data/tracks/books-wikidata-v2")


async def _prompting_messages(trace: dict, simulator) -> list[str]:
    """For each logged decision, the agent message whose reply logged it."""
    from tau_rec.data_model.conversation import Message, Role

    conversation, owners = [], []
    for event in trace["events"]:
        if event.get("type") != "message":
            continue
        message = Message(role=Role(event["role"]), content=event["content"])
        if message.role == Role.USER:
            before = len(simulator.decision_log)
            await simulator.respond(conversation)
            prompt = conversation[-1].content if conversation else ""
            owners += [prompt] * (len(simulator.decision_log) - before)
        conversation.append(message)
    return owners


def extract(root: Path, panel_task_ids: list[str], track: Path = DEFAULT_TRACK) -> tuple[list[dict], dict]:
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import Task
    from tau_rec.simulator.user_simulator import UserSimulator
    from tau_rec.track import load_track

    panel = {t for t in panel_task_ids if t.endswith("_m")}
    spec = load_track(track).spec
    catalog = Catalog(items=Catalog.from_json(str(track / "catalog.json"), spec).items[:20000])
    tasks: dict[str, Task] = {}
    rows = []
    for stage in STAGES:
        for path in sorted((root / stage).glob("*/*/20000/*/traces/*.json")):
            agent, interface = path.parts[-6], path.parts[-5]
            trace = json.loads(path.read_text())
            if trace["task_id"] not in panel:
                continue
            task = tasks.setdefault(trace["task_id"], Task.from_json(str(track / "tasks" / f"{trace['task_id']}.json")))
            simulator = UserSimulator("unused", task, spec, catalog, interaction_mode="symbolic")
            owners = asyncio.run(_prompting_messages(trace, simulator))
            if simulator.decision_log != trace.get("simulator_decisions", []):
                raise SystemExit(f"recorded decisions do not replay: {path}")
            success = bool(trace["evaluation"].get("primary_reward"))
            for index, decision in enumerate(trace["simulator_decisions"]):
                if decision.get("action") != "disclose" or not decision.get("disclosed_fields"):
                    continue
                rows.append((owners[index], {
                    "source": f"{path.relative_to(root)}#decision{index}",
                    "agent": agent, "interface": interface, "intent": decision.get("intent"),
                    "requested_fields": decision.get("requested_fields", []),
                    "disclosed_fields": decision.get("disclosed_fields", []), "success": success,
                }))
    random.Random(SEED).shuffle(rows)
    packets = [{"uid": uid, "text": text} for uid, (text, _) in enumerate(rows)]
    key = {str(uid): meta for uid, (_, meta) in enumerate(rows)}
    return packets, key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("out/books-wikidata-v4"))
    parser.add_argument("--plan", type=Path, default=Path("analysis/books-wikidata-v2/experiment_plan_v4.json"))
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/books-wikidata-v2/audit_work/disclosure_misreads"))
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    tasks = sorted({t for s in plan["stages"] if s["name"] in STAGES for t in s["task_ids"]})
    packets, key = extract(args.root, tasks, args.track)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "key.json").write_text(json.dumps(key, indent=1, sort_keys=True))
    for start in range(0, len(packets), args.batch_size):
        batch = packets[start:start + args.batch_size]
        (args.output_dir / f"packets_{start // args.batch_size:02d}.json").write_text(json.dumps(batch, indent=1))
    print(f"{len(packets)} disclosure turns from {len({m['source'].split('#')[0] for m in key.values()})} trials "
          f"-> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
