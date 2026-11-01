"""Re-score every published trace and require the leaderboard entries to match.

The board publishes numbers re-derived from traces, so the traces plus the
current evaluator must reproduce every entry's per-task {n, c} exactly. This
is the gate that lets the harness be refactored without moving a published
number: if it passes after a change, the change did not alter scoring on any
of the 5,994 archived trials.

    uv run python scripts/rederive_archive.py --archive-root /path/with/extracted/tarballs
    uv run python scripts/rederive_archive.py --fetch /tmp/tau-rec-archives

`--fetch` downloads the archives listed in leaderboard/archives.json, verifies
their sha256, and extracts them before checking. Exit status is non-zero on
any delta, any entry without an archive, or any archived run without an entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRIES = ROOT / "leaderboard" / "entries"
ARCHIVES = ROOT / "leaderboard" / "archives.json"

# Archive directory -> entry id. The archives were named for readers, the
# entries for the renderer; neither name is derivable from the other, and a
# manifest exists only for g1 runs, so the map is explicit. The check below
# fails if it stops covering every entry on disk.
RUNS = {
    "tau-rec-g0-traces": {
        "gemini-25-flash": "g0-gemini25-flash",
        "gpt54-medium-thinking": "g0-gpt54-medium",
        "gpt54-no-thinking": "g0-gpt54",
        "sonnet46-no-thinking": "g0-sonnet46",
        "ds-v4-flash": "g0-dsv4-flash",
        "ds-v4-flash-high-thinking": "g0-dsv4-flash-high",
        "ds-v4-flash-max-thinking": "g0-dsv4-flash-max",
        "gpt5-mini": "g0-gpt5-mini",
        "qwen3-32b": "g0-qwen3-32b",
    },
    "tau-rec-g1-traces": {
        "deepseek-flash": "g1-deepseek-flash",
        "glm-53-flash": "g1-glm-53-flash",
        "gpt56-luna-medium": "g1-gpt56-luna-medium",
        "mistral-small-3": "g1-mistral-small-3",
        "grok-43": "g1-grok-43",
        "minimax-m3": "g1-minimax-m3",
        "kimi-k25": "g1-kimi-k25",
        "qwen3-32b": "g1-qwen3-32b",
        "gemini-25-flash": "g1-gemini25-flash",
        "gpt5-mini": "g1-gpt5-mini",
        "dsv4-flash": "g1-dsv4-flash",
        "dsv4-flash-high": "g1-dsv4-flash-high",
        "gpt56-sol-medium": "g1-gpt56-sol-medium",
        "gpt54-medium-thinking": "g1-gpt54-medium",
        "gpt54-no-thinking": "g1-gpt54",
        "dsv4-flash-max": "g1-dsv4-flash-max",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for a in json.loads(ARCHIVES.read_text())["archives"]:
        tarball = dest / f"{a['name']}.tar.gz"
        if not tarball.exists() or sha256(tarball) != a["sha256"]:
            print(f"fetching {a['url']}")
            urllib.request.urlretrieve(a["url"], tarball)
        got = sha256(tarball)
        if got != a["sha256"]:
            sys.exit(f"{tarball.name}: sha256 {got} != pinned {a['sha256']}")
        if not (dest / a["name"]).is_dir():
            with tarfile.open(tarball) as tf:
                tf.extractall(dest, filter="data")
        print(f"  {a['name']}: ok")


def find_archive_dirs(root: Path) -> dict[str, Path]:
    found = {}
    for name in RUNS:
        hits = [p for p in root.rglob(name) if p.is_dir()]
        if len(hits) != 1:
            sys.exit(f"expected exactly one {name}/ under {root}, found {len(hits)}")
        found[name] = hits[0]
    return found


def rederive(traces_dir: Path, tasks, evaluator) -> dict[str, dict[str, int]]:
    from tau_rec.data_model.conversation import ConversationTrace

    counts: dict[str, dict[str, int]] = {}
    for path in sorted(traces_dir.glob("*.json")):
        trace = ConversationTrace.model_validate_json(path.read_text())
        result = evaluator.evaluate(task=tasks[trace.task_id], trace=trace)
        row = counts.setdefault(trace.task_id, {"n": 0, "c": 0})
        row["n"] += 1
        row["c"] += 1 if result.primary_reward == 1.0 else 0
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--archive-root", type=Path, help="directory containing the extracted archives")
    g.add_argument("--fetch", type=Path, metavar="DIR", help="download, verify and extract into DIR, then check")
    args = ap.parse_args()

    root = args.fetch if args.fetch else args.archive_root
    if args.fetch:
        fetch(root)
    archive_dirs = find_archive_dirs(root)

    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import Task
    from tau_rec.evaluator.evaluator import CombinedEvaluator

    catalog = Catalog.from_json(str(ROOT / "data" / "catalog.json"))
    tasks = {p.stem: Task.from_json(str(p)) for p in sorted((ROOT / "data" / "tasks").glob("*.json"))}
    evaluator = CombinedEvaluator(catalog)

    on_disk = {p.stem for p in ENTRIES.glob("*.json")}
    mapped = {eid for runs in RUNS.values() for eid in runs.values()}
    problems: list[str] = []
    if on_disk - mapped:
        problems.append(f"entries with no archive mapping: {sorted(on_disk - mapped)}")
    if mapped - on_disk:
        problems.append(f"archive mappings with no entry: {sorted(mapped - on_disk)}")

    print(f"{'entry':<26}{'traces':>7}{'solved':>8}  deltas")
    print("-" * 60)
    total_traces = 0
    for archive, runs in RUNS.items():
        for run_dir, entry_id in runs.items():
            traces_dir = archive_dirs[archive] / run_dir / "traces"
            if not traces_dir.is_dir():
                problems.append(f"{entry_id}: no traces at {traces_dir}")
                continue
            entry_path = ENTRIES / f"{entry_id}.json"
            if not entry_path.exists():
                continue  # already reported above
            published = json.loads(entry_path.read_text())["per_task"]
            fresh = rederive(traces_dir, tasks, evaluator)
            total_traces += sum(r["n"] for r in fresh.values())
            deltas = [
                f"{tid}: published n={published.get(tid, {}).get('n')} c={published.get(tid, {}).get('c')}"
                f" rederived n={fresh.get(tid, {}).get('n')} c={fresh.get(tid, {}).get('c')}"
                for tid in sorted(set(published) | set(fresh))
                if published.get(tid) != fresh.get(tid)
            ]
            if deltas:
                problems.append(f"{entry_id}: {len(deltas)} task(s) differ")
            print(f"{entry_id:<26}{sum(r['n'] for r in fresh.values()):>7}"
                  f"{sum(r['c'] for r in fresh.values()):>8}  {'; '.join(deltas) or 'none'}")

    print(f"\n{total_traces} traces re-scored across {len(mapped)} entries.")
    if problems:
        print("FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("every published entry reproduces exactly from its archived traces.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
