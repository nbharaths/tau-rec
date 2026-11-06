#!/usr/bin/env python3
"""Report catalog years that disagree with the work's own Wikidata description.

Report-only: lists each disagreement and the families whose answer set would
change under the description's year, so errata candidates surface before a
rebuild. Each flag still needs a bibliographic source; the description can be
the wrong side (The Thousandth Floor).

    uv run python scripts/report_description_years.py \
      --track data/tracks/books-wikidata-v2 [--only-task-changes] [--json out.json]

Exits 0 unless ``--fail-on-task-changes`` is given and some unreviewed item
changes a family; items listed in ``--reviewed`` (one Wikidata ID per line,
``#`` comments allowed) are reported but never fail the run.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tau_rec.catalog.description_years import year_mismatches
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.domain import BOOKS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--only-task-changes", action="store_true", help="list only items that change a family")
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    parser.add_argument("--reviewed", type=Path, help="Wikidata IDs already checked against a source")
    parser.add_argument("--fail-on-task-changes", action="store_true")
    args = parser.parse_args()

    catalog = Catalog.from_json(str(args.track / "catalog.json"), BOOKS)
    tasks = {path.stem: Task.from_json(str(path)) for path in sorted((args.track / "tasks").glob("*.json"))}
    families = json.loads((args.track / "family_manifest.json").read_text())["families"]
    reviewed: set[str] = set()
    if args.reviewed:
        for line in args.reviewed.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                reviewed.add(line if line.startswith("wd_") else f"wd_{line}")

    rows = year_mismatches(catalog.items, tasks, families)
    changing = [row for row in rows if row.family_changes]
    shown = changing if args.only_task_changes else rows
    for row in shown:
        changes = ", ".join(f"{fid.removeprefix('books_')} {sign}" for fid, sign in sorted(row.family_changes.items()))
        mark = " (reviewed)" if row.item_id in reviewed else ""
        print(f"{row.item_id}{mark}: {row.title!r} catalog {row.catalog_year}, description {row.description_year}"
              f" ({row.description!r}){'; families: ' + changes if changes else ''}")
    print(f"\n{len(rows)} items disagree with their description year; {len(changing)} of them change a family.")
    if args.json:
        args.json.write_text(json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False) + "\n")

    unreviewed = [row for row in changing if row.item_id not in reviewed]
    if args.fail_on_task_changes and unreviewed:
        print(f"{len(unreviewed)} unreviewed items change a family.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
