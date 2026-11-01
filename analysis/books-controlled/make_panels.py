#!/usr/bin/env python3
"""Freeze outcome-independent family panels for staged paid experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _rank(rows: list[dict], salt: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{salt}:{row['family_id']}".encode()
        ).hexdigest(),
    )


def _select(
    families: list[dict],
    *,
    satisfiable_per_complexity: int,
    nvr_per_complexity: int,
    salt: str,
) -> list[dict]:
    selected = []
    for complexity in ("simple", "medium", "complex"):
        cell = [row for row in families if row["complexity"] == complexity]
        satisfiable = _rank(
            [row for row in cell if not row["no_valid_recommendation"]],
            f"{salt}:{complexity}:sat",
        )
        nvr = _rank(
            [row for row in cell if row["no_valid_recommendation"]],
            f"{salt}:{complexity}:nvr",
        )
        if len(satisfiable) < satisfiable_per_complexity or len(nvr) < nvr_per_complexity:
            raise ValueError(f"not enough {complexity} families for panel {salt}")
        selected.extend(satisfiable[:satisfiable_per_complexity])
        selected.extend(nvr[:nvr_per_complexity])
    return sorted(selected, key=lambda row: row["family_id"])


def _panel(name: str, rows: list[dict], reveals: tuple[str, ...]) -> dict:
    task_ids = [
        row["task_ids"][reveal]
        for row in rows for reveal in reveals
    ]
    return {
        "name": name,
        "selection_unit": "family",
        "outcome_blind": True,
        "reveals": list(reveals),
        "family_ids": [row["family_id"] for row in rows],
        "task_ids": task_ids,
        "tasks_filter": ",".join(task_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", default="books-controlled-panels-v2")
    args = parser.parse_args()

    output = args.output or args.artifact.parent / "panels.json"
    manifest = json.loads((args.artifact / "family_manifest.json").read_text())
    families = manifest["families"]
    primary_rows = _select(
        families,
        satisfiable_per_complexity=15,
        nvr_per_complexity=2,
        salt=f"{args.seed}:primary",
    )
    reliability_rows = _select(
        primary_rows,
        satisfiable_per_complexity=5,
        nvr_per_complexity=1,
        salt=f"{args.seed}:reliability",
    )
    reliability_ids = {row["family_id"] for row in reliability_rows}
    primary_remainder_rows = [
        row for row in primary_rows if row["family_id"] not in reliability_ids
    ]
    panels = {
        "smoke": _panel(
            "smoke",
            _select(
                reliability_rows,
                satisfiable_per_complexity=2,
                nvr_per_complexity=1,
                salt=f"{args.seed}:smoke",
            ),
            ("volunteer", "mixed", "hidden"),
        ),
        "primary": _panel(
            "primary",
            primary_rows,
            ("volunteer", "mixed", "hidden"),
        ),
        "primary_remainder": _panel(
            "primary_remainder",
            primary_remainder_rows,
            ("volunteer", "mixed", "hidden"),
        ),
        "scale": _panel(
            "scale",
            [row for row in primary_rows if not row["no_valid_recommendation"]],
            ("volunteer",),
        ),
        "reliability": _panel(
            "reliability",
            reliability_rows,
            ("volunteer", "mixed", "hidden"),
        ),
    }
    payload = {
        "version": "books-controlled-panels-v1",
        "seed": args.seed,
        "family_manifest_sha256": hashlib.sha256(
            (args.artifact / "family_manifest.json").read_bytes()
        ).hexdigest(),
        "panels": panels,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for panel in panels.values():
        print(
            f"{panel['name']}: {len(panel['family_ids'])} families / "
            f"{len(panel['task_ids'])} tasks"
        )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
