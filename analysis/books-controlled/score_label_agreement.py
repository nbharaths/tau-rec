#!/usr/bin/env python3
"""Agreement between two blind LLM labellers of the recogniser's ``final`` evaluation set (post hoc).

The ``final`` set of controller_eval_labels_v4.json (200 agent turns) was
labelled a second time by an independent blind labeller with the same rules
(controller_eval_labels_v4_second.json). This script reports Cohen's kappa on
the per-field asked/not-asked judgments (pooled and per field), raw agreement,
exact field-set agreement, and kappa on the generic flag, and re-scores the
recogniser against the second labeller's labels.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "analysis" / "books-wikidata-v2"
FIELDS = ("genres", "languages", "countries", "publication_year", "formats")


def _kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return round((observed - expected) / (1 - expected), 3)


def agreement(first: list[dict], second: list[dict]) -> dict:
    by_uid = {item["uid"]: item for item in second}
    pairs = [(item, by_uid[item["uid"]]) for item in first]
    per_field = {field: [(field in a["fields"], field in b["fields"]) for a, b in pairs] for field in FIELDS}
    pooled = [pair for values in per_field.values() for pair in values]
    return {
        "turns": len(pairs),
        "field_kappa": _kappa(pooled),
        "field_agreement": round(sum(a == b for a, b in pooled) / len(pooled), 3),
        "per_field_kappa": {field: _kappa(values) for field, values in per_field.items()},
        "exact_field_set": round(sum(set(a["fields"]) == set(b["fields"]) for a, b in pairs) / len(pairs), 3),
        "generic_kappa": _kappa([(a["generic"], b["generic"]) for a, b in pairs]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", type=Path, default=A / "controller_eval_labels_v4.json")
    parser.add_argument("--second", type=Path, default=A / "controller_eval_labels_v4_second.json")
    parser.add_argument("--output", type=Path, default=A / "controller_eval_second_labeller.json")
    args = parser.parse_args()
    first = json.loads(args.first.read_text())["sets"]["final"]["items"]
    second = json.loads(args.second.read_text())["labels"]
    spec = importlib.util.spec_from_file_location(
        "score_controller_eval_v4", Path(__file__).with_name("score_controller_eval_v4.py"))
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    texts = {item["uid"]: item["text"] for item in first}
    rescored = scorer.score([{**label, "text": texts[label["uid"]]} for label in second])
    result = {"status": "post hoc", "agreement": agreement(first, second),
              "recogniser_against_second_labeller": {k: v for k, v in rescored.items() if k != "counts"}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
