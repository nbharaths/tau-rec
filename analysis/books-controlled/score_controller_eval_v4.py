#!/usr/bin/env python3
"""Score the books symbolic user's question recogniser against blind labels.

Reads analysis/books-wikidata-v2/controller_eval_labels_v4.json: the held-out
``final`` set of agent turns from development runs with the symbolic user,
labelled blind by an LLM that never saw a recogniser output,
after the recogniser was frozen.  It is the reported estimate. Each turn is classified as the live
user classifies it: candidate-marker sentences dropped. Field precision and
recall are micro-averaged over genres, languages, countries, publication year
and format; generic is the open-request flag ("N/A" when a stratum has no
positives in either the labels or the predictions).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tau_rec.simulator.user_simulator import BOOKS_CONTROLLER_VERSION, UserSimulator, classify_book_question

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "analysis" / "books-wikidata-v2" / "controller_eval_labels_v4.json"
FIELDS = ("genres", "languages", "countries", "publication_year", "formats")


def _ratio(numerator: int, denominator: int) -> float | str:
    return round(numerator / denominator, 3) if denominator else "N/A"


def score(items: list[dict]) -> dict:
    counts = Counter()
    for item in items:
        intent = classify_book_question(UserSimulator._without_candidate_sentences(item["text"]))
        predicted = set(intent.fields) & set(FIELDS)
        gold = set(item["fields"])
        counts["tp"] += len(predicted & gold)
        counts["fp"] += len(predicted - gold)
        counts["fn"] += len(gold - predicted)
        counts["exact"] += predicted == gold
        counts["generic_tp"] += intent.generic and item["generic"]
        counts["generic_fp"] += intent.generic and not item["generic"]
        counts["generic_fn"] += item["generic"] and not intent.generic
    return {
        "turns": len(items),
        "field_precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
        "field_recall": _ratio(counts["tp"], counts["tp"] + counts["fn"]),
        "exact_field_set": _ratio(counts["exact"], len(items)),
        "generic_precision": _ratio(counts["generic_tp"], counts["generic_tp"] + counts["generic_fp"]),
        "generic_recall": _ratio(counts["generic_tp"], counts["generic_tp"] + counts["generic_fn"]),
        "counts": dict(sorted(counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.labels.read_text())
    report = {"controller_version": BOOKS_CONTROLLER_VERSION, "sets": {}}
    for name, block in payload["sets"].items():
        strata: dict[str, list[dict]] = defaultdict(list)
        for item in block["items"]:
            strata[item["stratum"]].append(item)
        natural = [item for item in block["items"] if item["stratum"] != "changed"]
        report["sets"][name] = {
            "all_model_strata": score(natural),
            **{stratum: score(items) for stratum, items in sorted(strata.items())},
        }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
