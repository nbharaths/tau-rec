#!/usr/bin/env python3
"""Generate the retriever-independent books-wikidata-v2 task artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tau_rec.catalog.books_tasks import (
    DEFAULT_TASK_SEED,
    MATCHED_FAMILIES_PER_COMPLEXITY,
    MATCHED_NVR_FAMILIES_PER_COMPLEXITY,
    RETRIEVER_INDEPENDENT_SAMPLING,
    TASK_PROFILES,
    WIKIDATA_CATEGORICAL_MIN_COUNT,
    WIKIDATA_MIN_PREDICATE_GROWTH,
    WIKIDATA_V2_TASK_PROFILE,
    BookTaskGenerator,
    write_matched_artifacts,
)
from tau_rec.catalog.taxonomy import BOOK_TAXONOMY_VERSION
from tau_rec.data_model.catalog import Catalog
from tau_rec.domain import BOOKS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACK = ROOT / "data" / "tracks" / "books-wikidata-v2"
DEFAULT_CATALOG = DEFAULT_TRACK / "catalog.json"
DEFAULT_OUTPUT = DEFAULT_TRACK
DEFAULT_PROVENANCE = DEFAULT_TRACK / "source_provenance.json"
SOURCE_STATUS = "public-redistributable"
PUBLIC_PROVENANCE_FIELDS = {
    "source_name",
    "source_url",
    "source_version",
    "retrieved_at",
    "license_id",
    "license_url",
    "attribution",
    "transformation_script",
    "output_catalog_sha256",
    "redistribution_allowed",
}


def _catalog_sizes(value: str) -> list[int]:
    try:
        sizes = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("catalog sizes must be comma-separated integers") from exc
    if not sizes or sizes[0] <= 0:
        raise argparse.ArgumentTypeError("catalog sizes must contain positive integers")
    return sizes


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_TASK_SEED)
    parser.add_argument(
        "--task-profile",
        choices=TASK_PROFILES,
        default=WIKIDATA_V2_TASK_PROFILE,
        help="Metadata fields/templates used to construct tasks (default: wikidata-v2)",
    )
    parser.add_argument(
        "--track-id",
        default="books-wikidata-v2",
        help="Track identifier written into frozen answer metadata",
    )
    parser.add_argument(
        "--families-per-complexity",
        type=int,
        default=MATCHED_FAMILIES_PER_COMPLEXITY,
    )
    parser.add_argument(
        "--nvr-families-per-complexity",
        type=int,
        default=MATCHED_NVR_FAMILIES_PER_COMPLEXITY,
    )
    parser.add_argument(
        "--catalog-sizes",
        type=_catalog_sizes,
        default=_catalog_sizes("5000,10000,20000"),
        help="Nested catalog sizes for the scale experiment (default: 5000,10000,20000)",
    )
    parser.add_argument(
        "--candidate-target-per-quartile",
        type=int,
        default=36,
        help="Deterministic candidate-pool depth before family selection",
    )
    parser.add_argument(
        "--source-provenance",
        type=Path,
        default=DEFAULT_PROVENANCE,
        help="JSON provenance record; it must affirm redistribution rights",
    )
    parser.add_argument(
        "--include-retrieval-diagnostics",
        action="store_true",
        help="Run BM25 only after task selection and record its outcomes",
    )
    args = parser.parse_args()

    if not args.catalog.is_file():
        parser.error(f"catalog is missing: {args.catalog}")
    try:
        provenance = json.loads(args.source_provenance.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read source provenance: {exc}")
    if not isinstance(provenance, dict):
        parser.error("source provenance must be a JSON object")
    provenance_text = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    missing = sorted(PUBLIC_PROVENANCE_FIELDS - set(provenance))
    if missing:
        parser.error(f"public source provenance is missing fields: {missing}")
    if provenance.get("redistribution_allowed") is not True:
        parser.error("source provenance does not affirm redistribution_allowed=true")
    catalog_sha = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    if provenance.get("output_catalog_sha256") != catalog_sha:
        parser.error("source provenance output_catalog_sha256 does not match --catalog")
    catalog = Catalog.from_json(str(args.catalog), BOOKS)
    if args.catalog_sizes[-1] > len(catalog):
        parser.error(
            f"largest requested catalog size {args.catalog_sizes[-1]:,} exceeds "
            f"the {len(catalog):,}-item catalog"
        )

    core_size = args.catalog_sizes[0]
    generator = BookTaskGenerator(
        catalog,
        seed=args.seed,
        core_size=core_size,
        sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
        task_profile=args.task_profile,
    )
    drafts, generation = generator.assemble_matched(
        families_per_complexity=args.families_per_complexity,
        nvr_families_per_complexity=args.nvr_families_per_complexity,
        candidate_target_per_quartile=args.candidate_target_per_quartile,
    )
    generator.materialize_matched(
        drafts,
        include_retrieval_diagnostics=args.include_retrieval_diagnostics,
    )
    extra_manifest = {
        "catalog_sizes": args.catalog_sizes,
        "catalog_prefix_id_sha256": {
            str(size): hashlib.sha256(
                "\n".join(item.id for item in catalog.items[:size]).encode()
            ).hexdigest()
            for size in args.catalog_sizes
        },
        "core_catalog_size": core_size,
        "source_status": SOURCE_STATUS,
        "source_provenance_sha256": hashlib.sha256(provenance_text.encode()).hexdigest(),
        "selection": {
            "candidate_pool": generation["candidate_pool"],
            "families_per_complexity": generation["families_per_complexity"],
            "nvr_families_per_complexity": generation[
                "nvr_families_per_complexity"
            ],
            "solution_set_diversity": generation["solution_set_diversity"],
        },
    }
    extra_manifest.update({
        "task_profile": args.task_profile,
        "taxonomy_version": BOOK_TAXONOMY_VERSION,
        "categorical_predicate_min_catalog_count": WIKIDATA_CATEGORICAL_MIN_COUNT,
        "minimum_predicate_expansion": {
            "relative": WIKIDATA_MIN_PREDICATE_GROWTH,
            "absolute_items": 1,
            "scope": "every predicate in every satisfiable family",
        },
        "policy_format_counts": dict(sorted(Counter(
            value
            for draft in drafts
            if draft.task is not None
            for value in draft.task.user_formats
        ).items())),
    })
    from tau_rec.catalog import books_tasks as rules
    extra_manifest["v2_rules"] = {
        "overlapping_genre_pairs": sorted(sorted(pair) for pair in rules.OVERLAPPING_GENRE_PAIRS),
        "nvr_fragility": "no near-miss with a neighbor genre, a description naming the genre, or a missing country label",
        "neighbor_genres": {genre: sorted(values) for genre, values in sorted(rules.NEIGHBOR_GENRES.items())},
        "task_values": "base languages and modern countries only (no dialect, historical, or multinational labels)",
        "history_read_list": "trap plus one core decoy per hidden predicate that fails only that predicate",
        "hidden_min_selectivity": rules.HIDDEN_MIN_SELECTIVITY,
        "hidden_selectivity_by_family": dict(sorted(generator.hidden_selectivity_by_family.items())),
        "format_trap": "at least one format-incompatible answer in the smallest prefix",
        "reveal_guard": rules.REVEAL_GUARD_VERSION,
    }
    extra_manifest["reveal_guard_version"] = rules.REVEAL_GUARD_VERSION
    manifest = write_matched_artifacts(
        drafts,
        catalog,
        args.catalog,
        args.output,
        seed=args.seed,
        track_id=args.track_id,
        generator_version=generator.generator_version,
        constraint_fields=generator.constraint_fields,
        extra_manifest=extra_manifest,
    )
    lineage = {
        "method": (
            "native deterministic catalog-predicate generation with matched reveal variants"
        ),
        "source_status": SOURCE_STATUS,
        "source_provenance_sha256": manifest.get("source_provenance_sha256"),
        "catalog_path": _portable_path(args.catalog),
        "catalog_sha256": manifest["catalog_sha256"],
        "sampling_mode": manifest["sampling_mode"],
        "design_version": manifest["design_version"],
        "seed": args.seed,
        "catalog_sizes": args.catalog_sizes,
        "family_count": manifest["n_families"],
        "task_count": manifest["n_tasks"],
        "retrieval_diagnostics_recorded": args.include_retrieval_diagnostics,
    }
    (args.output / "lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "source_provenance.json").write_text(provenance_text, encoding="utf-8")
    print(
        f"Wrote {manifest['n_families']} matched families / {manifest['n_tasks']} tasks "
        f"to {args.output}"
    )
    print(f"Sampling: {manifest['sampling_mode']}; retrieval: {manifest['retrieval_role']}")
    print(f"Source status: {SOURCE_STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
