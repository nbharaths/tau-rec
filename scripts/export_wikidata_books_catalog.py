#!/usr/bin/env python3
"""Freeze a public CC0 books catalog from the audited Wikidata slice.

Candidate enumeration and factual fields come from the same cached SPARQL
queries as ``audit_wikidata_books.py``.  The resulting catalog is a frozen,
redistributable artifact; the live endpoint is not consulted at benchmark run
time.  Format availability is deterministic synthetic environment state and
is labeled as such in every row and in the manifest. Sponsorship is disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from audit_wikidata_books import (
    DEFAULT_ENDPOINT,
    DEFAULT_OUTPUT as AUDIT_OUTPUT,
    RELATIONS,
    SEED,
    CachedHttp,
    QLeverClient,
    build_profiles,
    ids_query,
    relation_query,
)
from tau_rec.catalog.wikidata_books import (
    WIKIDATA_GENRE_MAP,
    WIKIDATA_GENRE_MAP_V2,
    WIKIDATA_LICENSE_ID,
    WIKIDATA_LICENSE_URL,
    booklike_reason,
    apply_label_overrides,
    canonical_genres,
    dedupe_key_v2,
    exclusion_reason,
    deduplicate_profiles,
    qid_from_value,
    sparql_literal,
    stable_order_key,
)
from tau_rec.catalog.taxonomy import BOOK_TAXONOMY_V2_VERSION, derived_labels_v2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACK = ROOT / "data" / "tracks" / "books-wikidata-v2"
FORMATS = ("audiobook", "ebook", "hardcover", "paperback")
SYNTHETIC_STATE_VERSION = "book-access-state-v1"


def linked_label_query(predicate: str) -> str:
    from audit_wikidata_books import PREFIXES, _candidate_subquery

    return PREFIXES + f"""
SELECT DISTINCT ?value ?label WHERE {{
  {{ {_candidate_subquery()} }}
  ?item {predicate} ?value.
  ?value rdfs:label ?label.
  FILTER(LANG(?label) = "en")
}}
"""


def sitelink_query() -> str:
    from audit_wikidata_books import PREFIXES, _candidate_subquery

    return PREFIXES + f"""
SELECT ?item (COUNT(DISTINCT ?article) AS ?value) WHERE {{
  {{ {_candidate_subquery()} }}
  OPTIONAL {{ ?article schema:about ?item. }}
}}
GROUP BY ?item
"""


def label_map(rows: list[dict[str, str]]) -> dict[str, str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        qid = qid_from_value(row.get("value", ""))
        if qid:
            grouped[qid].add(sparql_literal(row.get("label", "")))
    return {
        qid: min(labels, key=lambda label: (len(label), label))
        for qid, labels in grouped.items() if labels
    }


def synthetic_formats(item_id: str) -> list[str]:
    digest = hashlib.sha256(f"{SYNTHETIC_STATE_VERSION}:formats:{item_id}".encode()).digest()
    thresholds = (36, 132, 196, 238)  # ~14%, 52%, 77%, 93%
    selected = [name for name, value, limit in zip(FORMATS, digest, thresholds) if value < limit]
    if not selected:
        selected = [FORMATS[digest[4] % len(FORMATS)]]
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--audit-cache", type=Path, default=AUDIT_OUTPUT / ".cache")
    parser.add_argument("--output-track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--catalog-size", type=int, default=20_000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--request-delay", type=float, default=0.8)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--catalog-profile", choices=("v2",), default="v2",
                        help="catalog rules version")
    parser.add_argument("--label-overrides", type=Path,
                        help="reviewed corrections of clear source-label errors")
    args = parser.parse_args()
    genre_map = WIKIDATA_GENRE_MAP_V2
    overrides = {}
    if args.label_overrides:
        overrides = json.loads(args.label_overrides.read_text())

    http = CachedHttp(
        args.audit_cache,
        offline=args.offline,
        refresh=args.refresh,
        timeout=args.timeout,
        request_delay=args.request_delay,
    )
    client = QLeverClient(args.endpoint, http)
    candidate_rows, _ = client.query("candidate-ids", ids_query())
    candidate_ids = sorted(
        {qid for row in candidate_rows if (qid := qid_from_value(row.get("item", "")))},
        key=lambda qid: int(qid[1:]),
    )
    relation_rows = {
        name: client.query(name, relation_query(predicate, suffix))[0]
        for name, (predicate, suffix) in RELATIONS.items()
    }
    profiles, diagnostics = build_profiles(candidate_ids, relation_rows, genre_map=genre_map)
    profiles = [row for row in profiles if booklike_reason(row) is not None]
    profiles, dedupe = deduplicate_profiles(profiles, key=dedupe_key_v2)

    label_queries = {
        "authors": "wdt:P50",
        "languages": "wdt:P407",
        "countries": "wdt:P495",
        "publishers": "wdt:P123",
    }
    labels = {
        name: label_map(client.query(f"labels-{name}", linked_label_query(predicate))[0])
        for name, predicate in label_queries.items()
    }
    sitelink_rows, _ = client.query("sitelinks", sitelink_query())
    sitelinks = {
        qid: int(sparql_literal(row["value"]))
        for row in sitelink_rows
        if (qid := qid_from_value(row.get("item", "")))
    }

    eligible = []
    rejected: dict[str, int] = defaultdict(int)
    for row in profiles:
        author_names = [labels["authors"].get(qid) for qid in row.author_ids]
        language_names = [labels["languages"].get(qid) for qid in row.language_ids]
        if any(name is None for name in author_names):
            rejected["missing_author_label"] += 1
            continue
        if any(name is None for name in language_names):
            rejected["missing_language_label"] += 1
            continue
        fields = {
            "title": row.title,
            "description": row.description,
            "genres": list(canonical_genres(row.genre_ids, genre_map=genre_map)),
            "languages": list(language_names),
            "countries": sorted(
                {labels["countries"][qid] for qid in row.country_ids if qid in labels["countries"]}
            ),
            "publication_year": row.year,
        }
        if overrides:
            reason = exclusion_reason(row.item_id, overrides, title=row.title, description=row.description)
            fixed = apply_label_overrides(
                row.item_id, fields, overrides, title=row.title, description=row.description,
            )
            if fixed is None:
                rejected[reason or "override_removed_all_genres"] += 1
                continue
            fields = fixed
        eligible.append((row, author_names, fields))
    eligible.sort(key=lambda value: stable_order_key(value[0].item_id, SEED))
    if len(eligible) < args.catalog_size:
        parser.error(
            f"only {len(eligible):,} labeled book-like works; need {args.catalog_size:,}"
        )

    catalog = []
    source_rows = []
    for row, author_names, fields in eligible[: args.catalog_size]:
        publisher_names = sorted(
            {labels["publishers"][qid] for qid in row.publisher_ids if qid in labels["publishers"]}
        )
        formats = synthetic_formats(row.item_id)
        item = {
            "id": f"wd_{row.item_id}",
            "title": fields["title"],
            "authors": list(author_names),
            "publication_year": fields["publication_year"],
            "pages": row.pages,
            "genres": fields["genres"],
            "description": fields["description"],
            "publisher": "; ".join(publisher_names),
            "rating": None,
            "vote_count": None,
            "formats": formats,
            "sponsored": False,
            "content_rating": "General",
            "languages": derived_labels_v2("languages", fields["languages"]),
            "countries": derived_labels_v2("countries", fields["countries"]),
            "sitelink_count": sitelinks.get(row.item_id, 0),
            "source_url": f"https://www.wikidata.org/wiki/{row.item_id}",
            "formats_source": SYNTHETIC_STATE_VERSION,
        }
        item["source_languages"] = fields["languages"]
        item["source_countries"] = fields["countries"]
        catalog.append(item)
        source_rows.append(
            {
                "wikidata_id": row.item_id,
                "author_ids": list(row.author_ids),
                "genre_ids": list(row.genre_ids),
                "language_ids": list(row.language_ids),
                "country_ids": list(row.country_ids),
                "publisher_ids": list(row.publisher_ids),
                "instance_ids": list(row.instance_ids),
                "form_ids": list(row.form_ids),
                "booklike_reason": booklike_reason(row),
            }
        )

    args.output_track.mkdir(parents=True, exist_ok=True)
    catalog_text = json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    catalog_path = args.output_track / "catalog.json"
    catalog_path.write_text(catalog_text, encoding="utf-8")
    source_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in source_rows)
    source_path = args.output_track / "source_selection.jsonl"
    source_path.write_text(source_text, encoding="utf-8")

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "wikidata-books-catalog-v1",
        "generated_at": generated_at,
        "catalog_size": len(catalog),
        "catalog_sha256": hashlib.sha256(catalog_text.encode()).hexdigest(),
        "source_selection_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "selection_seed": SEED,
        "license": {"id": WIKIDATA_LICENSE_ID, "url": WIKIDATA_LICENSE_URL},
        "source": {
            "name": "Wikidata",
            "query_endpoint": args.endpoint,
            "retrieval": "frozen query result transformed into a committed catalog",
            "query_response_records": http.records,
        },
        "synthetic_environment_state": {
            "version": SYNTHETIC_STATE_VERSION,
            "fields": ["formats"],
            "warning": "This field is benchmark state, not a factual claim about a work.",
        },
        "missing_source_fields": {
            "rating": "null; no comparable Wikidata field",
            "vote_count": "null; sitelink_count is stored separately and is not called votes",
        },
        "selection_diagnostics": {"profile": diagnostics, "dedupe": dedupe, "rejected": dict(rejected)},
    }
    manifest["catalog_profile"] = {
        "version": "books-wikidata-catalog-v2",
        "dropped_genre_rules": sorted(set(WIKIDATA_GENRE_MAP) - set(WIKIDATA_GENRE_MAP_V2)),
        "dedupe": "normalized title without part/volume suffix + author set",
        "derived_parent_labels": BOOK_TAXONOMY_V2_VERSION,
        "label_overrides_sha256": (
            hashlib.sha256(args.label_overrides.read_bytes()).hexdigest() if args.label_overrides else None
        ),
        "label_overrides_applied": sorted(
            set(overrides.get("overrides", {})) & {row.item_id for row, _, _ in eligible[: args.catalog_size]}
        ),
    }
    (args.output_track / "catalog_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "source_name": "Wikidata",
        "source_url": "https://www.wikidata.org/",
        "source_version": f"frozen-query-snapshot-{generated_at[:10]}",
        "retrieved_at": generated_at,
        "license_id": WIKIDATA_LICENSE_ID,
        "license_url": WIKIDATA_LICENSE_URL,
        "attribution": "Wikidata contributors",
        "query_transport": args.endpoint,
        "transformation_script": "scripts/export_wikidata_books_catalog.py",
        "output_catalog_sha256": manifest["catalog_sha256"],
        "source_selection_sha256": manifest["source_selection_sha256"],
        "redistribution_allowed": True,
        "reproducibility_note": (
            "The committed catalog is the frozen artifact. The rolling query endpoint "
            "may change; catalog_manifest.json pins every query response hash."
        ),
    }
    (args.output_track / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(catalog):,} CC0 catalog records to {catalog_path}")
    print(f"catalog sha256: {manifest['catalog_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
