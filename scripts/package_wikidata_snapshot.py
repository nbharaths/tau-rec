#!/usr/bin/env python3
"""Package the hash-pinned Wikidata query responses as a release asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau_rec.catalog.wikidata_snapshot import build_source_snapshot


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACK = ROOT / "data" / "tracks" / "books-wikidata-v2"
DEFAULT_CACHE = ROOT / ".cache" / "wikidata-books" / ".cache"
DEFAULT_OUTPUT = ROOT / "dist" / "books-wikidata-v2-source-snapshot.tar.gz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--record",
        type=Path,
        help="optional JSON path for the release-asset identity record",
    )
    args = parser.parse_args()

    record = build_source_snapshot(args.track, args.cache, args.output)
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )

    print(f"wrote {record['query_response_file_count']} responses to {args.output}")
    print(f"archive bytes: {record['archive_bytes']:,}")
    print(f"archive sha256: {record['archive_sha256']}")
    if args.record:
        print(f"identity record: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
