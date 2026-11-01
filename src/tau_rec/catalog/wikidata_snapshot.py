"""Build a deterministic release bundle for frozen Wikidata query responses."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any


SNAPSHOT_SCHEMA_VERSION = "tau-rec-wikidata-source-snapshot-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRACK_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class SnapshotError(ValueError):
    """The cache cannot reproduce the source snapshot declared by the track."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"{path} must contain a JSON object")
    return value


def _validated_sources(
    track_dir: Path, cache_dir: Path
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path]]]:
    catalog_manifest_path = track_dir / "catalog_manifest.json"
    catalog_manifest = _read_object(catalog_manifest_path)
    records = catalog_manifest.get("source", {}).get("query_response_records")
    if not isinstance(records, list) or not records:
        raise SnapshotError("catalog manifest has no query-response records")

    sources: list[tuple[dict[str, Any], Path]] = []
    seen_names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SnapshotError(f"query-response record {index} is not an object")
        name = record.get("cache")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise SnapshotError(f"query-response record {index} has an unsafe cache name")
        if name in seen_names:
            raise SnapshotError(f"duplicate cache file in catalog manifest: {name}")
        seen_names.add(name)

        expected_bytes = record.get("bytes")
        response_sha256 = record.get("response_sha256")
        request_sha256 = record.get("request_sha256")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise SnapshotError(f"{name}: invalid byte count")
        if not isinstance(response_sha256, str) or not _SHA256_RE.fullmatch(
            response_sha256
        ):
            raise SnapshotError(f"{name}: invalid response SHA-256")
        if not isinstance(request_sha256, str) or not _SHA256_RE.fullmatch(
            request_sha256
        ):
            raise SnapshotError(f"{name}: invalid request SHA-256")

        source_path = cache_dir / name
        if not source_path.is_file():
            raise SnapshotError(f"missing query-response file: {source_path}")
        actual_bytes = source_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise SnapshotError(
                f"{name}: expected {expected_bytes} bytes, found {actual_bytes}"
            )
        actual_sha256 = sha256_path(source_path)
        if actual_sha256 != response_sha256:
            raise SnapshotError(
                f"{name}: response SHA-256 {actual_sha256} != {response_sha256}"
            )
        sources.append((record, source_path))

    return catalog_manifest, sources


def source_snapshot_manifest(
    track_dir: Path, cache_dir: Path
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path]]]:
    """Validate the frozen cache and return its archive manifest and files."""
    catalog_manifest, sources = _validated_sources(track_dir, cache_dir)
    track = _read_object(track_dir / "track.json")
    track_id = track.get("id")
    if not isinstance(track_id, str) or not _TRACK_ID_RE.fullmatch(track_id):
        raise SnapshotError("track.json has no safe track id")

    catalog_sha256 = catalog_manifest.get("catalog_sha256")
    if not isinstance(catalog_sha256, str) or not _SHA256_RE.fullmatch(
        catalog_sha256
    ):
        raise SnapshotError("catalog manifest has no valid catalog SHA-256")

    catalog_manifest_path = track_dir / "catalog_manifest.json"
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "track_id": track_id,
        "catalog_sha256": catalog_sha256,
        "catalog_manifest_sha256": sha256_path(catalog_manifest_path),
        "source_name": catalog_manifest.get("source", {}).get("name"),
        "query_endpoint": catalog_manifest.get("source", {}).get("query_endpoint"),
        "files": [
            {
                "path": f"cache/{record['cache']}",
                "query_name": record.get("name"),
                "bytes": record["bytes"],
                "request_sha256": record["request_sha256"],
                "response_sha256": record["response_sha256"],
            }
            for record, _ in sorted(sources, key=lambda entry: entry[0]["cache"])
        ],
    }
    return manifest, sources


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def build_source_snapshot(
    track_dir: Path, cache_dir: Path, output_path: Path
) -> dict[str, Any]:
    """Create an atomic, deterministic ``tar.gz`` containing the pinned cache."""
    manifest, sources = source_snapshot_manifest(track_dir, cache_dir)
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    archive_root = f"{manifest['track_id']}-source-snapshot"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="", fileobj=raw_output, mode="wb", compresslevel=9, mtime=0
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
                ) as archive:
                    manifest_name = f"{archive_root}/SNAPSHOT_MANIFEST.json"
                    archive.addfile(
                        _tar_info(manifest_name, len(manifest_bytes)),
                        io.BytesIO(manifest_bytes),
                    )
                    for record, source_path in sorted(
                        sources, key=lambda entry: entry[0]["cache"]
                    ):
                        archive_name = f"{archive_root}/cache/{record['cache']}"
                        with source_path.open("rb") as source:
                            archive.addfile(
                                _tar_info(archive_name, record["bytes"]), source
                            )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": "built-local-not-published",
        "track_id": manifest["track_id"],
        "catalog_sha256": manifest["catalog_sha256"],
        "filename": output_path.name,
        "archive_bytes": output_path.stat().st_size,
        "archive_sha256": sha256_path(output_path),
        "snapshot_manifest_sha256": manifest_sha256,
        "query_response_file_count": len(sources),
        "uncompressed_response_bytes": sum(
            record["bytes"] for record, _ in sources
        ),
        "url": None,
    }
