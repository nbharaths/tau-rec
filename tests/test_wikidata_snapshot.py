import hashlib
import json
import tarfile

import pytest

from tau_rec.catalog.wikidata_snapshot import SnapshotError, build_source_snapshot


def _write_fixture(tmp_path):
    track = tmp_path / "track"
    cache = tmp_path / "cache"
    track.mkdir()
    cache.mkdir()
    (track / "track.json").write_text(json.dumps({"id": "books-test"}))

    records = []
    for index, payload in enumerate((b"first response\n", b"second response\n"), 1):
        name = f"query-{index}.tsv"
        (cache / name).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        records.append({
            "name": f"query-{index}",
            "cache": name,
            "bytes": len(payload),
            "request_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "response_sha256": digest,
        })
    catalog_manifest = {
        "catalog_sha256": "a" * 64,
        "source": {
            "name": "Wikidata",
            "query_endpoint": "https://example.test/query",
            "query_response_records": records,
        },
    }
    (track / "catalog_manifest.json").write_text(json.dumps(catalog_manifest))
    return track, cache


def test_snapshot_archive_is_deterministic_and_self_describing(tmp_path):
    track, cache = _write_fixture(tmp_path)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    first_record = build_source_snapshot(track, cache, first)
    second_record = build_source_snapshot(track, cache, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_record["archive_sha256"] == second_record["archive_sha256"]
    assert first_record["query_response_file_count"] == 2
    with tarfile.open(first, "r:gz") as archive:
        names = archive.getnames()
        assert names == [
            "books-test-source-snapshot/SNAPSHOT_MANIFEST.json",
            "books-test-source-snapshot/cache/query-1.tsv",
            "books-test-source-snapshot/cache/query-2.tsv",
        ]
        manifest_file = archive.extractfile(names[0])
        assert manifest_file is not None
        manifest = json.load(manifest_file)
        assert manifest["schema_version"] == "tau-rec-wikidata-source-snapshot-v1"
        assert manifest["catalog_sha256"] == "a" * 64


def test_snapshot_rejects_tampered_response(tmp_path):
    track, cache = _write_fixture(tmp_path)
    (cache / "query-1.tsv").write_text("tampered\n")

    with pytest.raises(SnapshotError, match="query-1.tsv"):
        build_source_snapshot(track, cache, tmp_path / "snapshot.tar.gz")


def test_snapshot_rejects_unsafe_cache_name(tmp_path):
    track, cache = _write_fixture(tmp_path)
    manifest_path = track / "catalog_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["query_response_records"][0]["cache"] = "../escape.tsv"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SnapshotError, match="unsafe cache name"):
        build_source_snapshot(track, cache, tmp_path / "snapshot.tar.gz")


def test_snapshot_rejects_unsafe_track_id(tmp_path):
    track, cache = _write_fixture(tmp_path)
    (track / "track.json").write_text(json.dumps({"id": "../escape"}))

    with pytest.raises(SnapshotError, match="safe track id"):
        build_source_snapshot(track, cache, tmp_path / "snapshot.tar.gz")
