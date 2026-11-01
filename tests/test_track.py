import json
from pathlib import Path

import pytest

from tau_rec.domain import BOOKS, MOVIES
from tau_rec.track import Track, load_track, track_for_catalog

ROOT = Path(__file__).resolve().parent.parent


def test_data_dir_is_the_movies_v1_track():
    track = load_track(ROOT / "data")
    assert (track.id, track.domain, track.distribution) == ("movies-v1", "movies", "catalog")
    assert track.spec is MOVIES
    assert track.catalog_path == ROOT / "data" / "catalog.json"
    assert track.tasks_path == ROOT / "data" / "tasks"
    assert track.policy_path == ROOT / "data" / "policy.md"
    assert track.release_status == "public"
    assert track.require_catalog().exists()


def test_wikidata_books_track_is_a_committed_public_catalog():
    track = load_track(ROOT / "data" / "tracks" / "books-wikidata-v2")
    assert (track.id, track.domain, track.distribution) == (
        "books-wikidata-v2", "books", "catalog",
    )
    assert track.spec is BOOKS
    assert track.search_profile == "bm25f-v1"
    assert track.structured_profile == "books-wikidata-core-v3"
    assert track.release_status == "public"
    assert track.require_catalog().is_file()


def test_catalog_without_track_json_is_movies_v1(tmp_path):
    (tmp_path / "catalog.json").write_text("[]")
    track = track_for_catalog(tmp_path / "catalog.json")
    assert track.id == "movies-v1" and track.spec is MOVIES
    # And with one beside it, the file wins.
    (tmp_path / "track.json").write_text(json.dumps({"id": "books-example", "domain": "books", "distribution": "catalog"}))
    track = track_for_catalog(tmp_path / "catalog.json")
    assert track.id == "books-example" and track.spec is BOOKS


def test_track_search_profile_is_optional_and_validated(tmp_path):
    (tmp_path / "track.json").write_text(json.dumps({
        "id": "books-example", "domain": "books", "distribution": "catalog",
        "search_profile": "bm25f-v1",
    }))
    assert load_track(tmp_path).search_profile == "bm25f-v1"
    (tmp_path / "track.json").write_text(json.dumps({
        "id": "books-example", "domain": "books", "distribution": "catalog",
        "search_profile": "mystery-search",
    }))
    with pytest.raises(ValueError, match="search_profile"):
        load_track(tmp_path)


def test_track_structured_profile_is_optional_and_validated(tmp_path):
    meta = {
        "id": "books-example", "domain": "books", "distribution": "catalog",
        "structured_profile": "books-wikidata-core-v3",
    }
    (tmp_path / "track.json").write_text(json.dumps(meta))
    assert load_track(tmp_path).structured_profile == "books-wikidata-core-v3"
    meta["structured_profile"] = "mystery-structured"
    (tmp_path / "track.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="structured_profile"):
        load_track(tmp_path)


def test_track_release_status_is_optional_and_validated(tmp_path):
    (tmp_path / "track.json").write_text(json.dumps({
        "id": "books-example", "domain": "books", "distribution": "catalog",
    }))
    assert load_track(tmp_path).release_status == "public"
    (tmp_path / "track.json").write_text(json.dumps({
        "id": "books-example", "domain": "books", "distribution": "catalog",
        "release_status": "secret-ish",
    }))
    with pytest.raises(ValueError, match="release_status"):
        load_track(tmp_path)


def test_track_without_its_catalog_says_so(tmp_path):
    (tmp_path / "track.json").write_text(json.dumps({"id": "books-example", "domain": "books", "distribution": "catalog"}))
    track = load_track(tmp_path)
    with pytest.raises(FileNotFoundError, match="no catalog at"):
        track.require_catalog()


@pytest.mark.parametrize("meta, message", [
    ({"id": "x", "domain": "podcasts", "distribution": "catalog"}, "unknown domain"),
    ({"id": "x", "domain": "movies", "distribution": "tarball"}, "distribution"),
    ({"domain": "movies", "distribution": "catalog"}, "missing 'id'"),
])
def test_bad_track_json_fails_loudly(tmp_path, meta, message):
    (tmp_path / "track.json").write_text(json.dumps(meta))
    with pytest.raises((ValueError, KeyError), match=message):
        load_track(tmp_path)


def test_missing_track_json():
    with pytest.raises(FileNotFoundError):
        load_track(Path("/nonexistent"))
