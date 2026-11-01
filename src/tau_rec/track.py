"""A track is one scored content set: a domain plus the catalog, tasks and
policy that a run is scored against.

Generations still exist *within* a track (leaderboard/GENERATIONS.md); the
track is the coarser label that says which catalog and which domain
vocabulary a number came from. Two entries on different tracks are never
comparable and the renderer never puts them in one table.

Layout of a track directory:

    track.json          {"id": ..., "domain": ..., "distribution": ..., "search_profile": ...}
    catalog.json        the committed catalog
    tasks/*.json
    policy.md

`data/` is the movies-v1 track. Its track.json is additive: the paper cites
data/catalog.json, data/tasks and data/policy.md by name, and none of the
three digests recorded on the board change because a fourth file appeared
beside them. A catalog with no track.json beside it is movies-v1 — that is
every caller that predates tracks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tau_rec.domain import DomainSpec, get_spec
from tau_rec.harness import LEGACY, HarnessProfile, get_harness_profile

DISTRIBUTIONS = ("catalog",)
DEFAULT_TRACK_ID = "movies-v1"
DEFAULT_DOMAIN = "movies"
DEFAULT_SEARCH_PROFILE = "bm25-v1"
SEARCH_PROFILES = (DEFAULT_SEARCH_PROFILE, "bm25f-v1")
DEFAULT_STRUCTURED_PROFILE = "books-wikidata-core-v3"
STRUCTURED_PROFILES = (DEFAULT_STRUCTURED_PROFILE,)
RELEASE_STATUSES = ("public",)


@dataclass(frozen=True)
class Track:
    id: str
    domain: str
    distribution: str
    root: Path
    search_profile: str = DEFAULT_SEARCH_PROFILE
    structured_profile: str = DEFAULT_STRUCTURED_PROFILE
    release_status: str = "public"
    harness_profile: str = LEGACY.id

    @property
    def spec(self) -> DomainSpec:
        return get_spec(self.domain)

    @property
    def harness(self) -> HarnessProfile:
        return get_harness_profile(self.harness_profile)

    @property
    def catalog_path(self) -> Path:
        return self.root / "catalog.json"

    @property
    def tasks_path(self) -> Path:
        return self.root / "tasks"

    @property
    def policy_path(self) -> Path:
        return self.root / "policy.md"

    @property
    def memory_policy_path(self) -> Path:
        """The policy rendering memory mode uses: the policies that need no
        catalog tool, checked in as markdown so its digest is the identity."""
        return self.root / "policy_memory.md"

    def policy_for(self, mode: str) -> Path:
        path = self.memory_policy_path if mode == "memory" else self.policy_path
        if not path.exists():
            raise FileNotFoundError(f"{self.id}: no {mode}-mode policy at {path}")
        return path

    def require_catalog(self) -> Path:
        """The catalog file, or an error naming the missing path."""
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"{self.id}: no catalog at {self.catalog_path}")
        return self.catalog_path


def load_track(root: str | Path) -> Track:
    root = Path(root)
    meta_path = root / "track.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no track.json in {root}")
    meta = json.loads(meta_path.read_text())
    for key in ("id", "domain", "distribution"):
        if key not in meta:
            raise ValueError(f"{meta_path}: missing {key!r}")
    get_spec(meta["domain"])  # raises on an unknown domain, naming the file below
    if meta["distribution"] not in DISTRIBUTIONS:
        raise ValueError(
            f"{meta_path}: distribution {meta['distribution']!r} not in {DISTRIBUTIONS}"
        )
    search_profile = meta.get("search_profile", DEFAULT_SEARCH_PROFILE)
    if search_profile not in SEARCH_PROFILES:
        raise ValueError(
            f"{meta_path}: search_profile {search_profile!r} not in {SEARCH_PROFILES}"
        )
    structured_profile = meta.get(
        "structured_profile", DEFAULT_STRUCTURED_PROFILE
    )
    if structured_profile not in STRUCTURED_PROFILES:
        raise ValueError(
            f"{meta_path}: structured_profile {structured_profile!r} "
            f"not in {STRUCTURED_PROFILES}"
        )
    release_status = meta.get("release_status", "public")
    if release_status not in RELEASE_STATUSES:
        raise ValueError(
            f"{meta_path}: release_status {release_status!r} not in {RELEASE_STATUSES}"
        )
    harness_profile = meta.get("harness_profile", LEGACY.id)
    try:
        get_harness_profile(harness_profile)
    except ValueError as error:
        raise ValueError(f"{meta_path}: {error}") from None
    return Track(
        id=meta["id"], domain=meta["domain"], distribution=meta["distribution"],
        root=root, search_profile=search_profile,
        structured_profile=structured_profile, release_status=release_status,
        harness_profile=harness_profile,
    )


def find_track(track_id: str, repo_root: str | Path = ".") -> Track:
    """A track by id, in the repository layout: movies-v1 is data/, every
    other track is data/tracks/<id>/. Used when a run manifest names a track
    and the caller has only the repository to look in."""
    root = Path(repo_root)
    candidate = root / "data" if track_id == DEFAULT_TRACK_ID else root / "data" / "tracks" / track_id
    if not (candidate / "track.json").exists():
        raise FileNotFoundError(f"no track {track_id!r} at {candidate}")
    track = load_track(candidate)
    if track.id != track_id:
        raise ValueError(f"{candidate}/track.json says id {track.id!r}, expected {track_id!r}")
    return track


def track_for_catalog(catalog_path: str | Path) -> Track:
    """The track a catalog file belongs to.

    A track.json beside the catalog wins. Without one the catalog is treated
    as movies-v1, which keeps every pre-track invocation
    (`--catalog data/catalog.json --tasks data/tasks --policy data/policy.md`)
    meaning exactly what it did.
    """
    root = Path(catalog_path).resolve().parent
    if (root / "track.json").exists():
        return load_track(root)
    return Track(
        id=DEFAULT_TRACK_ID, domain=DEFAULT_DOMAIN, distribution="catalog", root=root,
        search_profile=DEFAULT_SEARCH_PROFILE, release_status="public",
    )
