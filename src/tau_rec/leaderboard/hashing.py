"""Canonical content digests for leaderboard comparability.

Two runs are only comparable if they scored the same tasks against the same
policy and catalog. These helpers define exactly one way to hash each, so a
digest recorded in a run manifest means the same thing as one recorded in a
leaderboard entry.

Directory hashing is deliberately *not* plain byte concatenation: it hashes a
sorted manifest of `<filename>:<file digest>` lines, so a file rename shows up
as drift and concatenation ambiguities cannot arise.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["hash_file", "hash_task_dir", "file_manifest"]


def hash_file(path: str | Path) -> str:
    """sha256 of a single file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_manifest(task_dir: str | Path, pattern: str = "*.json") -> str:
    """The exact string that :func:`hash_task_dir` digests.

    One `<filename>:<sha256>` line per file, sorted by filename, newline
    separated with a trailing newline.
    """
    directory = Path(task_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    lines = [
        f"{p.name}:{hash_file(p)}"
        for p in sorted(directory.glob(pattern), key=lambda p: p.name)
    ]
    if not lines:
        raise ValueError(f"no files matching {pattern!r} in {directory}")
    return "\n".join(lines) + "\n"


def hash_task_dir(task_dir: str | Path, pattern: str = "*.json") -> str:
    """sha256 over the sorted file manifest of a directory."""
    return hashlib.sha256(file_manifest(task_dir, pattern).encode()).hexdigest()
