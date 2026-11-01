#!/usr/bin/env python3
"""Pack the books-wikidata-v2 run outputs into release-asset tarballs.

One tarball per output root. Each holds every
non-smoke stage's state file, every cell's run directory (traces, the stalled
first attempts kept under quarantine/, the run manifest and result rows), and
the non-smoke launcher logs, whose crash lines the analysis counts. Member
paths are relative to the repository root, so extracting an archive there
restores the layout the v4 analysis script reads with ``--root``.

Archives are deterministic: sorted members, no owner or timestamp metadata,
no macOS extended attributes, and a fixed gzip header. Home-directory prefixes
are cut from local paths in the logs, and the account identifiers and local
proxy addresses that provider error messages echo back are replaced with
``<redacted>``.

    uv run python scripts/pack_books_release.py \\
        --root out/books-wikidata-v4 --root out/books-wikidata-v4-addendum --output out/release-assets
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_PATH = re.compile(r"/(?:Users|home)/[^/\s]+/")
ACCOUNT_ID = re.compile(r"\borg_[A-Za-z0-9]{20,}")
LOCAL_PROXY = re.compile(r"https?://(?:localhost|127\.0\.0\.1):\d+")


def _files(root: Path, base: Path) -> list[Path]:
    """Every file to pack under ``root``, as paths relative to ``base``."""
    files: set[Path] = set()
    for state_path in root.glob("*_state.json"):
        if state_path.name.startswith("smoke"):
            continue
        files.add(state_path)
        for cell in json.loads(state_path.read_text())["cells"]:
            run_dir = base / cell["run_dir"]
            files.update(path for path in run_dir.rglob("*") if path.is_file() and not path.name.startswith("._"))
    files.update(log for log in root.glob("*launcher*.log") if not log.name.startswith("smoke"))
    return sorted(path.relative_to(base) for path in files)


def _info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = size, 0o644, 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _scrub(relative: Path, data: bytes) -> bytes:
    """The member's bytes with local paths, account ids and proxy addresses removed."""
    text = data.decode(errors="replace")
    if relative.suffix == ".log":
        text = LOCAL_PATH.sub("", text)
    elif not (ACCOUNT_ID.search(text) or LOCAL_PROXY.search(text)):
        return data
    return LOCAL_PROXY.sub("<redacted>", ACCOUNT_ID.sub("<redacted>", text)).encode()


def pack(root: Path, output: Path, *, base: Path = ROOT) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{root.name}.tar.gz"
    with (open(archive, "wb") as handle,
          gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as gz,
          tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar):
        for relative in _files(root, base):
            data = _scrub(relative, (base / relative).read_bytes())
            tar.addfile(_info(relative.as_posix(), len(data)), io.BytesIO(data))
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sums = []
    for root in args.root:
        archive = pack(root.resolve(), args.output)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        sums.append(f"{digest}  {archive.name}")
        print(f"{archive}: {archive.stat().st_size / 1e6:.1f} MB, sha256 {digest[:12]}")
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
