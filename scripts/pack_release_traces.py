"""Build the trace tarballs attached to a GitHub release.

One directory per run, holding the traces, the trial rows and the task
results behind one leaderboard entry.

Two things need care. Archives are packed without extended attributes or
owner metadata, because macOS xattrs travel as AppleDouble `._*` members
that carry local filesystem state and extract on GNU tar as binary files
wearing a `.json` extension. And a resumed run leaves superseded rows in the
`trial_results.json` of its earlier passes, so rows are kept only where a
matching trace exists — traces are authoritative.

    uv run python scripts/pack_release_traces.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = Path("/tmp/tau-rec-release/stage")
OUT = Path("/tmp/tau-rec-release")

# The nine directories behind the nine g0 board entries.
G0_RUNS = {
    "[final]run-gemini25flash": "gemini-25-flash",
    "[final]run-gpt54-medthink": "gpt54-medium-thinking",
    "[final]run-gpt54-nothink": "gpt54-no-thinking",
    "[final]run-sonnet46-nothink": "sonnet46-no-thinking",
    "ds-v4-flash-4t-gpt5mini": "ds-v4-flash",
    "ds-v4-flash-thinking-4t": "ds-v4-flash-high-thinking",
    "ds-v4-flash-thinking-max": "ds-v4-flash-max-thinking",
    "gpt5mini-paper": "gpt5-mini",
    "qwen3-32b-paper": "qwen3-32b",
}

G1_RUNS = {
    "dsflash": "deepseek-flash",
    "glm": "glm-53-flash",
    "luna": "gpt56-luna-medium",
    "mistral": "mistral-small-3",
    "grok": "grok-43",
}


def collect(run_dir: Path, dest: Path) -> tuple[int, int]:
    """Merge a run's timestamp dirs into one flat directory.

    A resumed run has several timestamp dirs whose traces union to the real
    trial set. Later passes win on collision.
    """
    (dest / "traces").mkdir(parents=True, exist_ok=True)
    for ts in sorted(run_dir.iterdir()):
        for trace in (ts / "traces").glob("*.json"):
            if trace.name.startswith("._"):
                continue
            shutil.copy2(trace, dest / "traces" / trace.name)
        for name in ("task_results.json", "run_manifest.json"):
            if (ts / name).exists():
                shutil.copy2(ts / name, dest / name)

    kept = {
        (p.name.rsplit("_trial", 1)[0], int(p.stem.rsplit("_trial", 1)[1]))
        for p in (dest / "traces").glob("*.json")
    }
    rows, seen = [], set()
    for ts in sorted(run_dir.iterdir(), reverse=True):
        f = ts / "trial_results.json"
        if not f.exists():
            continue
        for row in json.load(f.open()):
            key = (row["task_id"], row["trial"])
            if key in kept and key not in seen:
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda r: (r["task_id"], r["trial"]))
    (dest / "trial_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    return len(kept), len(rows)


def pack(stage_root: Path, name: str) -> Path:
    """tar without xattrs, AppleDouble members, or packer identity."""
    out = OUT / f"{name}.tar.gz"
    out.unlink(missing_ok=True)
    subprocess.run(
        ["tar", "--no-xattrs", "--no-mac-metadata",
         "--uid", "0", "--gid", "0", "--uname", "", "--gname", "",
         "-czf", str(out), "-C", str(stage_root.parent), stage_root.name],
        check=True, env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"},
    )
    return out


def main() -> int:
    shutil.rmtree(STAGE, ignore_errors=True)
    bad = []

    for label, runs, src_root in (
        ("tau-rec-g0-traces", G0_RUNS, ROOT / "results" / "final-traces"),
        ("tau-rec-g1-traces", G1_RUNS, ROOT / "out" / "g1"),
    ):
        stage_root = STAGE / label
        for src, clean in runs.items():
            n_traces, n_rows = collect(src_root / src, stage_root / clean)
            counts = Counter(
                p.name.rsplit("_trial", 1)[0]
                for p in (stage_root / clean / "traces").glob("*.json")
            )
            flag = "" if n_traces == n_rows else "  <-- ROW/TRACE MISMATCH"
            if n_traces != n_rows:
                bad.append(f"{label}/{clean}")
            print(f"  {clean:28s} {n_traces:4d} traces  {len(counts):2d} tasks  "
                  f"{n_rows:4d} rows{flag}")

        out = pack(stage_root, label)
        blob = out.read_bytes()
        import gzip
        raw = gzip.decompress(blob)
        for marker in (b"com.apple", b"nbs", b"/Users/"):
            if marker in raw:
                bad.append(f"{label}: archive still contains {marker!r}")
        print(f"{label}.tar.gz  {out.stat().st_size / 1e6:.1f} MB  "
              f"({len(raw.split(b'ustar')) - 1} members)\n")

    if bad:
        print("FAILED:\n  " + "\n  ".join(bad))
        return 1
    print("clean: no xattrs, no packer identity, no local paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
