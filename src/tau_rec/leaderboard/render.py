"""Render LEADERBOARD.md from committed submission entries.

The renderer must be a pure function of the entry files: no clock, no network,
no unseeded RNG. `render --check` re-renders and compares byte-for-byte, so any
impurity here turns into a CI failure.
"""
from __future__ import annotations

import json
from pathlib import Path

from tau_rec.leaderboard.entry import (
    CONTENT_DIGESTS_UNRECORDED,
    SAME_MODEL_AS_SIMULATOR,
    LeaderboardEntry,
)
from tau_rec.metrics.bootstrap import bootstrap_ci
from tau_rec.metrics.pass_k import aggregate_pass_k, pass_at_k

# The simulator every seed config used. Changing the simulator moves scores by
# ~13.7 pass^1 points against a ~28-point board span, so entries using anything
# else are ranked in their own section rather than against the main table.
STANDARD_SIMULATOR = "gpt-5-mini"

KS = (1, 2, 4)

# True of every entry in a generation, so noise if repeated per row.
GENERATION_WIDE_CAVEATS = {CONTENT_DIGESTS_UNRECORDED}

BANNER = """<!-- GENERATED FILE — DO NOT EDIT BY HAND.
     Regenerate with: uv run tau-rec leaderboard render
     Source of truth: leaderboard/entries/*.json -->
"""

# README carries its own top-N summary, which is the first thing most readers
# see and so the worst thing to leave stale -- it had drifted twice, still
# advertising "top 3 of 12" against a board of 16 and naming a leader that had
# been displaced. The renderer owns the region between these markers, and
# `render --check` fails on drift the same way it does for the board itself.
README_BEGIN = "<!-- LEADERBOARD:BEGIN — generated, do not edit by hand -->"
README_END = "<!-- LEADERBOARD:END -->"

README_TOP_N = 3

# Short descriptors for the README tables. An unlisted generation falls back to
# the bare label, so adding g2 changes the board without touching this file.
GENERATION_BLURBS = {"g1": "the current harness", "g0": "the paper cohort"}


def load_entries(entries_dir: str | Path) -> list[LeaderboardEntry]:
    paths = sorted(Path(entries_dir).glob("*.json"), key=lambda p: p.name)
    return [LeaderboardEntry.model_validate_json(p.read_text()) for p in paths]


def per_task_scores(entry: LeaderboardEntry, k: int) -> list[float]:
    return [pass_at_k(tc.n, tc.c, k) for tc in entry.per_task.values()]


def summarize(entry: LeaderboardEntry) -> dict:
    """All derived quantities for one entry. Nothing here is stored on disk."""
    out: dict = {"entry": entry}
    for k in KS:
        scores = per_task_scores(entry, k)
        out[f"pass_{k}"] = aggregate_pass_k(entry.task_results, k)
        out[f"ci_{k}"] = bootstrap_ci(scores) if scores else (0.0, 0.0)
    return out


def _sort_key(row: dict) -> tuple:
    # Best first: pass^4, then pass^1, then submission_id for a total order.
    return (-row["pass_4"], -row["pass_1"], row["entry"].submission_id)


def _fmt_row(rank: int, row: dict) -> str:
    e = row["entry"]
    cells = [str(rank), e.display_name]
    for k in KS:
        lo, hi = row[f"ci_{k}"]
        cells.append(f"{row[f'pass_{k}']:.3f} <sub>[{lo:.2f}, {hi:.2f}]</sub>")
    cells += [str(e.trials_per_task), e.harness_generation]
    # Caveats that hold for an entire generation are documented once under the
    # table rather than repeated on every row.
    shown = sorted(set(e.caveats) - GENERATION_WIDE_CAVEATS)
    cells.append(", ".join(f"`{c}`" for c in shown) or "—")
    # The row and the evidence for it should be one click apart; an entry
    # nobody can re-derive is a different kind of claim and says so here.
    cells.append(f"[archive]({e.traces_url})" if e.traces_url else "—")
    return "| " + " | ".join(cells) + " |"


def _table(rows: list[dict]) -> str:
    header = (
        "| # | Model | pass^1 | pass^2 | pass^4 | Trials | Gen | Notes | Traces |\n"
        "|---|-------|--------|--------|--------|--------|-----|-------|--------|"
    )
    body = "\n".join(_fmt_row(i, r) for i, r in enumerate(rows, start=1))
    return f"{header}\n{body}" if body else f"{header}\n| — | _no entries_ | | | | | | | |"


def _generation_sort_key(label: str) -> tuple:
    """Newest generation first. 'g10' must sort above 'g9', so compare the
    numeric suffix rather than the string."""
    digits = "".join(c for c in label if c.isdigit())
    return (-int(digits) if digits else 0, label)


def _generation_block(label: str, rows: list[dict], is_newest: bool) -> list[str]:
    standard = sorted(
        [r for r in rows if r["entry"].simulator_model == STANDARD_SIMULATOR], key=_sort_key
    )
    nonstandard = sorted(
        [r for r in rows if r["entry"].simulator_model != STANDARD_SIMULATOR], key=_sort_key
    )

    # "most recent" is a claim about this board only. It deliberately does not
    # say "current harness": the newest entries on the board may still predate
    # HEAD, which is the whole reason generations are labelled.
    heading = f"## Generation `{label}`" + (" — most recent" if is_newest else "")
    parts = [f"{heading}\n", f"Simulator: `{STANDARD_SIMULATOR}`, 60 tasks.\n", _table(standard)]

    if nonstandard:
        parts += [
            f"\n### `{label}`, non-standard simulator\n",
            "Not comparable with the table above — the user simulator is worth roughly "
            "13.7 pass^1 points across models, against a board span of ~28 points.\n",
            _table(nonstandard),
        ]
    return parts


def render(entries_dir: str | Path) -> str:
    rows = [summarize(e) for e in load_entries(entries_dir)]

    by_generation: dict[str, list[dict]] = {}
    for row in rows:
        by_generation.setdefault(row["entry"].harness_generation, []).append(row)
    labels = sorted(by_generation, key=_generation_sort_key)

    parts = [
        BANNER,
        "# τ-Rec Leaderboard\n",
        "`primary_reward = constraint_score × policy_score`, scored programmatically "
        "against the catalog — no LLM judge. `pass^k` is the unbiased estimator "
        "`C(c,k)/C(n,k)`, averaged over tasks; a task with fewer than `k` trials counts "
        "as zero. Brackets are 95% BCa bootstrap intervals over per-task scores.\n",
        "Entries store only raw per-task `{n, c}`. Every number below is derived at "
        "render time, so the whole board moves when the evaluator does.\n",
    ]

    if len(labels) > 1:
        parts.append(
            "**Generations are ranked separately and cannot be compared across tables.** "
            "Each was scored against different policy, task, or evaluator content; "
            "`leaderboard/GENERATIONS.md` records exactly what differs.\n"
        )

    for i, label in enumerate(labels):
        parts += _generation_block(
            label, by_generation[label], is_newest=(i == 0 and len(labels) > 1)
        )
        parts.append("")

    parts += [
        "\n## Columns\n",
        "- **Trials** — trials per task (`n`). The standard is 4; more is welcome, not required.\n"
        "- **Gen** — harness generation. Entries from different generations were scored "
        "against different policy/task/evaluator content; see `leaderboard/GENERATIONS.md`.\n"
        f"- **Notes** — `{SAME_MODEL_AS_SIMULATOR}` means the agent was graded by a "
        "simulator running the same model, which is a confound.\n"
        "- **Traces** — every conversation behind the row, so the numbers can be "
        "re-derived rather than taken on trust. `—` means the entry ships no traces.\n",
    ]

    if any(CONTENT_DIGESTS_UNRECORDED in r["entry"].caveats for r in rows):
        parts.append(
            f"Some entries carry `{CONTENT_DIGESTS_UNRECORDED}`: they predate content-digest "
            "recording, so their policy/task/catalog hashes cannot be verified against the "
            "repository. See `leaderboard/GENERATIONS.md`.\n"
        )
    return "\n".join(parts).rstrip() + "\n"


def _readme_table(rows: list[dict], top: int) -> str:
    header = ("| # | Model | pass^1 | pass^2 | pass^4 |\n"
              "|---|-------|--------|--------|--------|")
    lines = []
    for rank, row in enumerate(rows[:top], start=1):
        # The board ranks on pass^4, so bold it on the leader to show which
        # column the ordering came from.
        p4 = f"{row['pass_4']:.3f}"
        cells = [str(rank), row["entry"].display_name,
                 f"{row['pass_1']:.3f}", f"{row['pass_2']:.3f}",
                 f"**{p4}**" if rank == 1 else p4]
        lines.append("| " + " | ".join(cells) + " |")
    return header + "\n" + "\n".join(lines)


def readme_block(entries_dir: str | Path, top: int = README_TOP_N) -> str:
    """The generated region of README.md: one top-N table per generation."""
    rows = [summarize(e) for e in load_entries(entries_dir)]
    by_generation: dict[str, list[dict]] = {}
    for row in rows:
        # Non-standard-simulator entries are not comparable with the rest, so
        # they are excluded here exactly as they are on the main board.
        if row["entry"].simulator_model != STANDARD_SIMULATOR:
            continue
        by_generation.setdefault(row["entry"].harness_generation, []).append(row)

    parts = [README_BEGIN, ""]
    for label in sorted(by_generation, key=_generation_sort_key):
        ranked = sorted(by_generation[label], key=_sort_key)
        blurb = GENERATION_BLURBS.get(label)
        name = f"`{label}`, {blurb}" if blurb else f"`{label}`"
        shown = min(top, len(ranked))
        parts += [f"{name} (top {shown} of {len(ranked)}):", "",
                  _readme_table(ranked, top), ""]
    parts.append(README_END)
    return "\n".join(parts)


def sync_readme(entries_dir: str | Path, readme_path: str | Path,
                top: int = README_TOP_N) -> bool:
    """Rewrite README's generated region. True if the file changed.

    A README without the markers is left alone rather than rewritten: the
    region is opt-in, so this stays a no-op for forks that dropped it.
    """
    path = Path(readme_path)
    if not path.exists():
        return False
    text = path.read_text()
    if README_BEGIN not in text or README_END not in text:
        return False
    head, _, rest = text.partition(README_BEGIN)
    _, _, tail = rest.partition(README_END)
    updated = head + readme_block(entries_dir, top) + tail
    if updated == text:
        return False
    path.write_text(updated)
    return True


def write(entries_dir: str | Path, out_path: str | Path) -> str:
    text = render(entries_dir)
    Path(out_path).write_text(text)
    return text


def check(entries_dir: str | Path, out_path: str | Path,
          readme_path: str | Path | None = None) -> bool:
    """True if the generated files already match a fresh render."""
    path = Path(out_path)
    if not path.exists() or path.read_text() != render(entries_dir):
        return False
    if readme_path is None:
        return True
    readme = Path(readme_path)
    if not readme.exists():
        return True
    text = readme.read_text()
    if README_BEGIN not in text or README_END not in text:
        return True
    current = README_BEGIN + text.partition(README_BEGIN)[2].partition(README_END)[0]
    return current + README_END == readme_block(entries_dir)
