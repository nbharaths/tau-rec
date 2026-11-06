"""Catalog years that disagree with the year in the work's own Wikidata description.

Wikidata descriptions of literary works usually open with the first-publication
year ("2013 novel by Tui T. Sutherland"). When that year differs from the
catalog's ``publication_year``, one of the two is usually wrong: the catalog
picked a later edition (Ruby Red: 2000 vs 2009) or the description is stale
(The Thousandth Floor: catalog 2016 is right, the description's 2017 is not).
The check only flags; every flag needs a bibliographic source before a label
override is written. It never changes released bytes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# A four-digit year, up to four words, then a work noun: "2013 novel by ...",
# "1994 children's fantasy book", "2008 crime novel". Four digits avoid typos
# such as "199 biography".
_DESCRIPTION_YEAR = re.compile(
    r"^(\d{4})\s+(?:[\w'’-]+\s+){0,4}?"
    r"(novel|novella|book|poem|poetry collection|collection|anthology|memoir|autobiography|biography"
    r"|essay|work|story|stories|play|comic|graphic novel|picture book|nonfiction|non-fiction)\b",
    re.IGNORECASE,
)


def description_year(description: str | None) -> int | None:
    """The year a description opens with, or None when it does not open with one."""
    match = _DESCRIPTION_YEAR.match(description or "")
    return int(match.group(1)) if match else None


@dataclass
class YearMismatch:
    item_id: str
    title: str
    catalog_year: int | None
    description_year: int
    description: str
    # family_id -> "+" (would join the answer set) or "-" (would leave it)
    family_changes: dict[str, str] = field(default_factory=dict)


def year_mismatches(items, tasks, families) -> list[YearMismatch]:
    """Every item whose description year differs from its catalog year.

    ``families`` are family-manifest rows; a family changes when the item's
    membership in the volunteer task's answer set differs under the two years.
    """
    volunteer = {
        family["family_id"]: [tagged.constraint for tagged in tasks[family["task_ids"]["volunteer"]].constraints]
        for family in families
    }
    out: list[YearMismatch] = []
    for item in items:
        year = description_year(item.description)
        if year is None or year == item.publication_year:
            continue
        row = YearMismatch(item.id, item.title, item.publication_year, year, item.description)
        relabelled = item.model_copy(update={"publication_year": year})
        for family_id, constraints in volunteer.items():
            now = all(c.evaluate(item) for c in constraints)
            then = all(c.evaluate(relabelled) for c in constraints)
            if now != then:
                row.family_changes[family_id] = "+" if then else "-"
        out.append(row)
    return out
