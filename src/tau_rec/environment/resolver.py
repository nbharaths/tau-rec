"""Map a title a model names from memory onto a catalog id — or refuse.

Memory mode gives the agent one tool, `recommend(title)`, and no catalog. To
score the recommendation the title has to become an id, and the resolver is
where that must not go wrong in the model's favour. The rule is exact match
after normalisation, nothing else: sequel titles can sit at 0.98 string
similarity (*Part II* vs *Part III*), so fuzzy acceptance would credit the
wrong item. A near miss is *recorded* so the audit can count
resolver misses, but it never resolves.

Normalisation is applied to both sides, so it only has to be consistent, not
clever: casefold, drop punctuation, drop leading articles and a trailing
"(YYYY)", drop the format and edition suffixes retail listings carry, and
write numerals one way (roman, ordinal words and number words become
digits). Two catalog titles that collide after normalisation are ambiguous
by construction, and a query hitting them resolves only if `year` or
`platform` picks one out.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.conversation import Resolution
from tau_rec.domain import DomainSpec

_ARTICLES = ("the ", "a ", "an ")
_YEAR_SUFFIX = re.compile(r"\s*[\(\[]\s*(19|20)\d{2}\s*[\)\]]\s*$")
_SUFFIXES = re.compile(
    r"\s*[-–—:(\[]\s*(kindle\s*edition|kindle|e-?book|"
    r"audiobook|audio\s*cd|paperback|hardcover|mass\s*market|standard\s*edition|"
    r"deluxe\s*edition|remastered|definitive\s*edition)"
    r"\s*[\)\]]?\s*$",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^\w\s]")
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7",
          "viii": "8", "ix": "9", "x": "10", "xi": "11", "xii": "12"}
_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
          "seven": "7", "eight": "8", "nine": "9", "ten": "10",
          "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
          "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
          "1st": "1", "2nd": "2", "3rd": "3", "4th": "4", "5th": "5"}


def normalize_title(title: str) -> str:
    s = title.casefold().strip()
    s = _YEAR_SUFFIX.sub("", s)
    # Suffixes can stack ("... - Paperback (Standard Edition)").
    for _ in range(3):
        new = _SUFFIXES.sub("", s)
        if new == s:
            break
        s = new
    s = _PUNCT.sub(" ", s)
    tokens = [_ROMAN.get(t, _WORDS.get(t, t)) for t in s.split()]
    s = " ".join(tokens)
    for art in _ARTICLES:
        if s.startswith(art):
            s = s[len(art):]
            break
    return s.strip()


def _item_year(item) -> int | None:
    year = getattr(item, "publication_year", None)
    if year:
        return int(year)
    date = getattr(item, "release_date", None)
    if date and len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    return None


@dataclass
class TitleResolver:
    catalog: Catalog
    spec: DomainSpec

    def __post_init__(self) -> None:
        self._index: dict[str, list[str]] = {}
        for item in self.catalog.items:
            self._index.setdefault(normalize_title(item.title), []).append(item.id)

    def resolve(self, title: str, year: int | None = None, platform: str | None = None) -> Resolution:
        key = normalize_title(title)
        ids = self._index.get(key, [])
        res = Resolution(query=title, year=year, platform=platform, method="unresolved")

        if len(ids) == 1:
            res.resolved_id, res.method = ids[0], "exact"
            return res

        if len(ids) > 1:
            narrowed = ids
            if year is not None:
                narrowed = [i for i in narrowed if _item_year(self.catalog.get(i)) == year]
            if platform is not None and len(narrowed) != 1:
                want = self.spec.normalize_service(platform)
                narrowed = [i for i in narrowed
                            if want in (getattr(self.catalog.get(i), self.spec.availability_field, None) or [])]
            if len(narrowed) == 1:
                res.resolved_id, res.method = narrowed[0], "disambiguated"
            else:
                res.method = "ambiguous"
            return res

        # No exact match. Record the nearest title so the audit can tell a
        # resolver miss from a title the catalog simply lacks — but do not
        # resolve on it.
        if self._index:
            best = max(self._index, key=lambda k: difflib.SequenceMatcher(None, key, k).ratio())
            res.best_ratio = round(difflib.SequenceMatcher(None, key, best).ratio(), 3)
            res.best_candidate = self.catalog.get(self._index[best][0]).title
        return res
