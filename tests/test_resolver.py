"""The memory-mode resolver: exact match after normalisation, never fuzzy."""
import pytest

from tau_rec.data_model.catalog import Book, Catalog, Movie
from tau_rec.domain import BOOKS, MOVIES
from tau_rec.environment.resolver import TitleResolver, normalize_title


@pytest.mark.parametrize("raw, expected", [
    ("The Matrix", "matrix"),
    ("A Quiet Place (2018)", "quiet place"),
    ("Dune: Part Two", "dune part 2"),
    ("Rocky IV", "rocky 4"),
    ("The Godfather Part II", "godfather part 2"),
    ("Murder on the Orient Express - Paperback", "murder on the orient express"),
    ("Dune Messiah (Hardcover)", "dune messiah"),
    ("The Lord of the Rings Part I – Audiobook (Standard Edition)", "lord of the rings part 1"),
    ("Gone Girl (Kindle Edition)", "gone girl"),
    ("  ocean's  eleven  ", "ocean s eleven"),
])
def test_normalize_title(raw, expected):
    assert normalize_title(raw) == expected


def _movie(i, title, year):
    return Movie(id=i, title=title, release_date=f"{year}-01-01", runtime=100, genres=["Drama"],
                 overview="", cast=[], director="D", rating=7.0, vote_count=50, streaming_services=[])


def _book(i, title, year):
    return Book(id=i, title=title, authors=["A. Author"], publication_year=year, genres=["Fiction"],
                description="", publisher="", formats=["paperback"])


def test_exact_match_after_normalisation_resolves():
    catalog = Catalog(items=[_movie("m1", "The Godfather Part II", 1974), _movie("m2", "Heat", 1995)])
    r = TitleResolver(catalog, MOVIES)
    assert r.resolve("godfather part 2").resolved_id == "m1"
    assert r.resolve("The Godfather: Part II (1974)").method == "exact"
    assert r.resolve("HEAT").resolved_id == "m2"


def test_sequels_never_resolve_fuzzily():
    """Part II vs Part III sit at high similarity; only an exact title may
    resolve, and the near miss is recorded for the audit."""
    catalog = Catalog(items=[_movie("p2", "Mission Impossible Dead Reckoning Part II", 2025),
                             _movie("p3", "Mission Impossible Dead Reckoning Part III", 2026)])
    r = TitleResolver(catalog, MOVIES)
    assert r.resolve("Mission Impossible Dead Reckoning Part III").resolved_id == "p3"
    miss = r.resolve("Mission Impossible Dead Reckoning Part IIII")
    assert miss.resolved_id is None and miss.method == "unresolved"
    assert miss.best_ratio is not None and miss.best_ratio > 0.9
    assert miss.best_candidate in ("Mission Impossible Dead Reckoning Part II",
                                   "Mission Impossible Dead Reckoning Part III")


def test_colliding_titles_are_ambiguous_unless_disambiguated():
    catalog = Catalog(items=[_book("b78", "The Stand", 1978), _book("b90", "The Stand", 1990)])
    r = TitleResolver(catalog, BOOKS)
    assert r.resolve("The Stand").method == "ambiguous"
    assert r.resolve("The Stand", year=1990).resolved_id == "b90"
    assert r.resolve("The Stand", year=1999).method == "ambiguous"


def test_unknown_title_is_unresolved_with_no_credit():
    catalog = Catalog(items=[_movie("m1", "Heat", 1995)])
    res = TitleResolver(catalog, MOVIES).resolve("Collateral")
    assert res.resolved_id is None and res.method == "unresolved"
