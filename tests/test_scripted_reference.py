"""The eliminating scripted reference (run_scripted_reference.py --order eliminate)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "analysis/books-controlled/run_scripted_reference.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_scripted_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(item, genres=(), year=2000, languages=("English",), countries=("United States",)):
    return {"id": item, "title": item, "genres": list(genres), "publication_year": year,
            "languages": list(languages), "countries": list(countries)}


def test_refusals_name_the_fields_that_hold_hidden_requirements():
    refused = _load().refused_fields
    reply = ("To answer your question: it should include Science Fiction. I'd rather not say more about "
             "the genre. I'd rather not say about the publication year. No preference on the language.")
    assert refused(reply) == ["genres", "publication_year"]
    assert refused("No preference on the genre or the language.") == []


def test_a_rejection_rules_out_every_hypothesis_the_rejected_book_satisfies():
    hypotheses = _load().Hypotheses(
        [_row("a", ["SF", "Fantasy"]), _row("b", ["SF", "Horror"]), _row("c", ["SF", "Young Adult"])],
        ["genres"], stated_genres=["SF"])
    assert hypotheses.best() == "a"  # a tie goes to the search tool's order
    hypotheses.reject("a")
    assert hypotheses.best() == "b"
    hypotheses.reject("b")
    assert hypotheses.best() == "c"
    hypotheses.reject("c")
    assert hypotheses.best() is None  # no remaining hypothesis fits any candidate: abstain


def test_the_next_proposal_covers_the_most_remaining_hypotheses():
    hypotheses = _load().Hypotheses(
        [_row("one", ["Fantasy"]), _row("three", ["Fantasy", "Horror", "Young Adult"]), _row("two", ["Horror", "Mystery"])],
        ["genres"])
    assert hypotheses.best() == "three"
    hypotheses.reject("three")
    assert hypotheses.best() == "two"  # Mystery is the only label not yet ruled out


def test_year_hypotheses_keep_the_bounds_a_rejected_year_leaves_open():
    hypotheses = _load().Hypotheses(
        [_row("mid", year=1990), _row("new", year=2010), _row("old", year=1950)], ["publication_year"])
    hypotheses.reject("mid")
    # Rejecting 1990 rules out every bound 1990 meets, so the next proposal is an extreme year.
    assert hypotheses.best() in {"new", "old"}
    hypotheses.reject("new")
    assert hypotheses.best() == "old"  # "no later than 1950" or a range ending before 1990 is still open


def test_two_refused_fields_are_ruled_out_jointly():
    hypotheses = _load().Hypotheses(
        [_row("ya2000", ["Young Adult"], 2000), _row("ya1990", ["Young Adult"], 1990), _row("h2000", ["Horror"], 2000)],
        ["genres", "publication_year"])
    hypotheses.reject("ya2000")
    # Young Adult with an earlier bound is still possible, so the 1990 Young Adult book stays a candidate.
    assert hypotheses.score("ya1990") > 0
    assert hypotheses.score("ya2000") == 0


def test_countries_match_parent_labels_as_the_evaluator_does():
    hypotheses = _load().Hypotheses(
        [_row("eng", countries=["England"]), _row("sco", countries=["Scotland"]), _row("us", countries=["United States"])],
        ["countries"])
    # England also satisfies "country includes United Kingdom", so it covers twice the single-label weight.
    assert hypotheses.score("eng") > hypotheses.score("us")
    hypotheses.reject("eng")
    assert hypotheses.score("sco") == hypotheses.score("us")  # United Kingdom went with England


def test_broad_hypotheses_keep_two_values_where_the_templates_allow_one():
    rows = [_row("en", languages=["English"]), _row("fr", languages=["French"]),
            _row("both", languages=["English", "French"])]
    template, broad = (_load().Hypotheses(rows, ["languages"], broad=flag) for flag in (False, True))
    for hypotheses in (template, broad):
        hypotheses.reject("en")
        hypotheses.reject("fr")
    # The templates give a hidden language one label, both now ruled out; any two labels keep the pair open.
    assert template.best() is None
    assert broad.best() == "both"


def test_broad_year_hypotheses_keep_both_bounds_after_a_stated_bound():
    rows = [_row("y1960", year=1960), _row("y1990", year=1990), _row("y2010", year=2010)]
    template, broad = (_load().Hypotheses(rows, ["publication_year"], stated_min=1950, broad=flag)
                       for flag in (False, True))
    for hypotheses in (template, broad):
        hypotheses.reject("y1960")
    # The templates leave only the open upper bound, which 1960 met for every year; a lower bound stays possible.
    assert template.best() is None
    assert broad.best() in {"y1990", "y2010"}


def test_broad_hypotheses_pair_two_values_with_a_second_refused_field():
    rows = [_row("x", ["X"]), _row("y", ["Y"]), _row("xy", ["X", "Y"])]
    template, broad = (_load().Hypotheses(rows, ["genres", "languages"], broad=flag) for flag in (False, True))
    for hypotheses in (template, broad):
        hypotheses.reject("x")
        hypotheses.reject("y")
    # Two refused fields hold one predicate each under the templates; "X and Y, in English" is still open.
    assert template.best() is None
    assert broad.best() == "xy"
