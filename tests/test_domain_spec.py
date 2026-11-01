"""DomainSpec is the only place domain words live.

tests/test_goldens.py proves the MOVIES spec renders the harness exactly as
it was hardcoded. This file proves the other half: the BOOKS spec
reaches every model-facing string, gates on its own rating set and
availability field, and leaves no stray "movie" in shared harness code.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from tau_rec.agents.litellm_agent import build_agent_system_prompt
from tau_rec.catalog.search import CatalogSearch
from tau_rec.data_model.catalog import Book, Catalog
from tau_rec.data_model.task import Constraint, RevealTag, Task, TaskConstraint
from tau_rec.domain import BOOKS, MOVIES, SPECS, DomainSpec, get_spec
from tau_rec.environment.tools import ToolKit, memory_tool_definitions, tool_definitions
from tau_rec.orchestrator.orchestrator import Orchestrator
from tau_rec.simulator.user_simulator import build_simulator_prompt

MOVIE_WORDS = ("movie", "streaming", "watch history", "nc-17")

# Vocabulary the harness uses only in validator messages and CLI output. It
# never reaches a model, so the reaches-a-model check skips it on purpose.
OPERATOR_FACING = {
    "id", "id_prefix", "availability_field", "item_plural", "consumed_adj",
    "history_key", "task_availability_field", "memory_availability_argument",
    "tool_availability_argument",
}
# Vocabulary only the age-gate tool shows, which a domain with no age policy
# does not get.
AGE_GATE_FIELDS = {"restricted_examples", "rating_examples"}


def _book(i: str, formats: list[str], rating: str = "General") -> Book:
    return Book(id=i, title=f"Book {i}", authors=["A. Writer"], publication_year=2015, pages=300,
                genres=["Mystery"], description="a twisty mystery", publisher="P", rating=4.1,
                vote_count=500, formats=formats, content_rating=rating)


def _task(services: list[str], genre: str) -> Task:
    return Task(
        id="t1", persona="p", complexity="simple", reveal_difficulty="volunteer",
        constraints=[TaskConstraint(constraint=Constraint(field="genres", op="contains", value=genre),
                                    reveal=RevealTag.VOLUNTEER)],
        policy_flags=["availability", "age_restricted"], user_services=services,
        user_history={"user_1": {"watched": [], "ratings": {}}},
    )


@pytest.mark.parametrize("spec", [BOOKS], ids=lambda s: s.id)
def test_every_spec_string_reaches_a_model_and_no_movie_word_survives(spec: DomainSpec):
    task = _task(["x"], "Mystery")
    rendered = "\n".join([
        build_agent_system_prompt("POLICY", "user_1", spec=spec),
        build_agent_system_prompt("POLICY", "user_1", no_tools=True, spec=spec),
        build_simulator_prompt(task, spec),
        build_simulator_prompt(_task([], "Mystery"), spec),
        json.dumps(tool_definitions(spec)),
        spec.greeting,
    ])
    for f in dataclasses.fields(spec):
        value = getattr(spec, f.name)
        if not spec.age_policy_enabled and f.name in AGE_GATE_FIELDS:
            continue
        if isinstance(value, str) and f.name not in OPERATOR_FACING:
            probe = value.split("{")[0].strip() or value   # templates: check their literal head
            assert probe in rendered, f"{spec.id}.{f.name} never reaches a model"
    low = rendered.lower()
    for word in MOVIE_WORDS:
        assert word not in low, f"{word!r} leaked into the {spec.id} rendering"


def test_books_tools_check_formats_and_have_no_age_gate():
    # Three docs: with two, BM25's IDF for a token unique to one doc is
    # log((2-1+0.5)/(1+0.5)) = 0 and every search returns nothing.
    catalog = Catalog(items=[_book("wd_1", ["ebook", "paperback"], rating="Adult"),
                             _book("wd_2", ["hardcover"]), _book("wd_3", ["ebook", "audiobook"])])
    kit = ToolKit(catalog, CatalogSearch(catalog, BOOKS), {}, spec=BOOKS)

    assert json.loads(kit.call("check_content_preference", {"content_rating": "Adult"})) == {
        "error": "Unknown tool: check_content_preference"
    }
    avail = json.loads(kit.call("check_availability",
                                {"item_id": "wd_1", "formats": ["ebook", "Kindle", "audiobook"]}))
    assert avail == {"ebook": True, "Kindle": True, "audiobook": False}
    assert json.loads(kit.call("get_metadata", {"item_id": "nope"})) == {"error": "Book not found: nope"}
    # A token present in only one of two docs, so BM25's IDF is positive; a
    # term in every doc scores zero and is filtered, which is not a bug here.
    hits = json.loads(kit.call("search_catalog", {"query": "wd_2"}))
    assert [h["id"] for h in hits] == ["wd_2"]
    assert set(hits[0]) == set(BOOKS.search_result_fields)

    assert "check_content_preference" not in [t["function"]["name"] for t in tool_definitions(BOOKS)]


def test_books_structured_retrieval_filters_full_catalog_deterministically():
    books = [
        _book("wd_1", ["ebook"]).model_copy(update={
            "vote_count": 20, "publication_year": 2020, "genres": ["Mystery", "Fantasy"],
        }),
        _book("wd_2", ["paperback"]).model_copy(update={
            "vote_count": 200, "publication_year": 2021, "genres": ["Mystery", "Fantasy"],
        }),
        _book("wd_3", ["ebook"]).model_copy(update={
            "vote_count": 500, "publication_year": 2010, "genres": ["Mystery"],
        }),
    ]
    catalog = Catalog(items=books)
    kit = ToolKit(
        catalog,
        None,
        {},
        spec=BOOKS,
        retrieval_mode="structured",
    )

    schema = kit.tool_definitions()[0]["function"]
    assert schema["name"] == "search_catalog"
    assert "query" not in schema["parameters"]["properties"]
    result = json.loads(kit.call("search_catalog", {
        "genres": ["Mystery", "Fantasy"],
        "min_publication_year": 2015,
        "formats": ["ebook"],
    }))
    assert [row["id"] for row in result] == ["wd_1"]
    newest = json.loads(kit.call("search_catalog", {"genres": ["Mystery"], "sort_by": "newest"}))
    assert [row["id"] for row in newest] == ["wd_2", "wd_1", "wd_3"]

    invalid = json.loads(kit.call("search_catalog", {
        "min_publication_year": 2021, "max_publication_year": 2020,
    }))
    assert "error" in invalid


def test_books_structured_retrieval_supports_public_fields():
    books = [
        _book("wd_1", ["ebook"]).model_copy(update={
            "languages": ["French"], "countries": ["France"],
            "pages": None, "rating": None, "vote_count": None, "sitelink_count": 40,
        }),
        _book("wd_2", ["paperback"]).model_copy(update={
            "languages": ["English"], "countries": ["Canada"],
            "pages": None, "rating": None, "vote_count": None, "sitelink_count": 80,
        }),
    ]
    kit = ToolKit(
        Catalog(items=books), None, {}, spec=BOOKS, retrieval_mode="structured",
    )

    properties = kit.tool_definitions()[0]["function"]["parameters"]["properties"]
    assert {"languages", "countries"} <= set(properties)
    result = json.loads(kit.call("search_catalog", {
        "languages": ["french"], "countries": ["FRANCE"],
    }))
    assert [row["id"] for row in result] == ["wd_1"]
    assert result[0]["languages"] == ["French"]
    assert result[0]["countries"] == ["France"]
    popular = json.loads(kit.call("search_catalog", {"sort_by": "popularity"}))
    assert [row["id"] for row in popular] == ["wd_2", "wd_1"]


def test_wikidata_core_structured_profile_excludes_sparse_numeric_fields():
    books = [
        _book("wd_1", ["ebook"]).model_copy(update={
            "languages": ["French"], "countries": ["France"],
            "pages": None, "rating": None, "publication_year": 2020,
        }),
        _book("wd_2", ["paperback"]).model_copy(update={
            "languages": ["English"], "countries": ["Canada"],
            "pages": None, "rating": None, "publication_year": 2021,
        }),
    ]
    kit = ToolKit(
        Catalog(items=books), None, {}, spec=BOOKS,
        retrieval_mode="structured",
        structured_profile="books-wikidata-core-v3",
    )
    properties = kit.tool_definitions()[0]["function"]["parameters"]["properties"]
    assert set(properties) == {
        "genres", "languages", "countries", "min_publication_year",
        "max_publication_year", "formats", "sort_by", "offset",
    }
    assert properties["sort_by"]["enum"] == ["popularity", "newest"]
    assert len(json.loads(kit.call("search_catalog", {}))) == 2
    error = json.loads(kit.call("search_catalog", {"author": "A. Writer"}))
    assert "Unknown structured filter" in error["error"]


def test_books_structured_retrieval_prompt_matches_tool_contract():
    prompt = build_agent_system_prompt(
        "POLICY", "user_1", spec=BOOKS, retrieval_mode="structured"
    )
    assert "typed metadata" in prompt
    assert "min/max publication year" in prompt
    with pytest.raises(ValueError, match="structured retrieval profile"):
        build_agent_system_prompt(
            "POLICY", "user_1", spec=BOOKS, retrieval_mode="structured",
            structured_profile="no-such-profile",
        )


def test_books_native_history_and_memory_format_names():
    catalog = Catalog(items=[_book("wd_1", ["ebook"])])
    kit = ToolKit(
        catalog, CatalogSearch(catalog, BOOKS),
        {"user_1": {"read": ["wd_1"], "ratings": {}}}, spec=BOOKS,
    )
    history = json.loads(kit.call("get_user_history", {"user_id": "user_1"}))
    assert history["read"] == [{"id": "wd_1", "title": "Book wd_1"}]
    assert "watched" not in history
    props = memory_tool_definitions(BOOKS)[0]["function"]["parameters"]["properties"]
    assert set(props) == {"title", "year", "format"}
    availability = next(t for t in tool_definitions(BOOKS) if t["function"]["name"] == "check_availability")
    assert set(availability["function"]["parameters"]["properties"]) == {"item_id", "formats"}


def test_books_simulator_describes_source_year_semantics():
    task = Task(
        id="t-year", persona="p", complexity="simple", reveal_difficulty="volunteer",
        constraints=[TaskConstraint(
            constraint=Constraint(field="publication_year", op="<=", value=1990),
            reveal=RevealTag.VOLUNTEER,
        )],
    )
    prompt = build_simulator_prompt(task, BOOKS)
    assert "earliest catalog-edition year must be at most 1990" in prompt
    assert "1990 or any earlier value satisfies this requirement" in prompt
    assert "publication_year <= 1990" not in prompt


def test_books_simulator_spells_out_predicate_direction_and_set_semantics():
    task = Task(
        id="t-literal", persona="p", complexity="medium", reveal_difficulty="hidden",
        constraints=[
            TaskConstraint(
                constraint=Constraint(field="publication_year", op=">=", value=1950),
                reveal=RevealTag.HIDDEN,
            ),
            TaskConstraint(
                constraint=Constraint(field="genres", op="contains", value="Science Fiction"),
                reveal=RevealTag.HIDDEN,
            ),
        ],
    )
    prompt = build_simulator_prompt(task, BOOKS)
    assert "earliest catalog-edition year must be at least 1950" in prompt
    assert "1950 or any later value satisfies this requirement" in prompt
    assert 'genre list must include "Science Fiction"' in prompt
    assert "additional values are allowed" in prompt
    assert "The persona is background context only" in prompt
    assert "Do not invent requirements" in prompt


def test_books_simulator_describes_language_and_country_constraints():
    task = Task(
        id="t-origin", persona="p", complexity="simple", reveal_difficulty="volunteer",
        constraints=[
            TaskConstraint(
                constraint=Constraint(field="languages", op="contains", value="French"),
                reveal=RevealTag.VOLUNTEER,
            ),
            TaskConstraint(
                constraint=Constraint(field="countries", op="contains", value="France"),
                reveal=RevealTag.VOLUNTEER,
            ),
        ],
    )
    prompt = build_simulator_prompt(task, BOOKS)
    assert 'language-of-work list must include "French"' in prompt
    assert 'country-of-origin list must include "France"' in prompt


def test_books_constraint_fidelity_guard_does_not_change_movies_prompt():
    assert "## Constraint Fidelity" in build_simulator_prompt(_task([], "Mystery"), BOOKS)
    assert "## Constraint Fidelity" not in build_simulator_prompt(_task([], "Mystery"), MOVIES)


def test_books_agent_prompt_requires_candidate_id_without_changing_other_modes():
    books = build_agent_system_prompt("POLICY", "user_1", spec=BOOKS)
    movies = build_agent_system_prompt("POLICY", "user_1", spec=MOVIES)
    memory = build_agent_system_prompt("POLICY", "user_1", spec=BOOKS, mode="memory")
    assert "[[candidate_id:ID]]" in books
    assert "[[candidate_id:NONE]]" not in books
    assert "recommend(item_id=null)" in books
    assert "[[candidate_id:ID]]" not in movies
    assert "[[candidate_id:ID]]" not in memory


def test_books_parent_labels_apply_to_constraints_and_structured_search():
    book = _book("wd_1", ["ebook"]).model_copy(update={
        "languages": ["American English"],
        "countries": ["England"],
    })
    assert Constraint(
        field="languages", op="contains", value="English"
    ).evaluate(book)
    assert not Constraint(
        field="languages", op="contains", value="British English"
    ).evaluate(book)
    assert Constraint(
        field="countries", op="contains", value="United Kingdom"
    ).evaluate(book)

    catalog = Catalog(items=[book])
    kit = ToolKit(
        catalog,
        CatalogSearch(catalog, BOOKS),
        {},
        spec=BOOKS,
        mode="tools",
        retrieval_mode="structured",
    )
    rows = json.loads(kit.call("search_catalog", {
        "languages": ["English"],
        "countries": ["United Kingdom"],
    }))
    assert [row["id"] for row in rows] == ["wd_1"]


def test_movies_keep_exact_match_availability():
    """movies-v1 scored under exact matching; the alias field is None there."""
    assert MOVIES.service_aliases is None
    assert MOVIES.normalize_service("Amazon Prime") == "Amazon Prime"


def test_orchestrator_greets_in_domain():
    catalog = Catalog(items=[_book("wd_1", ["ebook"])])
    kit = ToolKit(catalog, CatalogSearch(catalog, BOOKS), {}, spec=BOOKS)
    assert Orchestrator(agent=None, simulator=None, toolkit=kit, spec=BOOKS)._greeting == BOOKS.greeting
    assert Orchestrator(agent=None, simulator=None, toolkit=kit)._greeting == MOVIES.greeting


def test_spec_registry():
    assert set(SPECS) == {"movies", "books"}
    assert get_spec("books") is BOOKS
    with pytest.raises(ValueError, match="unknown domain"):
        get_spec("podcasts")
    for spec in SPECS.values():
        assert isinstance(spec.restricted_ratings, tuple), "an ordered tuple, so rendering is deterministic"
        assert all(hasattr(spec.item_model, f) or f in spec.item_model.model_fields
                   for f, _ in spec.search_fields), f"{spec.id} searches a field its model lacks"
        assert spec.availability_field in spec.item_model.model_fields
