import tau_rec.catalog.books_tasks as books_tasks
from tau_rec.catalog.books_tasks import (
    RETRIEVER_INDEPENDENT_SAMPLING,
    SOLUTION_BANDS,
    WIKIDATA_TASK_FORMATS,
    WIKIDATA_TEMPLATES,
    MAX_FAMILY_SOLUTION_JACCARD,
    BookTaskGenerator,
    Candidate,
    Predicate,
    TaskDraft,
    _tag_constraints_matched,
    _solution_jaccard_masks,
    matched_family_manifest,
    validate_matched_families,
)
from tau_rec.data_model.catalog import Book, Catalog
from tau_rec.data_model.task import RevealTag


def _constraint_count(template):
    return sum(2 if slot == "G2" else 1 for slot in template)


def test_solution_bands():
    assert SOLUTION_BANDS == {"simple": (100, 500), "medium": (20, 100), "complex": (3, 20)}


def test_wikidata_templates_use_only_well_covered_factual_fields():
    allowed = {"simple": {2}, "medium": {3, 4}, "complex": {5, 6}}
    for complexity, templates in WIKIDATA_TEMPLATES.items():
        assert {_constraint_count(template) for template in templates} <= allowed[complexity]
    assert all(
        slot in {"G1", "G2", *books_tasks.SLOT_KINDS}
        for templates in WIKIDATA_TEMPLATES.values()
        for template in templates
        for slot in template
    )


def test_solution_set_diversity_uses_jaccard_and_a_strict_ceiling():
    assert _solution_jaccard_masks(0b0011, 0b0110) == 1 / 3
    assert _solution_jaccard_masks(0b0011, 0b0011) == 1.0
    assert _solution_jaccard_masks(0, 0) == 0.0
    assert MAX_FAMILY_SOLUTION_JACCARD == 0.95


def _catalog_for_controlled_generation() -> Catalog:
    items = []
    for index in range(12):
        genres = ["Mystery"] if index < 9 else ["Fantasy"]
        if index in {0, 1, 2}:
            genres.append("Fantasy")
        items.append(Book(
            id=f"book_{index:02d}",
            title=f"Book {index}",
            authors=["A. Author"],
            publication_year=2000 + index,
            pages=200 + index * 25,
            genres=genres,
            description="A catalog description.",
            publisher="Publisher",
            rating=3.5 + index / 20,
            vote_count=index + 1,
            formats=["paperback" if index % 2 else "hardcover"],
        ))
    return Catalog(items=items)


def _catalog_for_wikidata_profile() -> Catalog:
    base = _catalog_for_controlled_generation().items
    return Catalog(items=[
        base[index % len(base)].model_copy(update={
            "id": f"wd_{index:02d}",
            "pages": None,
            "rating": None,
            "vote_count": None,
            "languages": ["English" if index < 20 else "French"],
            "countries": ["United States" if index % 2 else "Canada"],
            "formats": [WIKIDATA_TASK_FORMATS[index % len(WIKIDATA_TASK_FORMATS)]],
            "sitelink_count": index,
        })
        for index in range(40)
    ])


def test_wikidata_task_profile_avoids_sparse_numeric_fields():
    generator = BookTaskGenerator(
        _catalog_for_wikidata_profile(),
        core_size=40,
        sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
        task_profile=books_tasks.WIKIDATA_V2_TASK_PROFILE,
    )

    assert generator.constraint_fields == [
        "genres", "languages", "countries", "publication_year",
    ]
    assert generator.by_kind["language"]
    assert generator.by_kind["country"]
    assert {predicate.value for predicate in generator.by_kind["format"]} == set(
        WIKIDATA_TASK_FORMATS
    )
    assert all(
        predicate.mask.bit_count() >= 20
        for kind in ("language", "country")
        for predicate in generator.by_kind[kind]
    )
    assert generator.popularity["wd_00"] == 1
    assert generator.popularity["wd_39"] == 4


def test_the_wikidata_v2_profile_is_the_default():
    generator = BookTaskGenerator(_catalog_for_wikidata_profile(), core_size=40)
    assert generator.task_profile == books_tasks.WIKIDATA_V2_TASK_PROFILE
    assert generator.sampling_mode == RETRIEVER_INDEPENDENT_SAMPLING


def test_wikidata_satisfiable_predicates_must_each_reduce_the_answer_set():
    generator = BookTaskGenerator(
        _catalog_for_wikidata_profile(),
        core_size=40,
        sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
    )
    first = Predicate("genres", "contains", "Mystery", "genre", 0b0111)
    second = Predicate("languages", "contains", "English", "language", 0b1110)
    assert generator._every_predicate_is_material((first, second), 0b0110)
    redundant = Predicate("countries", "contains", "Canada", "country", 0b0110)
    assert not generator._every_predicate_is_material(
        (first, second, redundant), 0b0110,
    )


def test_wikidata_format_policy_can_pin_each_synthetic_format():
    generator = BookTaskGenerator(
        _catalog_for_wikidata_profile(),
        core_size=40,
        sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
    )
    for expected in WIKIDATA_TASK_FORMATS:
        value, mask = generator._chosen_format_option(
            generator.all_mask, required=expected,
        )
        assert value == expected
        assert mask.bit_count() == 10


def test_retriever_independent_constructor_never_builds_search_index(monkeypatch):
    class SearchMustNotBeConstructed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("retrieval leaked into controlled task construction")

    monkeypatch.setattr(books_tasks, "CatalogSearch", SearchMustNotBeConstructed)
    generator = BookTaskGenerator(
        _catalog_for_controlled_generation(),
        core_size=12,
        sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
    )

    assert generator.search is None
    assert generator.anchor_pools[1]
    assert generator.anchor_pools[2]


def test_matched_materialization_changes_only_reveal_state():
    generator = BookTaskGenerator(
        _catalog_for_controlled_generation(),
        core_size=12,
        sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
    )
    year = generator.by_kind["year_lower"][0]
    genre = next(
        predicate for predicate in generator.by_kind["genre"]
        if predicate.value == "Fantasy"
    )
    candidate = Candidate(
        predicates=(genre, year),
        solution_mask=genre.mask & year.mask,
        query="",
        retrieval_rank=None,
        anchor_id="book_10",
        popularity_quartile=4,
    )
    drafts = [
        TaskDraft(
            complexity="simple",
            reveal=reveal,
            candidate=candidate,
            is_nvr=False,
            family_id="books_simple_001",
        )
        for reveal in books_tasks.REVEALS
    ]

    generator.materialize_matched(
        drafts,
        format_families_per_complexity=0,
        history_families_per_complexity=0,
    )

    assert validate_matched_families(drafts) == []
    assert len({tuple(draft.solution_ids or []) for draft in drafts}) == 1
    manifest = matched_family_manifest(drafts)
    assert manifest["family_count"] == 1
    assert set(manifest["families"][0]["task_ids"]) == set(books_tasks.REVEALS)

    assert drafts[0].task is not None
    drafts[0].task.persona += " This must be detected."
    assert any(
        "non-reveal task state varies" in error
        for error in validate_matched_families(drafts)
    )


def test_matched_reveal_assignment_is_not_constraint_position_based():
    predicates = tuple(
        Predicate("publication_year", ">=", 1900 + index, f"y{index}", 1)
        for index in range(5)
    )
    volunteer_positions = set()
    for family_number in range(100):
        tagged = _tag_constraints_matched(
            predicates,
            "mixed",
            family_id=f"family_{family_number:03d}",
            seed=20260926,
        )
        positions = [
            index for index, constraint in enumerate(tagged)
            if constraint.reveal == RevealTag.VOLUNTEER
        ]
        assert len(positions) == 1
        volunteer_positions.update(positions)
    assert volunteer_positions == set(range(len(predicates)))


# --- books-wikidata-v2 generator rules (task profile "wikidata-v2") -----------

def _v2_book(index: int, **fields) -> Book:
    base = {
        "id": f"wd_{index:03d}", "title": f"Work {index}", "authors": ["A. Author"],
        "publication_year": 1950 + index, "pages": None,
        "genres": [["Fantasy", "Mystery"], ["Mystery"], ["Fantasy", "Romance"]][index % 3],
        "description": "a novel", "publisher": "P", "rating": None, "vote_count": None,
        "formats": [WIKIDATA_TASK_FORMATS[index % len(WIKIDATA_TASK_FORMATS)]],
        "languages": ["English"], "countries": ["United States"], "sitelink_count": index,
    }
    base.update(fields)
    return Book(**base)


def test_v2_predicates_use_only_base_language_and_country_values():
    items = [
        _v2_book(i, languages=["British English", "English"], countries=["German Empire", "Germany"])
        for i in range(25)
    ] + [_v2_book(25 + i) for i in range(15)] + [_v2_book(40 + i, languages=["French"]) for i in range(10)]
    catalog = Catalog(items=items)
    v2 = BookTaskGenerator(catalog, core_size=50, sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
                           task_profile=books_tasks.WIKIDATA_V2_TASK_PROFILE)
    v2_values = {p.value for kind in ("language", "country") for p in v2.by_kind[kind]}
    # 25 works carry each dialect or historical label: frequent enough for a predicate.
    assert books_tasks.WIKIDATA_CATEGORICAL_MIN_COUNT <= 25
    assert not {"British English", "German Empire"} & v2_values
    assert {"English", "Germany", "United States"} <= v2_values
    assert v2.generator_version == books_tasks.WIKIDATA_V2_GENERATOR_VERSION


def test_v2_forbids_overlapping_genre_pairs():
    def genre(value):
        return Predicate("genres", "contains", value, "genre", 1)

    assert books_tasks.has_overlapping_genre_pair((genre("Nonfiction"), genre("Memoir")))
    assert books_tasks.has_overlapping_genre_pair((genre("Biography"), genre("Memoir")))
    assert books_tasks.has_overlapping_genre_pair((genre("History"), genre("Nonfiction")))
    assert not books_tasks.has_overlapping_genre_pair((genre("Fantasy"), genre("Mystery")))


def test_v2_nvr_rejects_near_misses_that_hinge_on_a_fragile_label():
    # Graphic Novel + French is impossible only because a French graphic novel is
    # labelled Nonfiction (its description says "graphic novel"), and a French
    # history book is impossible only because its country label is missing.
    items = [
        _v2_book(0, genres=["Nonfiction"], languages=["French"], description="graphic novel by Guy Delisle"),
        _v2_book(1, genres=["Graphic Novel"], languages=["English"]),
        _v2_book(2, genres=["Mystery", "Thriller"], countries=["Norway"]),
        _v2_book(3, genres=["Crime"], countries=[]),
    ] + [_v2_book(10 + i) for i in range(30)]
    generator = BookTaskGenerator(Catalog(items=items), core_size=34,
                                  sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
                                  task_profile=books_tasks.WIKIDATA_V2_TASK_PROFILE)

    def p(field, value, kind):
        return Predicate(field, "contains", value, kind, generator._mask(field, "contains", value))

    graphic_french = (p("genres", "Graphic Novel", "genre"), p("languages", "French", "language"))
    problems = generator.nvr_fragility(graphic_french)
    assert any("description" in reason for reason in problems)
    crime_norway = (p("genres", "Crime", "genre"), p("countries", "Norway", "country"))
    problems = generator.nvr_fragility(crime_norway)
    assert any("neighbor genre" in reason for reason in problems)       # Mystery/Thriller ~ Crime
    assert any("no country label" in reason for reason in problems)     # item 3 might be Norwegian


def test_v2_history_decoys_each_fail_exactly_one_hidden_predicate():
    items = [_v2_book(i, countries=["United States"], publication_year=2000 + i % 10) for i in range(40)]
    items += [_v2_book(40 + i, countries=["Canada"], publication_year=2005) for i in range(5)]
    generator = BookTaskGenerator(Catalog(items=items), core_size=45,
                                  sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
                                  task_profile=books_tasks.WIKIDATA_V2_TASK_PROFILE)
    def p(field, op, value, kind):
        return Predicate(field, op, value, kind, generator._mask(field, op, value))

    mystery = p("genres", "contains", "Mystery", "genre")
    us = p("countries", "contains", "United States", "country")
    year = p("publication_year", ">=", 2005, "year_lower")
    predicates = (mystery, us, year)
    solution_mask = predicates[0].mask & predicates[1].mask & predicates[2].mask
    read_ids = generator.history_read_ids("books_simple_001", predicates, hidden_indices=[1, 2],
                                          eligible_mask=solution_mask)
    by_id = {item.id: item for item in items}
    trap, decoys = read_ids[0], read_ids[1:]
    assert all(p.as_constraint().evaluate(by_id[trap]) for p in predicates)
    for hidden_index in (1, 2):
        assert any(
            not predicates[hidden_index].as_constraint().evaluate(by_id[decoy])
            and all(predicates[i].as_constraint().evaluate(by_id[decoy]) for i in range(3) if i != hidden_index)
            for decoy in decoys
        ), f"no decoy fails only hidden predicate {hidden_index}"


def test_v2_hidden_selectivity_measures_what_the_hidden_facts_exclude():
    items = [_v2_book(i, countries=["United States"] if i < 30 else ["Canada"]) for i in range(40)]
    generator = BookTaskGenerator(Catalog(items=items), core_size=40,
                                  sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
                                  task_profile=books_tasks.WIKIDATA_V2_TASK_PROFILE)
    mystery = Predicate("genres", "contains", "Mystery", "genre", generator.all_mask)
    canada = generator._predicate("countries", "contains", "Canada", "country")
    us = generator._predicate("countries", "contains", "United States", "country")
    # Hiding Canada excludes 30/40 revealed-consistent works; hiding the US excludes 10/40.
    assert abs(generator.hidden_selectivity((mystery, canada), [1]) - 0.75) < 1e-9
    assert abs(generator.hidden_selectivity((mystery, us), [1]) - 0.25) < 1e-9


def test_v2_hidden_choice_prefers_selective_subsets_without_leaks():
    items = [_v2_book(i, countries=["United States"] if i < 30 else ["Canada"],
                      publication_year=1990 + i % 20) for i in range(40)]
    generator = BookTaskGenerator(Catalog(items=items), core_size=40,
                                  sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
                                  task_profile=books_tasks.WIKIDATA_V2_TASK_PROFILE)

    def p(field, op, value, kind):
        return Predicate(field, op, value, kind, generator._mask(field, op, value))

    predicates = (
        p("genres", "contains", "Mystery", "genre"),
        p("countries", "contains", "Canada", "country"),
        p("publication_year", ">=", 2000, "year_lower"),
        p("publication_year", "<=", 2005, "year_upper"),
    )
    hidden, selectivity = generator.v2_hidden_indices("books_medium_001", predicates)
    order = books_tasks.matched_reveal_order(len(predicates), family_id="books_medium_001", seed=generator.seed)
    assert order[0] not in hidden                       # the opening fact stays revealed
    assert len(hidden) == max(1, len(predicates) // 2)
    assert selectivity == generator.hidden_selectivity(predicates, hidden)
    best = max(
        generator.hidden_selectivity(predicates, subset)
        for subset in __import__("itertools").combinations(order[1:], len(hidden))
    )
    assert selectivity >= min(books_tasks.HIDDEN_MIN_SELECTIVITY, best)


def test_v2_admits_only_candidates_with_a_selective_hidden_subset_for_any_opening():
    items = [_v2_book(i, countries=["United States"] if i < 36 else ["Canada"]) for i in range(40)]
    generator = BookTaskGenerator(Catalog(items=items), core_size=40,
                                  sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
                                  task_profile=books_tasks.WIKIDATA_V2_TASK_PROFILE)

    def p(field, op, value, kind):
        return Predicate(field, op, value, kind, generator._mask(field, op, value))

    mystery = p("genres", "contains", "Mystery", "genre")
    # With the US as the only other fact, hiding it excludes ~10% of works.
    weak = (mystery, p("countries", "contains", "United States", "country"))
    strong = (mystery, p("countries", "contains", "Canada", "country"))
    assert not generator.selective_hidden_possible(weak)
    assert not generator.selective_hidden_possible(strong)  # opening on Canada leaves only Mystery to hide
    # Independent facts: Canada on every fourth work, years >= 1980 on the last quarter.
    items = [_v2_book(i, countries=["Canada"] if i % 4 == 0 else ["United States"]) for i in range(40)]
    generator = BookTaskGenerator(Catalog(items=items), core_size=40,
                                  sampling_mode=RETRIEVER_INDEPENDENT_SAMPLING,
                                  task_profile=books_tasks.WIKIDATA_V2_TASK_PROFILE)
    assert generator.selective_hidden_possible((
        p("genres", "contains", "Mystery", "genre"),
        p("countries", "contains", "Canada", "country"),
        p("publication_year", ">=", 1980, "year_lower"),
    ))
