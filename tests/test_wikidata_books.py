from tau_rec.catalog.wikidata_books import (
    WikidataBookProfile,
    booklike_reason,
    canonical_genres,
    deduplicate_profiles,
    qid_from_value,
    sparql_literal,
    stable_order_key,
    task_viability,
    year_from_value,
)


def _profile(index: int, *, genre: str, year: int, language: str = "Q1860"):
    return WikidataBookProfile(
        item_id=f"Q{index + 1}",
        title=f"Book {index}",
        author_ids=(f"Q{100_000 + index}",),
        year=year,
        genre_ids=(genre,),
        language_ids=(language,),
        country_ids=("Q30",),
    )


def test_wikidata_value_parsing():
    assert qid_from_value("<http://www.wikidata.org/entity/Q123>") == "Q123"
    assert qid_from_value("Q42") == "Q42"
    assert qid_from_value("not-an-entity") is None
    assert sparql_literal('"A \\"quoted\\" title"@en') == 'A "quoted" title'
    assert year_from_value('"+1984-00-00T00:00:00Z"^^<xsd:dateTime>') == 1984


def test_genre_mapping_is_deterministic_and_can_map_to_parents():
    assert canonical_genres(["Q603291", "Q24925", "Q603291"]) == (
        "Fantasy",
        "Historical Fiction",
        "Science Fiction",
    )


def test_deduplication_uses_normalized_title_and_author_set():
    first = WikidataBookProfile(
        item_id="Q20",
        title="The Book!",
        author_ids=("Q2",),
        year=2000,
        genre_ids=("Q24925",),
        language_ids=("Q1860",),
    )
    second = WikidataBookProfile(
        item_id="Q10",
        title="the book",
        author_ids=("Q2",),
        year=2001,
        genre_ids=("Q24925",),
        language_ids=("Q1860",),
    )
    unique, stats = deduplicate_profiles([first, second])
    assert [row.item_id for row in unique] == ["Q10"]
    assert stats["duplicate_records_removed"] == 1


def test_v2_genre_map_drops_two_overreaching_rules():
    from tau_rec.catalog.wikidata_books import WIKIDATA_GENRE_MAP_V2

    gothic, dystopian, horror = "Q20669641", "Q15062348", "Q16575965"
    assert canonical_genres([gothic]) == ("Horror",)
    assert canonical_genres([dystopian]) == ("Science Fiction",)
    assert canonical_genres([gothic], genre_map=WIKIDATA_GENRE_MAP_V2) == ()
    assert canonical_genres([dystopian], genre_map=WIKIDATA_GENRE_MAP_V2) == ()
    assert canonical_genres([gothic, horror], genre_map=WIKIDATA_GENRE_MAP_V2) == ("Horror",)


def test_v2_dedupe_key_folds_part_and_volume_editions():
    from tau_rec.catalog.wikidata_books import dedupe_key_v2

    whole = WikidataBookProfile(
        item_id="Q11", title="Dragonfly in Amber", author_ids=("Q2",), year=1992,
        genre_ids=("Q24925",), language_ids=("Q1860",),
    )
    part = WikidataBookProfile(
        item_id="Q99", title="Dragonfly in Amber (part II)", author_ids=("Q2",), year=1992,
        genre_ids=("Q24925",), language_ids=("Q1860",),
    )
    other = WikidataBookProfile(
        item_id="Q50", title="Dragonfly in Amber", author_ids=("Q3",), year=1992,
        genre_ids=("Q24925",), language_ids=("Q1860",),
    )
    unique, stats = deduplicate_profiles([whole, part, other], key=dedupe_key_v2)
    assert sorted(row.item_id for row in unique) == ["Q11", "Q50"]
    assert stats["duplicate_records_removed"] == 1
    unique_default, _ = deduplicate_profiles([whole, part, other])
    assert len(unique_default) == 3


def test_v2_derived_labels_add_modern_parents_but_not_for_multinational_states():
    from tau_rec.catalog.taxonomy import contains_label, derived_labels_v2

    assert derived_labels_v2("countries", ["German Empire"]) == ["German Empire", "Germany"]
    assert derived_labels_v2("countries", ["People's Republic of China"]) == ["China", "People's Republic of China"]
    assert derived_labels_v2("countries", ["Soviet Union"]) == ["Soviet Union"]
    assert derived_labels_v2("countries", ["England"]) == ["England", "United Kingdom"]
    assert derived_labels_v2("languages", ["Classical Chinese"]) == ["Chinese", "Classical Chinese"]
    # Parents are derived into the catalog, not matched at runtime.
    assert not contains_label("countries", ["German Empire"], "Germany")


def test_stable_order_key_depends_on_seed_and_id():
    assert stable_order_key("Q1", 7) == stable_order_key("Q1", 7)
    assert stable_order_key("Q1", 7) != stable_order_key("Q1", 8)


def test_task_viability_is_deterministic():
    profiles = [
        _profile(i, genre="Q24925" if i % 2 else "Q132311", year=1800 + i % 220)
        for i in range(600)
    ]
    first = task_viability(profiles, catalog_size=500, core_size=250, seed=11)
    second = task_viability(reversed(profiles), catalog_size=500, core_size=250, seed=11)
    assert first == second
    assert first["catalog_size"] == 500
    assert set(first["bands"]) == {"simple", "medium", "complex"}


def test_booklike_gate_rejects_individual_short_work():
    short_story = WikidataBookProfile(
        item_id="Q1",
        title="A Story",
        author_ids=("Q2",),
        year=2000,
        genre_ids=("Q24925",),
        language_ids=("Q1860",),
        form_ids=("Q49084",),
        description="short story by Example Author",
    )
    assert booklike_reason(short_story) is None


def test_booklike_gate_rejects_series_and_edition_description_fallbacks():
    for index, description in enumerate((
        "science fiction book series",
        "Book series by Example Author",
        "edition of academic book published in 2021",
    ), 1):
        profile = WikidataBookProfile(
            item_id=f"Q{index}",
            title=f"Not a work {index}",
            author_ids=("Q20",),
            year=2000,
            genre_ids=("Q24925",),
            language_ids=("Q1860",),
            description=description,
        )
        assert booklike_reason(profile) is None


def test_booklike_gate_accepts_collection_and_description_fallback():
    collection = WikidataBookProfile(
        item_id="Q1",
        title="Collected Stories",
        author_ids=("Q2",),
        year=2000,
        genre_ids=("Q24925",),
        language_ids=("Q1860",),
        form_ids=("Q1279564",),
        description="short story collection",
    )
    nonfiction = WikidataBookProfile(
        item_id="Q3",
        title="A History",
        author_ids=("Q4",),
        year=2001,
        genre_ids=("Q213051",),
        language_ids=("Q1860",),
        description="2001 non-fiction work by Example Author",
    )
    assert booklike_reason(collection) == "structured_book_form"
    assert booklike_reason(nonfiction) == "description_fallback"


def test_v2_overrides_support_exclusions_and_batch_rules():
    from tau_rec.catalog.wikidata_books import apply_label_overrides

    overrides = {
        "overrides": {"Q1": {"exclude": True}, "Q2": {"remove": {"genres": ["Horror"]}, "publication_year": 1900}},
        "exclusion_rules": [
            {"field": "description", "equals": "Taxation book"},
            {"field": "title", "startswith": "The Art of training"},
        ],
    }
    base = {"genres": ["Horror", "Fantasy"], "languages": ["English"], "countries": [], "publication_year": 1950}
    assert apply_label_overrides("Q1", dict(base), overrides, title="T", description="d") is None
    fixed = apply_label_overrides("Q2", dict(base), overrides, title="T", description="d")
    assert fixed["genres"] == ["Fantasy"] and fixed["publication_year"] == 1900
    assert apply_label_overrides("Q3", dict(base), overrides, title="T", description="Taxation book") is None
    assert apply_label_overrides("Q4", dict(base), overrides, title="The Art of training and development", description="d") is None
    assert apply_label_overrides("Q5", dict(base), overrides, title="T", description="d") == base
    no_genre = apply_label_overrides("Q6", {**base, "genres": ["Horror"]}, {"overrides": {"Q6": {"remove": {"genres": ["Horror"]}}}}, title="T", description="d")
    assert no_genre is None


def test_v2_overrides_can_correct_title_and_description():
    import pytest

    from tau_rec.catalog.wikidata_books import apply_label_overrides

    base = {"title": "Ekhan", "description": "1974 novel", "genres": ["Poetry"], "languages": ["Assamese"],
            "countries": [], "publication_year": 2019}
    overrides = {"overrides": {"Q7": {"title": "Ekhan", "set": {"title": "Ekhani", "description": "1969 novel"}}}}
    fixed = apply_label_overrides("Q7", base, overrides, title="Ekhan", description="1974 novel")
    assert (fixed["title"], fixed["description"]) == ("Ekhani", "1969 novel")
    assert base["title"] == "Ekhan"
    with pytest.raises(ValueError, match="titel"):
        apply_label_overrides("Q7", base, {"overrides": {"Q7": {"set": {"titel": "x"}}}}, title="Ekhan", description="d")
