from tau_rec.catalog.search import CatalogSearch
from tau_rec.data_model.catalog import Movie, Catalog

def _make_catalog():
    return Catalog(movies=[
        Movie(id="tt001", title="Galactic Storm", release_date="2025-03-01",
              runtime=110, genres=["Sci-Fi", "Action"], overview="An epic space battle.",
              cast=["Chris Nova"], director="Jane Star", rating=7.2, vote_count=80,
              streaming_services=["Netflix"], sponsored=False, content_rating="PG-13"),
        Movie(id="tt002", title="Quiet River", release_date="2025-04-15",
              runtime=95, genres=["Drama"], overview="A slow-burn drama about family.",
              cast=["Sam River"], director="Tom Calm", rating=8.1, vote_count=120,
              streaming_services=["Hulu"], sponsored=False, content_rating="PG"),
        Movie(id="tt003", title="Night Stalker", release_date="2025-05-20",
              runtime=130, genres=["Thriller", "Horror"], overview="A terrifying night hunt.",
              cast=["Dark Actor"], director="Fear Dir", rating=6.5, vote_count=30,
              streaming_services=["Amazon"], sponsored=True, content_rating="R"),
    ])

def test_search_by_keyword():
    catalog = _make_catalog()
    search = CatalogSearch(catalog)
    results = search.search("space battle sci-fi")
    assert len(results) > 0
    assert results[0]["id"] == "tt001"

def test_search_returns_top_k():
    catalog = _make_catalog()
    search = CatalogSearch(catalog)
    results = search.search("movie", top_k=2)
    assert len(results) <= 2

def test_search_result_fields():
    catalog = _make_catalog()
    search = CatalogSearch(catalog)
    results = search.search("drama")
    assert len(results) > 0
    assert "id" in results[0]
    assert "title" in results[0]
    assert "genres" in results[0]
    assert "release_date" in results[0]
    assert "rating" in results[0]
    assert "overview" in results[0]
    assert "cast" not in results[0]


# --------------- Validator tests ---------------

from tau_rec.catalog.validator import CatalogValidator
from tau_rec.data_model.task import Task, TaskConstraint, Constraint, RevealTag


def test_validator_solvable_task():
    catalog = _make_catalog()
    task = Task(
        id="t1",
        constraints=[
            TaskConstraint(constraint=Constraint(field="runtime", op="<=", value=120), reveal=RevealTag.VOLUNTEER),
        ],
        persona="test", complexity="simple", reveal_difficulty="volunteer",
    )
    result = CatalogValidator(catalog).validate_task(task)
    assert result.solvable is True
    assert result.solution_set_size >= 2


def test_validator_unsolvable_task():
    catalog = _make_catalog()
    task = Task(
        id="t2",
        constraints=[
            TaskConstraint(constraint=Constraint(field="runtime", op="<=", value=50), reveal=RevealTag.VOLUNTEER),
        ],
        persona="test", complexity="simple", reveal_difficulty="volunteer",
        no_valid_recommendation=False,
    )
    result = CatalogValidator(catalog).validate_task(task)
    assert result.solvable is False
    assert result.solution_set_size == 0


def test_validator_no_valid_rec_task():
    catalog = _make_catalog()
    task = Task(
        id="t3",
        constraints=[
            TaskConstraint(constraint=Constraint(field="runtime", op="<=", value=50), reveal=RevealTag.VOLUNTEER),
        ],
        persona="test", complexity="simple", reveal_difficulty="volunteer",
        no_valid_recommendation=True,
    )
    result = CatalogValidator(catalog).validate_task(task)
    assert result.solvable is False
    assert result.valid_as_designed is True


def test_validator_complexity_band_violation():
    """Mislabeling a 4-constraint task as 'simple' must fail the band check."""
    catalog = _make_catalog()
    task = Task(
        id="t4",
        constraints=[
            TaskConstraint(constraint=Constraint(field="runtime", op="<=", value=120), reveal=RevealTag.VOLUNTEER),
            TaskConstraint(constraint=Constraint(field="genres", op="contains", value="Drama"), reveal=RevealTag.VOLUNTEER),
            TaskConstraint(constraint=Constraint(field="rating", op=">=", value=6.0), reveal=RevealTag.VOLUNTEER),
            TaskConstraint(constraint=Constraint(field="content_rating", op="==", value="PG"), reveal=RevealTag.VOLUNTEER),
        ],
        persona="test", complexity="simple", reveal_difficulty="volunteer",
    )
    result = CatalogValidator(catalog).validate_task(task)
    assert any("complexity 'simple'" in e for e in result.errors)


def test_validator_complexity_band_nvr_exempt():
    """NVR tasks are exempt from the band check (any constraint count is fine)."""
    catalog = _make_catalog()
    task = Task(
        id="t5",
        constraints=[
            TaskConstraint(constraint=Constraint(field="runtime", op="<=", value=10), reveal=RevealTag.VOLUNTEER),
            TaskConstraint(constraint=Constraint(field="genres", op="contains", value="ZZZ"), reveal=RevealTag.VOLUNTEER),
            TaskConstraint(constraint=Constraint(field="rating", op=">=", value=99.0), reveal=RevealTag.VOLUNTEER),
        ],
        persona="test", complexity="simple", reveal_difficulty="volunteer",
        no_valid_recommendation=True,
    )
    result = CatalogValidator(catalog).validate_task(task)
    assert not any("complexity" in e for e in result.errors)


def test_task_signature_and_grid_summary():
    from tau_rec.catalog.validator import task_signature, grid_summary
    catalog = _make_catalog()
    task = Task(
        id="t6",
        constraints=[
            TaskConstraint(constraint=Constraint(field="runtime", op="<=", value=120), reveal=RevealTag.VOLUNTEER),
            TaskConstraint(constraint=Constraint(field="genres", op="contains", value="Drama"), reveal=RevealTag.HIDDEN),
        ],
        persona="test", complexity="simple", reveal_difficulty="mixed",
        policy_flags=["recommend_tool", "availability"],
    )
    result = CatalogValidator(catalog).validate_task(task)
    sig = task_signature(task, result)
    assert sig["n_constraints"] == 2
    assert sig["n_hidden"] == 1
    assert sig["n_volunteer"] == 1
    assert sig["policy_flags"] == ["availability", "recommend_tool"]  # sorted
    assert sig["complexity"] == "simple"

    grid = grid_summary([task, task])
    assert grid["simple"]["mixed"] == 2
    assert grid["medium"]["volunteer"] == 0


# --------------- Pipeline tests ---------------

from tau_rec.catalog.pipeline import TMDBPipeline, normalize_service_name, normalize_services


def test_normalize_service_name_strips_variants():
    assert normalize_service_name("Netflix") == "Netflix"
    assert normalize_service_name("Netflix Standard with Ads") == "Netflix"
    assert normalize_service_name("Amazon Prime Video with Ads") == "Amazon Prime Video"
    assert normalize_service_name("Starz Apple TV Channel") == "Starz"
    assert normalize_service_name("Starz Amazon Channel") == "Starz"
    assert normalize_service_name("Starz Roku Premium Channel") == "Starz"
    assert normalize_service_name("AMC Plus Apple TV Channel") == "AMC+"
    assert normalize_service_name("AMC+ Amazon Channel") == "AMC+"
    assert normalize_service_name("Peacock Premium") == "Peacock"
    assert normalize_service_name("Peacock Premium Plus") == "Peacock"
    assert normalize_service_name("Paramount Plus Essential") == "Paramount+"
    assert normalize_service_name("Paramount Plus Premium") == "Paramount+"
    assert normalize_service_name("Disney Plus") == "Disney+"
    assert normalize_service_name("Apple TV") == "Apple TV+"
    assert normalize_service_name("MGM Plus") == "MGM+"


def test_normalize_services_dedupes():
    out = normalize_services([
        "Netflix", "Netflix Standard with Ads",
        "Starz", "Starz Apple TV Channel", "Starz Roku Premium Channel",
    ])
    assert out == ["Netflix", "Starz"]


def test_pipeline_parse_movie():
    raw = {
        "id": 12345,
        "title": "Test Movie",
        "release_date": "2025-06-15",
        "runtime": 120,
        "genres": [{"id": 28, "name": "Action"}, {"id": 53, "name": "Thriller"}],
        "overview": "A test movie.",
        "credits": {
            "cast": [{"name": "Actor A"}, {"name": "Actor B"}],
            "crew": [{"job": "Director", "name": "Director X"}],
        },
        "vote_average": 7.5,
        "vote_count": 100,
        "watch/providers": {
            "results": {
                "US": {
                    "flatrate": [{"provider_name": "Netflix"}, {"provider_name": "Hulu"}],
                }
            }
        },
        "release_dates": {
            "results": [
                {"iso_3166_1": "US", "release_dates": [{"certification": "PG-13"}]},
            ]
        },
    }
    movie = TMDBPipeline._parse_movie(raw, region="US")
    assert movie.id == "tmdb_12345"
    assert movie.title == "Test Movie"
    assert movie.genres == ["Action", "Thriller"]
    assert movie.streaming_services == ["Netflix", "Hulu"]
    assert movie.content_rating == "PG-13"
    assert movie.director == "Director X"
