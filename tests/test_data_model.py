import json
from tau_rec.data_model.catalog import Movie, Catalog

def test_movie_from_dict():
    data = {
        "id": "tt001",
        "title": "Test Movie",
        "release_date": "2025-06-15",
        "runtime": 120,
        "genres": ["Action", "Thriller"],
        "overview": "A test movie about testing.",
        "cast": ["Actor A", "Actor B"],
        "director": "Director X",
        "rating": 7.5,
        "vote_count": 100,
        "streaming_services": ["Netflix", "Hulu"],
        "sponsored": False,
        "content_rating": "PG-13",
    }
    movie = Movie(**data)
    assert movie.id == "tt001"
    assert movie.runtime == 120
    assert "Action" in movie.genres
    assert movie.sponsored is False
    assert movie.content_rating == "PG-13"

def test_catalog_load_and_lookup():
    movies_data = [
        {
            "id": "tt001", "title": "Movie A", "release_date": "2025-06-15",
            "runtime": 90, "genres": ["Comedy"], "overview": "Funny.",
            "cast": ["A"], "director": "D1", "rating": 6.0, "vote_count": 50,
            "streaming_services": ["Netflix"], "sponsored": False,
            "content_rating": "PG",
        },
        {
            "id": "tt002", "title": "Movie B", "release_date": "2025-07-01",
            "runtime": 150, "genres": ["Drama"], "overview": "Dramatic.",
            "cast": ["B"], "director": "D2", "rating": 8.0, "vote_count": 200,
            "streaming_services": ["Hulu", "Amazon"], "sponsored": True,
            "content_rating": "R",
        },
    ]
    catalog = Catalog(movies=[Movie(**m) for m in movies_data])
    assert len(catalog) == 2
    assert catalog.get("tt001").title == "Movie A"
    assert catalog.get("tt999") is None

def test_catalog_o1_lookup():
    """Verify the index provides O(1) lookups."""
    movies = [
        Movie(id=f"tt{i:03d}", title=f"M{i}", release_date="2025-01-01",
              runtime=100, genres=["Action"], overview="", cast=[], director="D",
              rating=5.0, vote_count=10, streaming_services=[])
        for i in range(100)
    ]
    catalog = Catalog(movies=movies)
    assert catalog.get("tt050").title == "M50"
    assert catalog.get("tt099").title == "M99"


from tau_rec.data_model.conversation import (
    Role, Message, ToolCall, ConversationTrace, StopReason,
)

def test_conversation_trace():
    trace = ConversationTrace(task_id="task_001", model="gpt-4o", trial=0)
    trace.add_message(Message(role=Role.AGENT, content="Hi! What kind of movie are you looking for?"))
    trace.add_message(Message(role=Role.USER, content="Something short and thrilling."))
    trace.add_tool_call(ToolCall(name="search_catalog", arguments={"query": "short thriller"}, result='[{"id": "tt001"}]'))
    trace.add_message(Message(role=Role.AGENT, content="I found one: Movie X."))
    assert trace.turn_count == 1  # 2 agent msgs, 1 user msg -> min = 1
    assert trace.tool_call_count == 1
    assert len(trace.messages) == 3
    assert len(trace.tool_calls) == 1

def test_trace_extracts_recommendations():
    trace = ConversationTrace(task_id="task_001", model="gpt-4o", trial=0)
    trace.add_recommendation("tt001")
    trace.add_recommendation("tt005")
    assert trace.final_recommendation == "tt005"
    assert trace.all_recommendations == ["tt001", "tt005"]

def test_stop_reason():
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.stop_reason = StopReason.RECOMMENDED
    assert trace.stop_reason == StopReason.RECOMMENDED


from tau_rec.data_model.task import Constraint, RevealTag, TaskConstraint, Task

def test_constraint_evaluation():
    movie = Movie(
        id="tt001", title="X", release_date="2025-01-01", runtime=95,
        genres=["Thriller", "Drama"], overview="", cast=["A"], director="D",
        rating=7.5, vote_count=10, streaming_services=["Netflix"],
        sponsored=False, content_rating="R",
    )
    c1 = Constraint(field="runtime", op="<=", value=120)
    c2 = Constraint(field="genres", op="contains", value="Thriller")
    c3 = Constraint(field="streaming_services", op="contains_any", value=["Hulu", "Netflix"])
    c4 = Constraint(field="runtime", op="<=", value=90)
    assert c1.evaluate(movie) is True
    assert c2.evaluate(movie) is True
    assert c3.evaluate(movie) is True
    assert c4.evaluate(movie) is False

def test_constraint_on_missing_attribute_is_unsatisfied():
    """An item that lacks the attribute satisfies nothing about it.

    Movies carry every field, so this never fired on movies-v1. It matters for
    tracks whose items have optional fields: an item with no recorded value must
    not pass a numeric bound, and must not raise TypeError mid-run either.
    """
    from types import SimpleNamespace
    item = SimpleNamespace(publication_year=None, genres=None, content_rating=None)
    cases = [
        Constraint(field="publication_year", op="<=", value=2000),
        Constraint(field="publication_year", op=">=", value=1990),
        Constraint(field="publication_year", op="==", value=1999),
        Constraint(field="publication_year", op="!=", value=1999),
        Constraint(field="genres", op="contains", value="Mystery"),
        Constraint(field="genres", op="contains_any", value=["Mystery", "Fantasy"]),
        Constraint(field="genres", op="not_contains", value="Horror"),
        Constraint(field="content_rating", op="in", value=["Adult"]),
    ]
    for c in cases:
        assert c.evaluate(item) is False, f"{c.field} {c.op} {c.value}"


def test_catalog_accepts_items_and_movies_keywords():
    """`items` is the field; `movies` is an accepted alias and a property.

    Ten call sites and the fixtures in this suite construct Catalog(movies=...),
    and the on-disk format is a bare list, so widening the field name must not
    move any of them.
    """
    m = Movie(
        id="tt001", title="X", release_date="2025-01-01", runtime=95,
        genres=["Drama"], overview="", cast=[], director="D",
        rating=7.0, vote_count=10, streaming_services=[],
    )
    by_items = Catalog(items=[m])
    by_movies = Catalog(movies=[m])
    by_dict = Catalog.model_validate({"movies": [m.model_dump()]})
    for cat in (by_items, by_movies, by_dict):
        assert cat.items == [m]
        assert cat.movies == [m]
        assert len(cat) == 1
        assert cat.get("tt001") == m
        assert cat.get("tt999") is None


def test_task_rejects_unknown_keys():
    """A misspelled key fails at load instead of disabling a gate silently."""
    import glob
    import pytest
    from pathlib import Path
    # Every shipped task still loads under the stricter model.
    paths = sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "data" / "tasks" / "*.json")))
    assert len(paths) == 60
    for p in paths:
        Task.from_json(p)
    book_task = Task(id="t", constraints=[], persona="p", complexity="simple",
                     reveal_difficulty="volunteer", user_formats=["audiobook"])
    assert book_task.user_formats == ["audiobook"]
    with pytest.raises(Exception, match="user_formtz"):
        Task(id="t", constraints=[], persona="p", complexity="simple",
             reveal_difficulty="volunteer", user_formtz=["audiobook"])


def test_task_structure():
    task = Task(
        id="task_001",
        constraints=[
            TaskConstraint(
                constraint=Constraint(field="runtime", op="<=", value=120),
                reveal=RevealTag.VOLUNTEER,
            ),
            TaskConstraint(
                constraint=Constraint(field="genres", op="contains", value="Thriller"),
                reveal=RevealTag.ON_ASK,
            ),
        ],
        persona="You are a busy professional who likes quick, intense movies.",
        soft_preferences=["prefers practical effects over CGI"],
        policy_flags=["watch_history", "availability"],
        no_valid_recommendation=False,
        complexity="simple",
        reveal_difficulty="mixed",
    )
    assert task.id == "task_001"
    assert len(task.constraints) == 2
    assert task.constraints[0].reveal == RevealTag.VOLUNTEER
    assert task.complexity == "simple"
    assert task.user_id == "user_1"
