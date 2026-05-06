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
