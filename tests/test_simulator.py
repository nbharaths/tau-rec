import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tau_rec.data_model.catalog import Book, Catalog
from tau_rec.data_model.conversation import Message, Role
from tau_rec.data_model.task import Constraint, RevealTag, Task, TaskConstraint
from tau_rec.domain import BOOKS, MOVIES
from tau_rec.simulator.user_simulator import (
    BOOKS_CONTROLLER_VERSION,
    BOOKS_GROUNDING_VERSION,
    BookDisclosureController,
    UserSimulator,
    build_simulator_prompt,
    classify_book_question,
    request_parameter_manifest,
)


def test_build_simulator_prompt():
    task = Task(
        id="t1",
        constraints=[
            TaskConstraint(
                constraint=Constraint(field="runtime", op="<=", value=120),
                reveal=RevealTag.VOLUNTEER,
            ),
            TaskConstraint(
                constraint=Constraint(field="genres", op="contains", value="Thriller"),
                reveal=RevealTag.ON_ASK,
            ),
            TaskConstraint(
                constraint=Constraint(field="streaming_services", op="contains_any", value=["Netflix"]),
                reveal=RevealTag.HIDDEN,
            ),
        ],
        persona="You are a college student looking for a weekend movie.",
        soft_preferences=["likes plot twists"],
        complexity="medium", reveal_difficulty="mixed",
    )
    prompt = build_simulator_prompt(task)
    assert "###ACCEPTED###" in prompt
    assert "###REJECTED###" in prompt
    assert "college student" in prompt
    assert "runtime" in prompt


def _book(item_id: str, title: str, *, year: int, genres: list[str]) -> Book:
    return Book(
        id=item_id,
        title=title,
        authors=["A. Writer"],
        publication_year=year,
        pages=300,
        genres=genres,
        description="",
        publisher="P",
        rating=None,
        vote_count=None,
        formats=["hardcover"],
    )


def _grounded_simulator(*, nvr: bool = False) -> tuple[UserSimulator, Book, Book]:
    good = _book("wd_good", "The Good Book", year=2020, genres=["Mystery"])
    bad = _book("wd_bad", "The Wrong Book", year=1980, genres=["Mystery"])
    catalog = Catalog(items=[good, bad])
    task = Task(
        id="t-books",
        persona="You want a book.",
        complexity="simple",
        reveal_difficulty="hidden",
        no_valid_recommendation=nvr,
        constraints=[TaskConstraint(
            constraint=Constraint(field="publication_year", op=">=", value=2000),
            reveal=RevealTag.HIDDEN,
        )],
    )
    return UserSimulator("must-not-be-called", task, BOOKS, catalog), good, bad


async def test_books_candidate_marker_gets_deterministic_typed_verdict():
    simulator, good, bad = _grounded_simulator()
    accepted = await simulator.respond([
        Message(role=Role.AGENT, content=f"I recommend {good.title} [[candidate_id:{good.id}]]")
    ])
    rejected = await simulator.respond([
        Message(role=Role.AGENT, content=f"I recommend {bad.title} [[candidate_id:{bad.id}]]")
    ])
    assert accepted.content.startswith("###ACCEPTED###")
    assert rejected.content.startswith("###REJECTED###")
    assert "doesn't satisfy all of my requirements" in rejected.content
    assert "4.0" not in rejected.content  # hidden threshold stays hidden


async def test_books_grounding_rejects_every_candidate_for_nvr_task():
    simulator, good, _ = _grounded_simulator(nvr=True)
    response = await simulator.respond([
        Message(role=Role.AGENT, content=f"My pick is {good.title} [[candidate_id:{good.id}]]")
    ])
    assert response.content.startswith("###REJECTED###")


async def test_books_grounding_does_not_reveal_nvr_state_from_abstention_probe():
    nvr_simulator, _, _ = _grounded_simulator(nvr=True)
    satisfiable_simulator, _, _ = _grounded_simulator()

    nvr_response = await nvr_simulator.respond([
        Message(
            role=Role.AGENT,
            content="Nothing in the catalog matches. [[candidate_id:NONE]]",
        )
    ])
    satisfiable_response = await satisfiable_simulator.respond([
        Message(
            role=Role.AGENT,
            content="Nothing in the catalog matches. [[candidate_id:NONE]]",
        )
    ])

    assert nvr_response.content.startswith("###REJECTED###")
    assert satisfiable_response.content.startswith("###REJECTED###")
    assert nvr_response.content == satisfiable_response.content
    assert "cannot verify" in nvr_response.content


async def test_books_grounding_asks_for_one_valid_candidate_marker():
    simulator, good, bad = _grounded_simulator()
    multiple = await simulator.respond([Message(
        role=Role.AGENT,
        content=(
            f"Either {good.title} [[candidate_id:{good.id}]] or "
            f"{bad.title} [[candidate_id:{bad.id}]]"
        ),
    )])
    invalid = await simulator.respond([
        Message(role=Role.AGENT, content="My pick [[candidate_id:wd_missing]]")
    ])
    assert "###ACCEPTED###" not in multiple.content and "###REJECTED###" not in multiple.content
    assert "one catalog title" in multiple.content
    assert "###ACCEPTED###" not in invalid.content and "###REJECTED###" not in invalid.content
    assert "check its item ID" in invalid.content


def test_simulator_manifest_records_grounding_only_for_books():
    assert request_parameter_manifest(BOOKS)["proposal_grounding"] == BOOKS_GROUNDING_VERSION
    assert request_parameter_manifest(BOOKS)["timeout_seconds"] == 120
    assert "proposal_grounding" not in request_parameter_manifest(MOVIES)
    assert "proposal_grounding" not in request_parameter_manifest(BOOKS, mode="memory")


async def test_books_grounding_suppresses_ungrounded_llm_verdict(monkeypatch):
    import litellm

    simulator, _, _ = _grounded_simulator()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="###ACCEPTED### guessed"))],
        usage=None,
    )
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr(litellm, "acompletion", completion)

    message = await simulator.respond([
        Message(role=Role.AGENT, content="I have a possibility, but no concrete title yet.")
    ])

    assert "###ACCEPTED###" not in message.content
    assert "###REJECTED###" not in message.content
    assert "[[candidate_id:ID]]" in message.content
    assert len(simulator.call_log) == 1
    assert simulator.call_log[0]["status"] == "ok"
    assert simulator.call_log[0]["tokens"] == {
        "input": 0, "cached": 0, "cache_write": 0, "output": 0,
    }


def _llm_reply(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None,
    )


async def test_llm_user_answers_a_question_inside_a_candidate_turn(monkeypatch):
    import litellm

    simulator, _, _ = _grounded_simulator()
    completion = AsyncMock(return_value=_llm_reply("A mystery would be lovely."))
    monkeypatch.setattr(litellm, "acompletion", completion)

    reply = await simulator.respond([
        Message(
            role=Role.AGENT,
            content=(
                "How about The Wrong Book [[candidate_id:wd_bad]]? "
                "Also, do you have a preferred genre?"
            ),
        ),
    ])

    assert reply.content.startswith("###REJECTED###")
    assert reply.content.endswith("\nA mystery would be lovely.")
    # The model sees the whole turn and the verdict already given.
    messages = completion.await_args.kwargs["messages"]
    assert "preferred genre" in messages[-1]["content"]
    assert "candidate_id" in messages[-1]["content"]
    assert "doesn't satisfy all of my requirements" in messages[0]["content"]
    assert simulator.call_log[-1]["purpose"] == "embedded-answer-v2"


async def test_llm_user_resamples_an_empty_reply(monkeypatch):
    import litellm
    import tau_rec.simulator.user_simulator as module

    simulator, _, _ = _grounded_simulator()
    completion = AsyncMock(side_effect=[_llm_reply(None), _llm_reply("  "), _llm_reply("Hi there.")])
    monkeypatch.setattr(litellm, "acompletion", completion)
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())

    reply = await simulator.respond([Message(role=Role.AGENT, content="Hello! What do you like?")])

    assert reply.content == "Hi there."
    assert [call["status"] for call in simulator.call_log] == ["empty", "empty", "ok"]


async def test_llm_user_token_free_empty_reply_does_not_trip_the_cost_ceiling(monkeypatch):
    import litellm
    import tau_rec.simulator.user_simulator as module
    from tau_rec import cost

    simulator, _, _ = _grounded_simulator()
    priced = _llm_reply("Hi there.")
    priced._hidden_params = {"response_cost": 0.0001}
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(side_effect=[_llm_reply(None), priced]))
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
    cost.reset()
    cost.configure_provider_budget(1.0)
    try:
        reply = await simulator.respond([Message(role=Role.AGENT, content="Hello! What do you like?")])
    finally:
        cost.reset()

    assert reply.content == "Hi there."
    assert [call["status"] for call in simulator.call_log] == ["empty", "ok"]


async def test_llm_user_gives_up_after_repeated_empty_replies(monkeypatch):
    import litellm
    import pytest
    import tau_rec.simulator.user_simulator as module

    simulator, _, _ = _grounded_simulator()
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=_llm_reply(None)))
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())

    with pytest.raises(RuntimeError, match="no text"):
        await simulator.respond([Message(role=Role.AGENT, content="Hello! What do you like?")])
    assert len(simulator.call_log) == module.MAX_RETRIES


async def test_llm_user_candidate_turn_without_a_question_skips_the_model(monkeypatch):
    import litellm

    simulator, _, _ = _grounded_simulator()
    completion = AsyncMock(return_value=_llm_reply("unused"))
    monkeypatch.setattr(litellm, "acompletion", completion)

    reply = await simulator.respond([
        Message(role=Role.AGENT, content="How about The Wrong Book [[candidate_id:wd_bad]]?"),
    ])

    assert reply.content.startswith("###REJECTED###")
    assert completion.await_count == 0


async def test_llm_user_embedded_answer_never_carries_a_guessed_verdict(monkeypatch):
    import litellm

    simulator, _, _ = _grounded_simulator()
    completion = AsyncMock(return_value=_llm_reply("###ACCEPTED### sounds great"))
    monkeypatch.setattr(litellm, "acompletion", completion)

    reply = await simulator.respond([
        Message(
            role=Role.AGENT,
            content="How about The Wrong Book [[candidate_id:wd_bad]]? Any preferred genre?",
        ),
    ])

    assert reply.content.startswith("###REJECTED###")
    assert "###ACCEPTED###" not in reply.content


def _controller_task() -> Task:
    return Task(
        id="controlled-books",
        persona="You want one book.",
        complexity="medium",
        reveal_difficulty="hidden",
        constraints=[
            TaskConstraint(
                constraint=Constraint(field="genres", op="contains", value="Mystery"),
                reveal=RevealTag.VOLUNTEER,
            ),
            TaskConstraint(
                constraint=Constraint(field="pages", op="<=", value=320),
                reveal=RevealTag.ON_ASK,
            ),
            TaskConstraint(
                constraint=Constraint(field="rating", op=">=", value=4.0),
                reveal=RevealTag.ON_ASK,
            ),
            TaskConstraint(
                constraint=Constraint(field="publication_year", op=">=", value=2015),
                reveal=RevealTag.HIDDEN,
            ),
        ],
    )


def test_book_question_classifier_is_frozen_and_conservative():
    assert classify_book_question("How long should the book be?").fields == ("pages",)
    assert classify_book_question("Paperback, ebook, or audio?").fields == ("formats",)
    assert classify_book_question("What original language should it be written in?").fields == (
        "languages",
    )
    assert classify_book_question("Do you have a preferred country of origin?").fields == (
        "countries",
    )
    assert classify_book_question("Do you care about rating or year?").fields == (
        "rating", "publication_year",
    )
    assert classify_book_question("Any other requirements?").kind == "generic"
    assert classify_book_question("I will search the Mystery genre.").kind == "none"
    assert classify_book_question("I need to check the page count next.").kind == "none"
    assert classify_book_question("Who is your favorite author?").kind == "none"


def test_book_questions_recognize_natural_field_phrasings():
    def fields(message):
        return set(classify_book_question(message).fields)

    assert "genres" in fields("Any particular subgenre you enjoy?")
    assert "genres" in fields("Do you prefer mystery, sci-fi, romance, or horror?")
    assert "publication_year" in fields("Is there a time period or decade you prefer?")
    assert "publication_year" in fields("Any era in mind?")
    assert "publication_year" in fields("Something from the 1990s, maybe?")
    assert "countries" in fields("Where should the author be from?")
    assert "countries" in fields("Would you like something American or British?")
    assert "languages" in fields("Do you want a book originally in Japanese?")


def test_every_compact_genre_is_a_genre_question():
    from tau_rec.catalog.wikidata_books import PUBLIC_BOOK_GENRES

    for genre in PUBLIC_BOOK_GENRES:
        assert "genres" in classify_book_question(f"Would you like {genre}?").fields, genre


def test_generic_request_counts_alongside_field_words():
    intent = classify_book_question(
        "What format do you prefer? Is there anything else you'd like?"
    )
    assert intent.fields == ("formats",)
    assert intent.generic


def test_field_words_outside_question_sentences_are_ignored():
    intent = classify_book_question("I searched by genre and year. Any other requirements?")
    assert intent.kind == "generic"
    assert intent.fields == ()


def test_generic_and_field_question_discloses_one_more_fact():
    controller = BookDisclosureController(_controller_task())
    controller.respond("Hello")

    reply = controller.respond("Would you prefer a short book? Anything else?")

    assert "320 pages" in reply.content
    assert "4.0" in reply.content
    assert "2015" not in reply.content


def _symbolic_genre_simulator() -> UserSimulator:
    catalog = Catalog(items=[
        _book("wd_good", "The Good Book", year=2020, genres=["Mystery"]),
        _book("wd_bad", "The Wrong Book", year=1980, genres=["Mystery"]),
    ])
    task = _controller_task().model_copy(update={"constraints": [
        TaskConstraint(
            constraint=Constraint(field="publication_year", op=">=", value=2000),
            reveal=RevealTag.VOLUNTEER,
        ),
        TaskConstraint(
            constraint=Constraint(field="genres", op="contains", value="Mystery"),
            reveal=RevealTag.ON_ASK,
        ),
        TaskConstraint(
            constraint=Constraint(field="publication_year", op=">=", value=2015),
            reveal=RevealTag.HIDDEN,
        ),
    ]})
    return UserSimulator("must-not-be-called", task, BOOKS, catalog, interaction_mode="symbolic")


async def test_symbolic_user_answers_a_question_inside_a_candidate_turn():
    simulator = _symbolic_genre_simulator()
    opening = await simulator.respond([Message(role=Role.AGENT, content="Hello!")])

    reply = await simulator.respond([
        Message(role=Role.AGENT, content="Hello!"),
        opening,
        Message(
            role=Role.AGENT,
            content=(
                "How about The Wrong Book [[candidate_id:wd_bad]]? "
                "Also, do you have a preferred genre?"
            ),
        ),
    ])

    assert reply.content.startswith("###REJECTED###")
    assert "it should include Mystery" in reply.content
    assert "2015" not in reply.content


async def test_candidate_sentence_alone_is_not_a_field_question():
    simulator = _symbolic_genre_simulator()
    opening = await simulator.respond([Message(role=Role.AGENT, content="Hello!")])

    reply = await simulator.respond([
        Message(role=Role.AGENT, content="Hello!"),
        opening,
        Message(
            role=Role.AGENT,
            content="How about this mystery pick, The Wrong Book [[candidate_id:wd_bad]]?",
        ),
    ])

    assert reply.content.startswith("###REJECTED###")
    assert "it should include Mystery" not in reply.content


def test_purpose_clause_does_not_ask_for_an_extra_fact():
    intent = classify_book_question("To narrow it down, could you tell me what genres you enjoy?")
    assert intent.fields == ("genres",)
    assert not intent.generic


def test_a_books_own_year_is_not_a_year_question():
    assert classify_book_question("Is Dune (1965) okay?").kind == "none"


def test_a_genre_name_needs_a_choice_frame():
    assert classify_book_question("Is this mystery a good fit?").kind == "none"
    assert "genres" in classify_book_question("Do you like mystery?").fields


def test_story_setting_is_not_publication_era():
    intent = classify_book_question(
        "Do you have a preferred era or setting, e.g., Victorian or WWII?"
    )
    assert "publication_year" not in intent.fields


def test_controller_version_marks_the_v4_recognizer():
    assert BOOKS_CONTROLLER_VERSION == "book-disclosure-v4"


def test_symbolic_controller_owns_disclosure_state_and_never_reveals_hidden():
    controller = BookDisclosureController(_controller_task())

    opening = controller.respond("Hello! What kind of book can I help you find?")
    irrelevant = controller.respond("Who is your favorite author?")
    pages = controller.respond("Would you prefer a short or long book?")
    generic = controller.respond("Any other requirements?")
    hidden_probe = controller.respond("Does publication year matter?")

    assert "Mystery" in opening.content
    assert "320" not in opening.content and "2015" not in opening.content
    assert "narrow down" in irrelevant.content
    assert "320 pages" in pages.content
    assert "4.0" in generic.content
    assert "2015" not in hidden_probe.content
    assert controller.disclosed == {0, 1, 2}
    assert all(
        3 not in decision["disclosed_constraint_indices"]
        for decision in controller.decisions
    )


def test_symbolic_controller_discloses_policy_antecedents_without_catalog_facts():
    task = _controller_task().model_copy(update={
        "policy_flags": ["history", "format_verification"],
        "user_formats": ["ebook", "paperback"],
    })
    controller = BookDisclosureController(task)

    opening = controller.respond("Hello")
    formats = controller.respond("Which format do you prefer?")

    assert "haven't already read" in opening.content
    assert "ebook, paperback" in formats.content
    assert controller.decisions[0]["disclosed_policy_facts"] == ["history_requirement"]
    assert controller.decisions[1]["disclosed_policy_facts"] == ["accepted_formats"]


async def test_symbolic_simulator_uses_generic_rejection_and_no_llm(monkeypatch):
    import litellm

    catalog = Catalog(items=[
        _book("wd_good", "The Good Book", year=2020, genres=["Mystery"]),
        _book("wd_bad", "The Wrong Book", year=1980, genres=["Mystery"]),
    ])
    simulator = UserSimulator(
        "must-not-be-called",
        _controller_task(),
        BOOKS,
        catalog,
        interaction_mode="symbolic",
    )
    completion = AsyncMock(side_effect=AssertionError("symbolic mode called an LLM"))
    monkeypatch.setattr(litellm, "acompletion", completion)

    opening = await simulator.respond([
        Message(role=Role.AGENT, content="Hello! What kind of book can I help you find?")
    ])
    rejected = await simulator.respond([
        Message(role=Role.AGENT, content="Hello!"),
        opening,
        Message(
            role=Role.AGENT,
            content="I recommend The Wrong Book [[candidate_id:wd_bad]]",
        ),
    ])

    assert "Mystery" in opening.content
    assert rejected.content.startswith("###REJECTED###")
    assert "doesn't satisfy all" in rejected.content
    assert "year" not in rejected.content.lower()  # no failed-field hint
    assert completion.await_count == 0
    assert simulator.decision_log[-1]["action"] == "candidate_verdict"


def test_symbolic_simulator_manifest_records_non_stochastic_controller():
    manifest = request_parameter_manifest(BOOKS, interaction_mode="symbolic")
    assert manifest["disclosure_controller"] == BOOKS_CONTROLLER_VERSION
    assert manifest["temperature"] == "not applicable"
    assert manifest["rejection_feedback"].startswith("generic")


async def test_llm_user_request_carries_provider_pin(monkeypatch):
    import litellm

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="I'd like a mystery."))],
        usage=None,
        provider="Z.AI",
    )
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr(litellm, "acompletion", completion)
    simulator = UserSimulator(
        "openrouter/z-ai/glm-5.3-flash", _controller_task(), BOOKS, None,
        provider_order=["z-ai/fp8"],
    )

    await simulator.respond([Message(role=Role.AGENT, content="Hello!")])

    assert completion.await_args.kwargs["extra_body"]["provider"]["order"] == ["z-ai/fp8"]
    assert completion.await_args.kwargs["extra_body"]["provider"]["allow_fallbacks"] is False
    assert simulator.call_log[0]["served_provider"] == "Z.AI"
