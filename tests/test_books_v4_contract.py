"""Contract tests for the books v4 harness.

Each test pins one behaviour of the books-v4 harness profile. Movies behaviour
is pinned separately by tests/test_goldens.py and the zero-movement gate.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau_rec.agents.base import AgentResponse
from tau_rec.agents.litellm_agent import LiteLLMAgent, build_agent_system_prompt
from tau_rec.catalog.search import CatalogSearch
from tau_rec.catalog.validator import CatalogValidator
from tau_rec.data_model.catalog import Book, Catalog
from tau_rec.data_model.conversation import (
    AgentToolTextEvent, ConversationTrace, Message, MessageEvent, Role, StopReason, ToolCall,
    ToolCallEvent,
)
from tau_rec.data_model.task import Constraint, RevealTag, Task, TaskConstraint
from tau_rec.domain import BOOKS, MOVIES
from tau_rec.environment.tools import ToolKit, tool_definitions
from tau_rec.evaluator.policy import PolicyEvaluator
from tau_rec.harness import BOOKS_V4
from tau_rec.orchestrator.orchestrator import Orchestrator
from tau_rec.track import find_track, load_track

ROOT = Path(__file__).resolve().parent.parent
V2_TRACK = ROOT / "data" / "tracks" / "books-wikidata-v2"


def _book(item_id: str, formats: list[str], **fields) -> Book:
    base = {
        "id": item_id, "title": f"Work {item_id}", "authors": ["A. Writer"], "publication_year": 2001,
        "pages": None, "genres": ["Mystery"], "description": "a novel", "publisher": "P",
        "rating": None, "vote_count": None, "formats": formats, "languages": ["English"],
        "countries": ["United States"], "sitelink_count": 1,
    }
    base.update(fields)
    return Book(**base)


def _format_task(**fields) -> Task:
    base = {
        "id": "t1", "constraints": [], "persona": "p", "complexity": "simple",
        "reveal_difficulty": "mixed", "policy_flags": ["format_verification_v2"],
        "user_formats": ["audiobook"],
    }
    base.update(fields)
    return Task(**base)


# --- Formats: vocabulary and scoring ------------------------------------------


def test_format_names_are_case_insensitive_with_common_synonyms():
    """Format names match case-insensitively and accept common synonyms."""
    for spelling in ("Audiobook", "AUDIOBOOK", " audiobook ", "Audible", "audio book", "audiobooks"):
        assert BOOKS.normalize_service(spelling) == "audiobook"
    assert BOOKS.normalize_service("Hardback") == "hardcover"
    assert BOOKS.normalize_service("eBook") == "ebook"
    assert BOOKS.normalize_service("Kindle Edition") == "ebook"
    assert BOOKS.normalize_service("Paperbacks") == "paperback"
    assert BOOKS.normalize_service("vinyl") == "vinyl"

    catalog = Catalog(items=[_book("wd_1", ["ebook", "paperback"]), _book("wd_2", ["audiobook"]),
                             _book("wd_3", ["hardcover"])])
    kit = ToolKit(catalog, CatalogSearch(catalog, BOOKS), {}, spec=BOOKS)
    result = json.loads(kit.call("check_availability", {"item_id": "wd_1", "formats": ["EBOOK", "Audiobook"]}))
    assert result == {"EBOOK": True, "Audiobook": False}


def test_format_mentions_recognise_the_spellings_the_tools_accept():
    assert BOOKS.mentions_service("I mostly listen to Audiobooks these days.", "audiobook")
    assert BOOKS.mentions_service("I can use any of these formats: audiobook.", "audiobook")
    assert BOOKS.mentions_service("An audio book is fine.", "audiobook")
    assert not BOOKS.mentions_service("Any format is fine.", "audiobook")
    assert not BOOKS.mentions_service("I want an ebook.", "audiobook")


def test_format_rule_v2_applies_only_once_the_user_names_a_format():
    """The format check applies only after the user names a format."""
    catalog = Catalog(items=[_book("wd_1", ["paperback"]), _book("wd_2", ["audiobook"])])
    task = _format_task()
    evaluator = PolicyEvaluator(catalog, BOOKS)

    def trace(messages: list[tuple[Role, str]], item_id: str, *, verify: bool) -> ConversationTrace:
        result = ConversationTrace(task_id="t1", model="m", trial=0)
        for role, content in messages:
            result.add_message(Message(role=role, content=content))
        if verify:
            result.add_tool_call(ToolCall(name="get_metadata", arguments={"item_id": item_id},
                                          result=json.dumps({"id": item_id})))
        result.add_tool_call(ToolCall(name="recommend", arguments={"item_id": item_id}, result="{}"))
        result.add_recommendation(item_id)
        return result

    unnamed = [(Role.USER, "I'm looking for one book.")]
    assert evaluator.evaluate(task, trace(unnamed, "wd_1", verify=False)).score == 1.0
    # Only the user can name a format; the agent naming one does not count.
    agent_named = [(Role.AGENT, "Would an audiobook work?"), (Role.USER, "Yes.")]
    assert evaluator.evaluate(task, trace(agent_named, "wd_1", verify=False)).score == 1.0

    for named in ("I can use any of these formats: audiobook.", "I mostly listen to Audiobooks."):
        said = [(Role.USER, named)]
        assert evaluator.evaluate(task, trace(said, "wd_2", verify=True)).score == 1.0
        assert evaluator.evaluate(task, trace(said, "wd_2", verify=False)).violations == ["format_verification_v2"]
        assert evaluator.evaluate(task, trace(said, "wd_1", verify=True)).violations == ["format_verification_v2"]

    # The original format_verification rule still fires without a named format.
    v1_task = _format_task(policy_flags=["format_verification"])
    assert evaluator.evaluate(v1_task, trace(unnamed, "wd_1", verify=False)).violations == ["format_verification"]


def test_validator_requires_formats_for_the_v2_format_rule():
    catalog = Catalog(items=[_book("wd_1", ["paperback"])])
    result = CatalogValidator(catalog, BOOKS).validate_task(_format_task(user_formats=[]))
    assert any("requires non-empty user_formats" in error for error in result.errors)


def test_v2_track_scores_formats_with_the_v2_rule():
    tasks = [json.loads(path.read_text()) for path in sorted((V2_TRACK / "tasks").glob("*.json"))]
    assert len(tasks) == 450
    for task in tasks:
        assert ("format_verification_v2" in task["policy_flags"]) == bool(task["user_formats"]), task["id"]
        assert "format_verification" not in task["policy_flags"], task["id"]


# --- Tools -----------------------------------------------------------------------


def _structured_kit(items: list[Book]) -> ToolKit:
    return ToolKit(Catalog(items=items), None, {}, spec=BOOKS, retrieval_mode="structured",
                   structured_profile="books-wikidata-core-v3")


def test_books_tools_drop_the_content_preference_check():
    """Books have no age gate, so the content-preference tool is not offered."""
    assert "check_content_preference" not in [t["function"]["name"] for t in tool_definitions(BOOKS)]
    assert "check_content_preference" in [t["function"]["name"] for t in tool_definitions(MOVIES)]
    kit = _structured_kit([_book("wd_1", ["ebook"])])
    assert "check_content_preference" not in [t["function"]["name"] for t in kit.tool_definitions()]
    assert json.loads(kit.call("check_content_preference", {"content_rating": "Adult"})) == {
        "error": "Unknown tool: check_content_preference"
    }


def test_unknown_labels_and_formats_return_the_valid_values():
    """An unknown label returns the valid values, not an empty result."""
    kit = _structured_kit([
        _book("wd_1", ["ebook"], genres=["Mystery"], languages=["French"], countries=["France"]),
        _book("wd_2", ["paperback"], genres=["Fantasy"]),
        _book("wd_3", ["audiobook"], genres=["Fantasy"]),
    ])
    error = json.loads(kit.call("search_catalog", {"genres": ["Sci-Fi"]}))["error"]
    assert all(word in error for word in ("Sci-Fi", "Fantasy", "Mystery"))
    error = json.loads(kit.call("search_catalog", {"languages": ["Klingon"]}))["error"]
    assert all(word in error for word in ("Klingon", "French", "English"))
    error = json.loads(kit.call("search_catalog", {"countries": ["Atlantis"]}))["error"]
    assert all(word in error for word in ("Atlantis", "France", "United States"))
    error = json.loads(kit.call("search_catalog", {"formats": ["vinyl"]}))["error"]
    assert all(word in error for word in ("vinyl", "audiobook", "ebook", "paperback"))
    # Known labels still filter in any case; a known label with no match is an empty list.
    assert [row["id"] for row in json.loads(kit.call("search_catalog", {"genres": ["mystery"]}))] == ["wd_1"]
    assert json.loads(kit.call("search_catalog", {"genres": ["Mystery"], "formats": ["Paperback"]})) == []

    error = json.loads(kit.call("check_availability", {"item_id": "wd_1", "formats": ["vinyl"]}))["error"]
    assert all(word in error for word in ("vinyl", "audiobook", "ebook", "paperback"))
    assert json.loads(kit.call("check_availability", {"item_id": "wd_1", "formats": ["Audiobook"]})) == {
        "Audiobook": False
    }


def test_recommend_null_string_is_an_abstention():
    """Null-like item ids are abstentions, and the tool reports them as such."""
    kit = _structured_kit([_book("wd_1", ["ebook"])])
    for value in (None, "", "null", "None", " n/a "):
        assert json.loads(kit.call("recommend", {"item_id": value}))["status"] == "abstained"
    assert json.loads(kit.call("recommend", {"item_id": "wd_9"})) == {"error": "Book not found: wd_9"}


# --- Agent prompt ----------------------------------------------------------------


def test_books_agent_prompt_abstains_only_when_nothing_satisfies_every_requirement():
    """The books prompt abstains only when nothing satisfies every requirement (Policy 4)."""
    policy = (V2_TRACK / "policy.md").read_text()
    for retrieval in ("keyword", "structured"):
        prompt = build_agent_system_prompt(
            policy, "user_1", spec=BOOKS, retrieval_mode=retrieval,
            structured_profile="books-wikidata-core-v3",
        )
        for movie_rule in ("approximately", "When in doubt", "best-effort", "slightly lower rating",
                           "imperfect genre", "and ratings", "check_content_preference"):
            assert movie_rule not in prompt, (retrieval, movie_rule)
        assert "satisfies all of the user's requirements" in prompt
        assert "partial match" in prompt
    assert "ratings" not in json.dumps(tool_definitions(BOOKS))


# --- Orchestrator and model calls -------------------------------------------------


class ScriptedAgent:
    """Replays responses in order and repeats the last one."""

    def __init__(self, responses: list[AgentResponse], delay: float = 0.0) -> None:
        self.responses = list(responses)
        self.delay = delay

    async def respond(self, tool_results=None) -> AgentResponse:
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def reset(self) -> None:
        pass

    def add_user_message(self, content: str) -> None:
        pass

    def add_assistant_message(self, content: str) -> None:
        pass


class ScriptedUser:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.seen: list[str] = []

    async def respond(self, conversation: list[Message]) -> Message:
        self.seen.extend(message.content for message in conversation)
        if self.delay:
            await asyncio.sleep(self.delay)
        return Message(role=Role.USER, content="Tell me more.")


class CountingToolkit:
    mode = "tools"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, name: str, arguments: dict) -> str:
        self.calls.append(name)
        return json.dumps({"status": "recommended", "id": "wd_1"} if name == "recommend" else [])


def _tool(name: str, index: int = 0, **arguments) -> AgentResponse:
    return AgentResponse(message=None, tool_calls=[{"id": f"call_{index}", "name": name, "arguments": arguments}])


ASKS = AgentResponse(message="What genre would you like?")
RECOMMENDS = _tool("recommend", item_id="wd_1")


async def test_tool_budget_answers_with_an_error_and_ends_a_runaway_loop():
    """A per-step tool budget answers with errors and ends a runaway loop."""
    profile = dataclasses.replace(BOOKS_V4, max_tool_calls_per_step=3, budget_errors_before_stop=2)
    toolkit = CountingToolkit()
    orchestrator = Orchestrator(ScriptedAgent([_tool("search_catalog", query="x")]), ScriptedUser(),
                                toolkit, max_turns=5, spec=BOOKS, harness=profile)
    trace = await orchestrator.run("t1", "m", 0)
    assert trace.stop_reason == StopReason.TOOL_BUDGET
    assert toolkit.calls == ["search_catalog"] * 3
    results = [event.result for event in trace.events if isinstance(event, ToolCallEvent)]
    assert len(results) == 5
    assert all("budget" in json.loads(result)["error"] for result in results[3:])

    # The budget is per step, and recommend() is always allowed.
    searches = [_tool("search_catalog", i, query="x") for i in range(4)]
    toolkit = CountingToolkit()
    agent = ScriptedAgent([*searches[:3], ASKS, *searches, RECOMMENDS])
    trace = await Orchestrator(agent, ScriptedUser(), toolkit, max_turns=5, spec=BOOKS,
                               harness=profile).run("t1", "m", 0)
    assert trace.stop_reason == StopReason.RECOMMENDED
    assert toolkit.calls == ["search_catalog"] * 6 + ["recommend"]


async def test_trial_cap_counts_only_the_agent_time():
    """The trial cap counts only agent time."""
    legacy = Orchestrator(ScriptedAgent([ASKS]), ScriptedUser(delay=0.05), CountingToolkit(),
                          max_turns=4, spec=BOOKS, trial_timeout_seconds=0.12)
    assert (await legacy.run("t1", "m", 0)).stop_reason == StopReason.TIMEOUT

    patient = Orchestrator(ScriptedAgent([ASKS]), ScriptedUser(delay=0.05), CountingToolkit(),
                           max_turns=4, spec=BOOKS, trial_timeout_seconds=0.12, harness=BOOKS_V4)
    assert (await patient.run("t1", "m", 0)).stop_reason == StopReason.TURN_LIMIT

    slow = Orchestrator(ScriptedAgent([ASKS], delay=0.05), ScriptedUser(), CountingToolkit(),
                        max_turns=10, spec=BOOKS, trial_timeout_seconds=0.12, harness=BOOKS_V4)
    assert (await slow.run("t1", "m", 0)).stop_reason == StopReason.WALL_CAP


async def test_early_stops_have_distinct_reasons():
    """Early stops get distinct reasons."""
    trace = await Orchestrator(ScriptedAgent([ASKS]), ScriptedUser(), CountingToolkit(), max_turns=2,
                               spec=BOOKS, harness=BOOKS_V4).run("t1", "m", 0)
    assert trace.stop_reason == StopReason.TURN_LIMIT

    class Overflowing(ScriptedAgent):
        async def respond(self, tool_results=None):
            raise RuntimeError("This model's maximum context length is 262144 tokens.")

    trace = await Orchestrator(Overflowing([ASKS]), ScriptedUser(), CountingToolkit(), spec=BOOKS,
                               harness=BOOKS_V4).run("t1", "m", 0)
    assert trace.stop_reason == StopReason.CONTEXT_OVERFLOW
    # The legacy harness keeps "timeout", so published runs read as before.
    trace = await Orchestrator(Overflowing([ASKS]), ScriptedUser(), CountingToolkit(), spec=BOOKS).run("t1", "m", 0)
    assert trace.stop_reason == StopReason.TIMEOUT


async def test_agent_text_sent_with_tool_calls_is_logged_but_not_shown_to_the_user():
    """Text sent with tool calls is logged, not shown to the user."""
    talking = AgentResponse(message="Let me look that up.", tool_calls=[
        {"id": "call_1", "name": "search_catalog", "arguments": {"query": "x"}}])
    user = ScriptedUser()
    trace = await Orchestrator(ScriptedAgent([talking, ASKS, RECOMMENDS]), user, CountingToolkit(),
                               spec=BOOKS, harness=BOOKS_V4).run("t1", "m", 0)
    kinds = [type(event).__name__ for event in trace.events]
    assert kinds.index("AgentToolTextEvent") == kinds.index("ToolCallEvent") - 1
    assert [e.content for e in trace.events if isinstance(e, AgentToolTextEvent)] == ["Let me look that up."]
    assert "Let me look that up." not in user.seen

    legacy = await Orchestrator(ScriptedAgent([talking, ASKS, RECOMMENDS]), ScriptedUser(), CountingToolkit(),
                                spec=BOOKS).run("t1", "m", 0)
    assert not any(isinstance(event, AgentToolTextEvent) for event in legacy.events)


def _completion(content: str, finish_reason: str = "stop"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=None)


async def test_agent_caps_output_cancels_hung_calls_and_resamples_empty_replies(monkeypatch):
    """Agent calls cap output, cancel hung calls, and resample empty replies."""
    import litellm

    real_sleep = asyncio.sleep

    async def no_backoff(seconds):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_backoff)
    calls: list[dict] = []

    async def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            await asyncio.Event().wait()  # hangs until cancelled
        return _completion("", "length") if len(calls) == 2 else _completion("Hello")

    monkeypatch.setattr(litellm, "acompletion", completion)
    profile = dataclasses.replace(BOOKS_V4, call_timeout_seconds=0.05)
    agent = LiteLLMAgent("openrouter/qwen/qwen3.8-27b", "s", [], harness=profile)
    result = await agent.respond()
    assert result.message == "Hello"
    assert [call["max_tokens"] for call in calls] == [32_768] * 3
    log = agent.call_log
    assert [row["status"] for row in log] == ["error", "empty", "ok"]
    assert log[0]["error_type"] == "TimeoutError" and log[0]["retrying"] is True
    assert [log[1]["finish_reason"], log[2]["finish_reason"]] == ["length", "stop"]
    assert agent._message_history == [{"role": "assistant", "content": "Hello"}]

    calls.clear()
    legacy = LiteLLMAgent("openrouter/qwen/qwen3.8-27b", "s", [])
    calls.append({})  # skip the hang
    await legacy.respond()
    assert "max_tokens" not in calls[1] and "finish_reason" not in legacy.call_log[0]


async def test_llm_user_calls_are_cancelled_when_they_hang(monkeypatch):
    import litellm

    from tau_rec.simulator.user_simulator import UserSimulator

    real_sleep = asyncio.sleep

    async def no_backoff(seconds):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_backoff)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            await asyncio.Event().wait()
        return _completion("I'd like a mystery.")

    monkeypatch.setattr(litellm, "acompletion", completion)
    task = _format_task(policy_flags=["single_recommendation"], user_formats=[])
    profile = dataclasses.replace(BOOKS_V4, call_timeout_seconds=0.05)
    simulator = UserSimulator("openrouter/z-ai/glm-5.3-flash", task, spec=BOOKS, harness=profile)
    reply = await simulator.respond([Message(role=Role.AGENT, content="Hi! What would you like?")])
    assert reply.content == "I'd like a mystery."
    assert [row["status"] for row in simulator.call_log] == ["error", "ok"]
    assert simulator.call_log[0]["error_type"] == "TimeoutError"


def test_books_v2_track_runs_under_the_v4_harness_and_movies_stay_legacy():
    assert load_track(V2_TRACK).harness_profile == "books-v4"
    assert find_track("movies-v1", ROOT).harness_profile == "legacy"


# --- Symbolic user (controller v4) and candidate verdicts ---------------------------

from tau_rec.simulator.user_simulator import (  # noqa: E402
    BOOKS_CONTROLLER_VERSION, BOOKS_GROUNDING_VERSION, BookDisclosureController, UserSimulator,
    classify_book_question,
)


def _constraint(field: str, op: str, value, reveal: RevealTag) -> TaskConstraint:
    return TaskConstraint(constraint=Constraint(field=field, op=op, value=value), reveal=reveal)


def _user_task(reveal: str = "hidden") -> Task:
    """Genre split across tags, an askable language, a hidden year, formats and history."""
    tags = {
        "volunteer": [RevealTag.VOLUNTEER] * 4,
        "mixed": [RevealTag.VOLUNTEER, RevealTag.ON_ASK, RevealTag.ON_ASK, RevealTag.ON_ASK],
        "hidden": [RevealTag.VOLUNTEER, RevealTag.HIDDEN, RevealTag.ON_ASK, RevealTag.HIDDEN],
    }[reveal]
    return Task(
        id=f"task_001_{reveal[0]}", persona="p", complexity="medium", reveal_difficulty=reveal,
        constraints=[
            _constraint("genres", "contains", "Mystery", tags[0]),
            _constraint("genres", "contains", "Fantasy", tags[1]),
            _constraint("languages", "contains", "French", tags[2]),
            _constraint("publication_year", ">=", 1950, tags[3]),
        ],
        policy_flags=["history", "format_verification_v2", "single_recommendation", "recommend_tool"],
        user_formats=["ebook"], user_id="user_1",
        user_history={"user_1": {"read": ["wd_read"], "ratings": {}}},
    )


def _user_catalog() -> Catalog:
    facts = {"genres": ["Mystery", "Fantasy"], "languages": ["French"], "countries": ["France"],
             "publication_year": 1960}
    return Catalog(items=[
        _book("wd_ok", ["ebook", "paperback"], title="Later", **facts),
        _book("wd_read", ["ebook"], **facts),
        _book("wd_paper", ["paperback"], **facts),
        _book("wd_bad", ["ebook"], genres=["Mystery"], languages=["French"], publication_year=1960),
    ])


def test_symbolic_replies_tell_withheld_unset_and_restated_facts_apart():
    """Withheld, unset and restated facts get distinct replies."""
    controller = BookDisclosureController(_user_task("hidden"))
    opening = controller.respond("Hi! What kind of book are you in the mood for?").content
    assert opening == "I'm looking for one book. It should include Mystery. I want something I haven't already read."

    assert controller.respond("What language should it be in?").content == (
        "To answer your question: its language should include French.")
    assert controller.respond("Sorry, which language was that?").content == (
        "To answer your question: its language should include French.")
    assert controller.respond("Any preference on the country of origin?").content == "No preference on that."
    assert controller.respond("Does the publication year matter?").content == "I'd rather not say."
    assert controller.respond("Any other genre preferences?").content == (
        "To answer your question: it should include Mystery. I'd rather not say more about the genre.")
    assert controller.respond("Which formats can you use?").content == "I can use any of these formats: ebook."
    assert controller.respond("Do you have a preferred language or country?").content == (
        "To answer your question: its language should include French. No preference on the country of origin.")
    assert controller.respond("Anything else I should know?").content == "That's everything I can share."
    assert controller.respond("Thanks, searching now.").content == (
        "Please ask about anything you need to narrow down one book.")
    hidden = {1, 3}
    assert not any(hidden & set(d["disclosed_constraint_indices"] + d.get("restated_constraint_indices", []))
                   for d in controller.decisions)
    assert all(d["controller_version"] == BOOKS_CONTROLLER_VERSION == "book-disclosure-v4"
               for d in controller.decisions)


def test_volunteer_opening_states_formats():
    """The volunteer opening states the accepted formats."""
    controller = BookDisclosureController(_user_task("volunteer"))
    assert controller.respond("Hello!").content == (
        "I'm looking for one book. It should include Mystery; and it should include Fantasy; and its "
        "language should include French; and it should have been published no earlier than 1950. "
        "I can use any of these formats: ebook. I want something I haven't already read.")
    assert controller.decisions[0]["disclosed_policy_facts"] == ["accepted_formats", "history_requirement"]
    mixed = BookDisclosureController(_user_task("mixed"))
    assert "formats" not in mixed.respond("Hello!").content


def _simulator(reveal: str = "mixed", mode: str = "symbolic") -> UserSimulator:
    return UserSimulator("must-not-be-called", _user_task(reveal), BOOKS, _user_catalog(), interaction_mode=mode)


async def _say(simulator: UserSimulator, history: list[Message], text: str) -> str:
    history.append(Message(role=Role.AGENT, content=text))
    reply = await simulator.respond(list(history))
    history.append(reply)
    return reply.content


async def test_answers_inside_candidate_turns_use_a_neutral_connector():
    """Answers inside candidate turns use a neutral connector."""
    simulator, history = _simulator(), []
    await _say(simulator, history, "Hi! What kind of book are you in the mood for?")
    reply = await _say(simulator, history, "How about Work wd_bad [[candidate_id:wd_bad]]? What language do you want?")
    assert reply.startswith("###REJECTED###")
    assert reply.endswith("To answer your question: its language should include French.")


async def test_verdicts_come_only_from_candidate_markers():
    """Verdicts come only from candidate markers, never from title matches."""
    catalog = _user_catalog()
    kit = ToolKit(catalog, CatalogSearch(catalog, BOOKS), {}, spec=BOOKS)
    agent = ScriptedAgent([
        _tool("search_catalog", query="Later"),
        AgentResponse(message="Would you like something published in 1999 or later? I recommend Later."),
        RECOMMENDS,
    ])
    trace = await Orchestrator(agent, _simulator(), kit, spec=BOOKS, harness=BOOKS_V4).run("t1", "m", 0)
    replies = [event.content for event in trace.events
               if isinstance(event, MessageEvent) and event.role == Role.USER]
    assert "###" not in replies[1]
    assert replies[1] == "To answer your question: it should have been published no earlier than 1950."


async def test_only_eligible_books_are_accepted():
    """Books already read or in excluded formats are rejected."""
    for mode in ("symbolic", "llm"):
        simulator, history = _simulator("mixed", mode), []
        if mode == "llm":
            history.append(Message(role=Role.USER, content="I'd like a book."))
        else:
            await _say(simulator, history, "Hi!")
        assert (await _say(simulator, history, "Try Work wd_read [[candidate_id:wd_read]].")).startswith("###REJECTED###")
        # Before formats are named, a paperback-only book can still be accepted.
        assert (await _say(simulator, history, "Try Work wd_paper [[candidate_id:wd_paper]].")).startswith("###ACCEPTED###")
        history.append(Message(role=Role.USER, content="I can use any of these formats: ebook."))
        assert (await _say(simulator, history, "Try Work wd_paper [[candidate_id:wd_paper]].")).startswith("###REJECTED###")
        assert (await _say(simulator, history, "Try Later [[candidate_id:wd_ok]].")).startswith("###ACCEPTED###")
        verdicts = [d for d in simulator.decision_log if d["action"] == "candidate_verdict"]
        assert [(d["candidate_id"], d["verdict"], d["failed_checks"]) for d in verdicts] == [
            ("wd_read", "rejected", ["already_read"]),
            ("wd_paper", "accepted", []),
            ("wd_paper", "rejected", ["format"]),
            ("wd_ok", "accepted", []),
        ], mode
    assert BOOKS_GROUNDING_VERSION == "candidate-id-v5"


async def test_malformed_candidate_markers_are_read():
    """Malformed and truncated candidate markers are parsed."""
    simulator, history = _simulator(), []
    await _say(simulator, history, "Hi!")
    assert (await _say(simulator, history, "Try Later [[candidate_id:wd_ok].")).startswith("###ACCEPTED###")
    assert (await _say(simulator, history, "Try Later [[ candidate_id = `wd_ok` ]].")).startswith("###ACCEPTED###")
    reply = await _say(simulator, history, "Later [[candidate_id:wd_ok]] or Work wd_read [[candidate_id:wd_re")
    assert "###" not in reply and "one catalog title" in reply


def test_generic_requests_and_check_ins():
    """Generic requests are recognised; check-ins are not fact requests."""
    for text in ("What requirement did that choice miss?", "What requirement is failing?",
                 "What was missing from the previous picks?", "What's off about it?",
                 "Can you tell me the remaining requirements?", "Could you give me one more hint?"):
        assert classify_book_question(text).generic, text
    assert classify_book_question("Is recency important to you?").fields == ("publication_year",)
    for text in ("Does that sound more like what you're after?",
                 "Is this closer to what you're looking for?",
                 "Does this one meet what you're looking for?"):
        intent = classify_book_question(text)
        assert not intent.generic and not intent.fields, text
    assert classify_book_question(
        "I'd like to suggest a novel published no earlier than 2010.").kind == "none"
    assert "publication_year" not in classify_book_question(
        "Are you interested in a particular era or place?").fields


# --- LLM user (books) ------------------------------------------------------------------

from tau_rec.simulator.user_simulator import (  # noqa: E402
    books_user_versions, build_simulator_prompt,
)


def test_books_llm_user_has_one_contract():
    """The books LLM user gets the same contract on every task and gives no verdicts."""
    satisfiable = _user_task("hidden")
    unsatisfiable = satisfiable.model_copy(update={"no_valid_recommendation": True})
    prompt = build_simulator_prompt(satisfiable, BOOKS)
    assert prompt == build_simulator_prompt(unsatisfiable, BOOKS)
    for phrase in ("Push back", "hint at why", "impossible", "commit to a single recommendation"):
        assert phrase not in prompt, phrase
    assert "Never write ###ACCEPTED### or ###REJECTED###" in prompt


def test_books_llm_user_first_message_and_reply_rules():
    """The first message holds only volunteer facts."""
    mixed = build_simulator_prompt(_user_task("mixed"), BOOKS)
    volunteer = build_simulator_prompt(_user_task("volunteer"), BOOKS)
    hidden = build_simulator_prompt(_user_task("hidden"), BOOKS)
    assert "never volunteer" in mixed and "including in your first message" in mixed
    assert "You read in these formats: ebook. State them in your first message." in volunteer
    assert "Share them only if the assistant asks which formats you read." in mixed
    assert "You want a book you have not already read." in mixed.split("**Share only if asked**")[0]
    assert "say you'd rather not say" in hidden and "say you'd rather not say more" in hidden
    assert "no preference" in hidden and "repeat it" in hidden
    no_formats = build_simulator_prompt(_user_task("mixed").model_copy(update={"user_formats": []}), BOOKS)
    assert "say you have no preference on format" in no_formats


async def test_llm_embedded_answer_sees_the_full_turn_and_the_verdict(monkeypatch):
    """The embedded-answer call sees the full turn and the verdict."""
    import litellm

    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return _completion("Ideally French.")

    monkeypatch.setattr(litellm, "acompletion", completion)
    simulator = _simulator("mixed", mode="llm")
    history = [Message(role=Role.AGENT, content="Hi!"), Message(role=Role.USER, content="I'd like a mystery.")]
    turn = "How about Work wd_bad [[candidate_id:wd_bad]]? Also, which language do you prefer?"
    reply = await simulator.respond([*history, Message(role=Role.AGENT, content=turn)])
    assert reply.content.startswith("###REJECTED###") and reply.content.endswith("Ideally French.")
    messages = calls[0]["messages"]
    assert messages[-1]["content"] == turn
    assert "doesn't satisfy all of my requirements" in messages[0]["content"]


def test_books_user_versions_are_recorded():
    assert books_user_versions("symbolic") == {
        "proposal_grounding": "candidate-id-v5", "disclosure_controller": "book-disclosure-v4"}
    assert books_user_versions("llm") == {
        "proposal_grounding": "candidate-id-v5", "user_prompt": "books-llm-user-v3",
        "embedded_answers": "embedded-answer-v2"}


def test_run_manifests_can_pin_the_harness_commit():
    """Run manifests can pin the harness commit."""
    from tau_rec.harness import harness_commit

    if not (Path(__file__).resolve().parents[1] / ".git").exists():
        pytest.skip("needs a git checkout")
    commit = harness_commit()
    assert commit is not None and len(commit["sha"]) == 40 and isinstance(commit["dirty"], bool)


def test_classic_as_a_kind_of_book_and_more_open_requests():
    """More genre phrasings and open requests."""
    assert classify_book_question("Would you enjoy a classic whodunit?").fields == ("genres",)
    assert classify_book_question("Would a classic pirate adventure appeal to you?").fields == ("genres",)
    assert "publication_year" in classify_book_question("Classic or modern mystery?").fields
    assert classify_book_question("Would you like a classic?").fields == ("publication_year",)
    for text in ("What part isn't matching?", "Could you tell me what the previous picks were missing?",
                 "Could you tell me a bit about your taste?", "What's not quite right about it?",
                 "What are you in the mood for this time?",
                 "Could you tell me what requirement the previous suggestions missed?"):
        assert classify_book_question(text).generic, text
    assert not classify_book_question("What kind of horror are you in the mood for?").generic


def test_more_genre_and_open_request_phrasings():
    assert classify_book_question("Does a mystery sound appealing?").fields == ("genres",)
    assert classify_book_question("Could you tell me what missed the mark on the last pick?").generic
    assert not classify_book_question("What kind of book are you in the mood for?").generic
    assert "genres" in classify_book_question("What kind of book are you in the mood for?").fields
    assert "publication_year" not in classify_book_question("Would you prefer a children's classic like this?").fields
    assert "publication_year" not in classify_book_question(
        "Would you prefer a focus on a particular place, era, or event?").fields
    # A bare "era or setting" is read as an era.
    assert "publication_year" in classify_book_question("Any particular era or setting?").fields
    assert {"countries", "publication_year"} <= set(classify_book_question(
        "Do you have a preference for a particular region, era, or style?").fields)



def test_books_llm_user_first_message_is_only_the_volunteer_facts():
    """The LLM user's first message states only the volunteer facts."""
    prompt = build_simulator_prompt(_user_task("mixed"), BOOKS)
    assert "Your first message states only the items under \"State in your first message\"" in prompt
    assert "do not mention any other genre, language, country, publication year or format" in prompt
    assert "do not say that you have other requirements" in prompt
    assert books_user_versions("llm")["user_prompt"] == "books-llm-user-v3"
