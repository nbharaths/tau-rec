"""The label-free bound on missed questions (recognizer_bounds.py)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "analysis/books-controlled/recognizer_bounds.py"


def _load():
    spec = importlib.util.spec_from_file_location("recognizer_bounds", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK = {"constraints": [{"reveal": "volunteer"}, {"reveal": "on_ask"}, {"reveal": "on_ask"}]}


def _mixed(*agent_turns, disclosed=(), success=False):
    """A mixed trial after the seeded greeting: each agent turn gets a user reply; decisions disclose the given indices."""
    events = [{"type": "message", "role": "agent", "content": "Hi! What kind of book are you in the mood for?"},
              {"type": "message", "role": "user", "content": "I'm looking for one book."}]
    for text in agent_turns:
        events += [{"type": "message", "role": "agent", "content": text},
                   {"type": "message", "role": "user", "content": "..."}]
    events.append({"type": "tool_call", "name": "recommend", "arguments": {"item_id": "x"}})
    return {"evaluation": {"primary_reward": 1.0 if success else 0.0}, "events": events,
            "simulator_decisions": [{"action": "disclose", "disclosed_constraint_indices": list(disclosed)}]}


def test_question_sentences_skip_candidate_proposals_and_statements():
    sentences = _load().question_sentences
    text = "Here are options. How about Dune [[candidate_id:wd_Q1]]? Do you have a favourite era? Great."
    assert sentences(text) == ["Do you have a favourite era?"]


def test_a_failure_with_every_askable_fact_disclosed_cannot_be_a_missed_question():
    classify = _load().classify_mixed_failure
    assert classify(_mixed("Any genre or year?", disclosed=[1, 2]), TASK) == "all_disclosed"


def test_a_failure_without_any_question_cannot_be_a_missed_question():
    classify = _load().classify_mixed_failure
    assert classify(_mixed("I will search now.", "Dune fits [[candidate_id:wd_Q1]]?"), TASK) == "no_question"


def test_a_question_the_recognizer_maps_to_nothing_is_a_possible_miss():
    classify = _load().classify_mixed_failure
    # The bound counts every question the recognizer maps to no field and no generic request, whatever it asks.
    assert classify(_mixed("Does that sound good?", disclosed=[1]), TASK) == "unrecognized_question"
    assert classify(_mixed("Which genre do you like?", disclosed=[1]), TASK) == "recognized_only"


def test_bound_counts_possible_misses_over_all_mixed_trials():
    bound = _load().missed_question_bound
    trials = [(_mixed("Does that sound good?"), TASK), (_mixed("Which genre?", disclosed=[1, 2]), TASK),
              (_mixed("Which genre?", disclosed=[1, 2], success=True), TASK), (_mixed("Searching."), TASK)]
    assert bound(trials) == {"trials": 4, "failures": 3, "all_disclosed": 1, "no_question": 1,
                             "recognized_only": 0, "unrecognized_question": 1, "max_gain": 0.25}


def test_the_seeded_greeting_is_not_the_agents_question():
    assert _load().classify_mixed_failure(_mixed("I will search now."), TASK) == "no_question"


FIELDED = {"constraints": [{"reveal": "volunteer", "constraint": {"field": "genres"}},
                           {"reveal": "on_ask", "constraint": {"field": "languages"}},
                           {"reveal": "on_ask", "constraint": {"field": "publication_year"}}]}


def test_a_labelled_question_about_a_field_left_undisclosed_is_a_miss():
    miss = _load().labelled_miss
    labels = {"Does that sound good?": {"fields": [], "generic": False},
              "Should it be in Spanish, maybe?": {"fields": ["languages"], "generic": False}}
    # Language stayed undisclosed, and the agent asked about it: the recognizer missed that question.
    assert miss(_mixed("Should it be in Spanish, maybe?", disclosed=[2]), FIELDED, labels)
    # A check-in asks about no field, and a question about a disclosed field cannot be a miss.
    assert not miss(_mixed("Does that sound good?", disclosed=[2]), FIELDED, labels)
    assert not miss(_mixed("Should it be in Spanish, maybe?", disclosed=[1]), FIELDED, labels)


def test_a_labelled_generic_request_counts_only_if_the_recognizer_saw_none_in_that_turn():
    miss = _load().labelled_miss
    labels = {"Is there something I am missing?": {"fields": [], "generic": True},
              "Any other requirements?": {"fields": [], "generic": True}}
    assert miss(_mixed("Is there something I am missing?"), FIELDED, labels)
    # The recognizer treats "Any other requirements?" as a generic request, which released a fact by design.
    assert not miss(_mixed("Any other requirements?"), FIELDED, labels)


def test_bound_with_labels_adds_the_labelled_gain():
    bound = _load().missed_question_bound
    labels = {"Should it be in Spanish, maybe?": {"fields": ["languages"], "generic": False},
              "Does that sound good?": {"fields": [], "generic": False}}
    trials = [(_mixed("Should it be in Spanish, maybe?"), FIELDED), (_mixed("Does that sound good?"), FIELDED),
              (_mixed("Searching.", success=True), FIELDED), (_mixed("Searching."), FIELDED)]
    out = bound(trials, labels)
    assert (out["labelled_misses"], out["max_gain_labelled"]) == (1, 0.25)
