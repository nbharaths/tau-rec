"""How hidden satisfiable trials fail (hidden_failures.py)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "analysis/books-controlled/hidden_failures.py"


def _load():
    spec = importlib.util.spec_from_file_location("hidden_failures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(success, recommended=None, verdicts=()):
    events = [{"type": "tool_call", "name": "recommend", "arguments": {"item_id": recommended}}] if recommended else []
    return {"evaluation": {"primary_reward": 1.0 if success else 0.0}, "events": events,
            "simulator_decisions": [{"action": "candidate_verdict", "candidate_id": item, "verdict": verdict}
                                    for item, verdict in verdicts]}


def test_failures_are_classified_by_what_the_final_recommendation_saw():
    classify = _load().classify
    assert classify(_trace(True, "a", [("a", "accepted")])) == "success"
    assert classify(_trace(False)) == "no_recommendation"
    assert classify(_trace(False, "b", [("a", "rejected")])) == "untested"
    assert classify(_trace(False, "a", [("a", "rejected")])) == "rejected"
    assert classify(_trace(False, "a", [("a", "accepted")])) == "accepted"


def _dialogue(*turns, recommended="dune"):
    """Events from (role, text) turns, then the committing turn: a recommend call and a closing message."""
    events = [{"type": "message", "role": role, "content": text} for role, text in turns]
    events += [{"type": "tool_call", "name": "recommend", "arguments": {"item_id": recommended}},
               {"type": "message", "role": "agent", "content": "Final pick: Dune [[candidate_id:dune]]."}]
    return {"evaluation": {"primary_reward": 0.0}, "events": events, "simulator_decisions": []}


TITLES = {"dune": "Dune", "wd_Q2": "It", "wd_Q12": "Solaris"}


def test_untested_commitment_named_in_an_earlier_turn_without_a_verdict():
    named_earlier = _load().named_earlier
    assert named_earlier(_dialogue(("agent", "How about *Dune* by Frank Herbert?"), ("user", "Please ask about anything.")),
                         TITLES)
    # An identifier without a valid single marker also names the book.
    assert named_earlier(_dialogue(("agent", "Candidates: dune, other"), ("user", "Please choose one.")), TITLES)


def test_untested_commitment_named_only_in_the_committing_turn_counts_as_unnamed():
    named_earlier = _load().named_earlier
    assert not named_earlier(_dialogue(("agent", "Any genre?"), ("user", "No preference.")), TITLES)
    # The committing turn names the book, but the user never had a chance to answer it.
    assert not named_earlier(_dialogue(("agent", "Any genre?"), ("user", "No preference."),
                                       ("agent", "Let me check Dune's formats.")), TITLES)


def test_short_titles_and_identifier_prefixes_do_not_count_as_naming():
    named_earlier = _load().named_earlier
    assert not named_earlier(_dialogue(("agent", "Is it for a gift?"), ("user", "No preference."), recommended="wd_Q2"),
                             TITLES)
    assert not named_earlier(_dialogue(("agent", "Maybe wd_Q123?"), ("user", "Please choose one."), recommended="wd_Q12"),
                             TITLES)


def test_feasibility_counts_hidden_tasks_by_who_solved_them():
    feasibility = _load().feasibility
    trials = {"a_h": [False, True], "b_h": [False, False], "c_h": [False], "d_h": [True], "e_h": []}
    scripted = {"a_h": False, "b_h": True, "c_h": False, "d_h": False, "e_h": False}
    nvr = {"a_h": False, "b_h": False, "c_h": False, "d_h": True, "e_h": True}
    assert feasibility(trials, scripted, nvr) == {
        "satisfiable": {"tasks": 3, "scripted": 1, "agents_only": 1, "unsolved": 1},
        "nvr": {"tasks": 2, "scripted": 0, "agents_only": 1, "unsolved": 1},
    }


def test_commitment_context_reads_the_verdicts_before_the_commit():
    context = _load().commitment_context
    assert context(_trace(False, "b")) == "no_proposal"
    assert context(_trace(False, "c", [("a", "accepted")])) == "no_rejection"
    assert context(_trace(False, "c", [("a", "accepted"), ("b", "rejected")])) == "after_rejection"


def test_untagged_acceptable_mention_needs_a_valid_unjudged_book_before_the_commit():
    mention = _load().untagged_acceptable
    trace = _dialogue(("agent", "Options: *Solaris* or *Dune*?"), ("user", "Please choose one."))
    assert mention(trace, {"wd_Q12"}, TITLES)  # Solaris would have been accepted
    assert not mention(trace, {"wd_Q99"}, TITLES)  # neither named book is valid
    judged = dict(trace, simulator_decisions=[{"action": "candidate_verdict", "candidate_id": "wd_Q12",
                                               "verdict": "rejected"}])
    assert not mention(judged, {"wd_Q12"}, TITLES)  # a book with a verdict was not left untested
    late = _dialogue(("agent", "Any genre?"), ("user", "No preference."), ("agent", "Solaris fits."))
    assert not mention(late, {"wd_Q12"}, TITLES)  # named only in the committing turn


def test_untagged_acceptable_mention_counts_in_trials_that_stop_without_a_recommendation():
    mention = _load().untagged_acceptable
    stopped = {"evaluation": {"primary_reward": 0.0}, "simulator_decisions": [],
               "events": [{"type": "message", "role": "agent", "content": "Maybe Solaris?"},
                          {"type": "message", "role": "user", "content": "Please choose one."}]}
    assert mention(stopped, {"wd_Q12"}, TITLES)


def test_refusal_received_reads_withheld_fields_in_the_user_decisions():
    refusal_received = _load().refusal_received
    asked = {"simulator_decisions": [{"action": "answer", "withheld_fields": []},
                                     {"action": "answer", "withheld_fields": ["genres"]}]}
    never = {"simulator_decisions": [{"action": "answer", "withheld_fields": [], "no_preference_fields": ["languages"]},
                                     {"action": "candidate_verdict", "verdict": "rejected"}]}
    # A field question that hit a hidden field drew a refusal, which tells the agent a requirement exists.
    assert refusal_received(asked)
    # "No preference" and verdicts reveal no requirement.
    assert not refusal_received(never)
    assert not refusal_received({"simulator_decisions": []})
