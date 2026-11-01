"""Bound on the symbolic user's recogniser misreads (extract_disclosure_turns.py, score_disclosure_misreads.py)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis/books-controlled"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACK = ROOT / "data/tracks/books-wikidata-v2"


def _live_trace(task_id, agent_msgs, success=True):
    """A trace whose user turns and decision log come from the real symbolic user."""
    import asyncio

    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.conversation import Message, Role
    from tau_rec.data_model.task import Task
    from tau_rec.simulator.user_simulator import UserSimulator
    from tau_rec.track import load_track

    track = load_track(TRACK)
    catalog = Catalog.from_json(str(TRACK / "catalog.json"), track.spec)
    simulator = UserSimulator("unused", Task.from_json(str(TRACK / "tasks" / f"{task_id}.json")), track.spec,
                              catalog, interaction_mode="symbolic")
    conversation, events = [], []

    async def run():
        for text in ["Hi! What kind of book are you looking for?", *agent_msgs]:
            conversation.append(Message(role=Role.AGENT, content=text))
            reply = await simulator.respond(conversation)
            conversation.append(reply)
            events.extend([{"type": "tool_call", "name": "search_catalog", "arguments": {}, "result": "[]"},
                           {"type": "message", "role": "agent", "content": text},
                           {"type": "message", "role": "user", "content": reply.content}])

    asyncio.run(run())
    return {"task_id": task_id, "events": events, "simulator_decisions": simulator.decision_log,
            "evaluation": {"primary_reward": 1.0 if success else 0.0}}


def _write_cell(root, stage, agent, iface, traces):
    d = root / stage / agent / iface / "20000" / "20261014_000000" / "traces"
    d.mkdir(parents=True)
    for i, t in enumerate(traces):
        (d / f"{t['task_id']}_trial{i + 1}.json").write_text(json.dumps(t))


def test_packets_hide_recogniser_output(tmp_path):
    ext = _load("extract_disclosure_turns")
    trace = _live_trace("task_060_m", ["Do you have a publication year range in mind?"])
    _write_cell(tmp_path, "reliability", "glm53_flash_high", "structured", [trace])
    packets, key = ext.extract(tmp_path, ["task_060_m"], TRACK)
    assert packets == [{"uid": 0, "text": "Do you have a publication year range in mind?"}]
    assert set(key["0"]) == {"source", "agent", "interface", "intent", "requested_fields", "disclosed_fields",
                             "success"}
    assert key["0"]["disclosed_fields"] == ["publication_year"]


def test_only_mixed_disclosures_on_listed_tasks_are_extracted(tmp_path):
    ext = _load("extract_disclosure_turns")
    ask = ["Do you have a publication year range in mind?"]
    keep = _live_trace("task_060_m", ask)
    hidden = _live_trace("task_060_h", ask)
    nothing = _live_trace("task_060_m", ["Great, let me search."])
    off_panel = _live_trace("task_061_m", ask)
    _write_cell(tmp_path, "primary_remainder", "qwen38_27b_high", "keyword", [keep, hidden, nothing, off_panel])
    packets, key = ext.extract(tmp_path, ["task_060_m", "task_060_h"], TRACK)
    assert len(packets) == 1 and key["0"]["source"].endswith("task_060_m_trial1.json#decision1")


def test_each_disclosure_maps_to_the_agent_turn_that_prompted_it(tmp_path):
    """One user reply can log a verdict and a disclosure; later decisions must not shift turns."""
    ext = _load("extract_disclosure_turns")
    candidate = json.loads((TRACK / "catalog.json").read_text())[0]["id"]
    year = f"How about this one [[candidate_id:{candidate}]]? Also, any publication year range in mind?"
    genre = "Thanks. Any other genres you want it to include?"
    trace = _live_trace("task_060_m", [year, genre])
    assert [d["action"] for d in trace["simulator_decisions"]] == ["opening", "candidate_verdict", "disclose",
                                                                  "disclose"]
    _write_cell(tmp_path, "reliability", "deepseek_v41_flash_high", "structured", [trace])
    packets, key = ext.extract(tmp_path, ["task_060_m"], TRACK)
    by_field = {tuple(key[str(p["uid"])]["disclosed_fields"]): p["text"] for p in packets}
    assert by_field == {("publication_year",): year, ("genres",): genre}


def test_extraction_refuses_a_trace_whose_decisions_do_not_replay(tmp_path):
    import pytest

    ext = _load("extract_disclosure_turns")
    trace = _live_trace("task_060_m", ["Do you have a publication year range in mind?"])
    trace["simulator_decisions"][1]["disclosed_fields"] = ["languages"]
    _write_cell(tmp_path, "reliability", "glm53_flash_high", "structured", [trace])
    with pytest.raises(SystemExit, match="do not replay"):
        ext.extract(tmp_path, ["task_060_m"], TRACK)


def test_fp_rule():
    score = _load("score_disclosure_misreads")
    field = {"intent": "field", "requested_fields": ["genres"], "disclosed_fields": ["genres"]}
    assert score.false_positive(field, {"fields": ["languages"], "generic": False}) is True
    assert score.false_positive(field, {"fields": ["genres"], "generic": False}) is False
    generic = {"intent": "generic", "requested_fields": [], "disclosed_fields": ["countries"]}
    assert score.false_positive(generic, {"fields": [], "generic": False}) is True
    assert score.false_positive(generic, {"fields": [], "generic": True}) is False


def test_lower_bound_counts_fp_successes_as_failures():
    score = _load("score_disclosure_misreads")
    key, labels = {}, {}
    trials = {"t1": (True, True), "t2": (True, False), "t3": (True, False), "t4": (False, False)}
    uid = 0
    for trial, (success, misread) in trials.items():
        key[str(uid)] = {"source": f"x/{trial}.json#turn1", "agent": "glm53_flash_high", "interface": "structured",
                         "intent": "field", "requested_fields": ["genres"], "disclosed_fields": ["genres"],
                         "success": success}
        labels[str(uid)] = {"fields": [] if misread else ["genres"], "generic": False}
        uid += 1
    mixed = {("glm53_flash_high", "structured"): {f"x/{t}.json": s for t, (s, _) in trials.items()}}
    result = score.score(key, labels, mixed)
    row = result["by_cell"]["glm53_flash_high/structured"]
    assert row["mixed_successes"] == 3 and row["successes_with_fp_disclosure"] == 1
    assert row["pass1"] == 0.75 and row["pass1_lower_bound"] == 0.5


def test_generic_request_covers_an_extra_fact_on_a_field_turn():
    """Field questions plus 'tell me more' advance one extra fact by design; that is not a misread."""
    score = _load("score_disclosure_misreads")
    entry = {"intent": "field", "requested_fields": ["genres"], "disclosed_fields": ["genres", "publication_year"]}
    assert score.false_positive(entry, {"fields": ["genres"], "generic": True}) is False
    assert score.false_positive(entry, {"fields": ["genres"], "generic": False}) is True


def test_a_generic_request_covers_only_one_unasked_fact():
    """The controller advances one fact per generic request, so a second unasked fact is a misread."""
    score = _load("score_disclosure_misreads")
    entry = {"intent": "generic", "requested_fields": [], "disclosed_fields": ["countries", "languages"]}
    assert score.false_positive(entry, {"fields": [], "generic": True}) is True
    assert score.false_positive(entry, {"fields": ["languages"], "generic": True}) is False
