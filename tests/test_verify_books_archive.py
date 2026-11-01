"""The archive verifier (scripts/verify_books_archive.py) on a real, LLM-free trace."""
import asyncio
import importlib.util
import json
from pathlib import Path

from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.domain import BOOKS
from tau_rec.environment.tools import ToolKit
from tau_rec.evaluator.evaluator import CombinedEvaluator
from tau_rec.orchestrator.orchestrator import Orchestrator
from tau_rec.simulator.user_simulator import UserSimulator
from tau_rec.track import load_track

ROOT = Path(__file__).resolve().parent.parent
TRACK = ROOT / "data/tracks/books-wikidata-v2"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY = _load("verify_books_archive", ROOT / "scripts/verify_books_archive.py")
SCRIPTED = _load("run_scripted_reference", ROOT / "analysis/books-controlled/run_scripted_reference.py")


def _scripted_run(tmp_path: Path) -> Path:
    """One symbolic, structured trial of the scripted agent, written as `tau-rec run` writes it."""
    catalog = Catalog.from_json(str(TRACK / "catalog.json"), BOOKS)
    task = Task.from_json(str(TRACK / "tasks/task_060_m.json"))
    toolkit = ToolKit(catalog=catalog, search=None, user_histories=task.user_history or {}, spec=BOOKS,
                      mode="tools", retrieval_mode="structured", structured_profile="books-wikidata-core-v3")
    simulator = UserSimulator("none", task, BOOKS, catalog, interaction_mode="symbolic")
    agent = SCRIPTED.ScriptedAgent(task, "tool", seed=1)
    trace = asyncio.run(Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=20, spec=BOOKS,
                                     harness=load_track(TRACK).harness).run(task.id, "scripted", 0))
    doc = trace.model_dump(mode="json")
    doc["evaluation"] = CombinedEvaluator(catalog, BOOKS).evaluate(task, trace).model_dump(mode="json")
    doc["simulator_decisions"] = simulator.decision_log
    run = tmp_path / "run"
    (run / "traces").mkdir(parents=True)
    (run / "traces" / "task_060_m_trial0.json").write_text(json.dumps(doc, indent=2, default=str))
    (run / "run_manifest.json").write_text(json.dumps({
        "catalog_size": 20000, "retrieval_mode": "structured", "structured_profile": "books-wikidata-core-v3",
        "search_profile": "bm25f-v1", "mode": "tools", "simulator_interaction_mode": "symbolic",
    }))
    return run


def _verify(run: Path):
    VERIFY._setup(str(TRACK))
    counts, _ = VERIFY.verify_run(str(run / "run_manifest.json"))
    return counts


def _edit(run: Path, change) -> None:
    path = run / "traces" / "task_060_m_trial0.json"
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))


def test_a_faithful_trace_reproduces_every_check(tmp_path):
    counts = _verify(_scripted_run(tmp_path))
    assert counts["tool_calls_same"] > 0 and counts["scores_same"] == 1
    assert counts["user_replies_same"] > 0 and counts["decision_logs_same"] == 1
    assert not any(count for key, count in counts.items() if key.endswith("_differ"))


def test_tampered_tool_results_scores_and_replies_are_caught(tmp_path):
    run = _scripted_run(tmp_path)

    def tamper(doc):
        call = next(event for event in doc["events"] if event.get("type") == "tool_call")
        call["result"] = "[]" if call["result"] != "[]" else '{"error": "x"}'
        reply = next(event for event in doc["events"] if event.get("type") == "message" and event["role"] == "user")
        reply["content"] += " (edited)"
        doc["evaluation"]["primary_reward"] = 1.0 - doc["evaluation"]["primary_reward"]

    _edit(run, tamper)
    counts = _verify(run)
    assert counts["tool_calls_differ"] == 1
    assert counts["user_replies_differ"] >= 1
    assert counts["scores_differ"] == 1
