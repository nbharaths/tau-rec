"""The one-run books report (scripts/report_books_run.py); no provider is called."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACK = ROOT / "data/tracks/books-wikidata-v2"
SCRIPT = ROOT / "scripts/report_books_run.py"


def _module():
    spec = importlib.util.spec_from_file_location("report_books_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(task_id: str, trial: int, *, success: bool, stop: str, disclosed=(), exposed=()) -> dict:
    return {
        "task_id": task_id, "model": "m", "trial": trial, "stop_reason": stop,
        "events": [{"type": "tool_call", "name": "search_catalog", "arguments": {},
                    "result": json.dumps([{"id": item} for item in exposed])}],
        "evaluation": {"primary_reward": 1.0 if success else 0.0},
        "simulator_decisions": [{"action": "disclose", "disclosed_constraint_indices": list(disclosed)}],
    }


def _run_dir(tmp_path: Path) -> Path:
    answers = json.loads((TRACK / "answers.json").read_text())
    eligible = answers["task_060_m"]["policy_eligible_solutions"][0]
    run = tmp_path / "run"
    (run / "traces").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({
        "model": "m", "retrieval_mode": "structured", "catalog_size": 20000,
        "simulator_interaction_mode": "symbolic", "track": "books-wikidata-v2",
    }))
    traces = [
        _trace("task_060_v", 0, success=True, stop="recommended"),
        _trace("task_060_v", 1, success=True, stop="recommended"),
        # task_060_m: Young Adult (index 1) and the year bound (index 2) are on_ask.
        _trace("task_060_m", 0, success=True, stop="recommended", disclosed=(1, 2), exposed=(eligible,)),
        _trace("task_060_m", 1, success=False, stop="wall_cap"),
        _trace("task_060_h", 0, success=False, stop="recommended"),
        _trace("task_060_h", 1, success=False, stop="turn_limit"),
    ]
    for trace in traces:
        (run / "traces" / f"{trace['task_id']}_trial{trace['trial']}.json").write_text(json.dumps(trace))
    return run


def test_report_counts_conditions_elicitation_exposure_and_stops(tmp_path):
    report = _module().report(_run_dir(tmp_path), TRACK)
    assert report["tasks"] == 3 and report["trials"] == 6
    assert report["pass^1"]["volunteer"]["estimate"] == 1.0
    assert report["pass^1"]["mixed"]["estimate"] == 0.5
    assert report["pass^1"]["hidden"]["estimate"] == 0.0
    assert report["pass^1"]["all"]["estimate"] == 0.5
    assert report["pass^4"] is None  # fewer than four trials per task
    assert report["mixed_complete_elicitation"] == 0.5
    assert report["mixed_valid_exposure"] == 0.5
    assert report["stop_classes"] == {"recommended": 4, "turn_limit": 1, "wall_cap": 1}


def test_report_skips_controller_measures_for_llm_users(tmp_path):
    run = _run_dir(tmp_path)
    manifest = json.loads((run / "run_manifest.json").read_text())
    manifest["simulator_interaction_mode"] = "llm"
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    report = _module().report(run, TRACK)
    assert report["mixed_complete_elicitation"] is None
    assert report["mixed_valid_exposure"] == 0.5


def test_report_caps_hidden_success_at_three_rejected_proposals(tmp_path):
    run = _run_dir(tmp_path)
    rejected = {"action": "candidate_verdict", "verdict": "rejected"}
    for trial, rejections in ((0, 3), (1, 4)):
        trace = _trace("task_060_h", trial, success=True, stop="recommended")
        trace["simulator_decisions"] = [rejected] * rejections
        (run / "traces" / f"task_060_h_trial{trial}.json").write_text(json.dumps(trace))
    report = _module().report(run, TRACK)
    assert report["pass^1"]["hidden/satisfiable"]["estimate"] == 1.0
    assert report["pass^1"]["hidden/satisfiable/within_3_rejections"]["estimate"] == 0.5
