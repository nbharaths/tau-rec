"""--resume fills only the (task, trial) pairs with no trace, and results are
always rebuilt from the traces on disk."""
import json
import shutil
from pathlib import Path

from click.testing import CliRunner

from tau_rec.cli import _existing_trials, _results_from_traces, main as cli
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.evaluator.evaluator import CombinedEvaluator

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_TRACE = ROOT / "tests" / "fixtures" / "sample_trace.json"


def test_existing_trials_parses_filenames(tmp_path):
    for name in ("task_001_trial0.json", "task_001_trial3.json", "task_012_trial1.json", "notes.txt", "task_x_trialq.json"):
        (tmp_path / name).write_text("{}")
    assert _existing_trials(tmp_path) == {("task_001", 0), ("task_001", 3), ("task_012", 1)}


def test_results_are_rebuilt_from_traces_on_disk(tmp_path):
    """Two traces for one task, none for another: n counts what exists."""
    traces = tmp_path / "traces"
    traces.mkdir()
    for i in (0, 2):
        shutil.copy(FIXTURE_TRACE, traces / f"task_001_trial{i}.json")
    catalog = Catalog.from_json(str(ROOT / "data" / "catalog.json"))
    tasks = {tid: Task.from_json(str(ROOT / "data" / "tasks" / f"{tid}.json")) for tid in ("task_001", "task_002")}
    task_results, rows = _results_from_traces(traces, tasks, CombinedEvaluator(catalog))
    assert task_results["task_001"]["n"] == 2 and task_results["task_002"] == {"n": 0, "c": 0}
    assert [r["task_id"] for r in rows] == ["task_001", "task_001"]
    assert task_results["task_001"]["c"] == sum(1 for r in rows if r["primary_reward"] == 1.0)


def test_resume_refuses_a_non_run_directory_and_the_old_ablation(tmp_path):
    result = CliRunner().invoke(cli, ["run", "--resume", str(tmp_path)])
    assert result.exit_code != 0 and "not a run directory" in result.output

    run = tmp_path / "run" / "20260910_000000"
    (run / "traces").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"model": "m", "no_tools": True, "trials_per_task": 4}))
    result = CliRunner().invoke(cli, ["run", "--resume", str(run)])
    assert result.exit_code != 0 and "old --no-tools ablation" in result.output


def test_resume_refuses_drifted_content(tmp_path):
    run = tmp_path / "run" / "20260910_000000"
    (run / "traces").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({
        "model": "m", "trials_per_task": 4, "mode": "tools", "track": "movies-v1",
        "tasks_sha256": "0" * 64,
    }))
    result = CliRunner().invoke(cli, ["run", "--resume", str(run)])
    assert result.exit_code != 0 and "tasks changed since this run started" in result.output


def test_resume_refuses_drifted_search_profile(tmp_path):
    run = tmp_path / "run" / "20260910_000000"
    (run / "traces").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({
        "model": "m", "trials_per_task": 4, "mode": "tools", "track": "movies-v1",
        "search_profile": "bm25f-v1",
    }))
    result = CliRunner().invoke(cli, ["run", "--resume", str(run)])
    assert result.exit_code != 0 and "search profile changed" in result.output


def test_run_without_output_or_resume_is_a_usage_error():
    result = CliRunner().invoke(cli, ["run", "--track", "data"])
    assert result.exit_code != 0 and "--output" in result.output
