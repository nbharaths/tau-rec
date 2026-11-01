"""The v4 books matrix launcher (analysis/books-controlled/run_v4_matrix.py)."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "analysis/books-controlled/run_v4_matrix.py"
ARGS = [
    "--artifact", str(ROOT / "data/tracks/books-wikidata-v2"),
    "--panels", str(ROOT / "analysis/books-wikidata-v2/panels.json"),
    "--model-config", str(ROOT / "analysis/books-wikidata-v2/model_config_v4.json"),
    "--plan", str(ROOT / "analysis/books-wikidata-v2/experiment_plan_v4.json"),
]


def _launcher():
    spec = importlib.util.spec_from_file_location("v4_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stalls_are_wall_caps_with_time_outside_successful_agent_calls():
    launcher = _launcher()
    calls = [{"status": "ok", "latency_seconds": 200.0}, {"status": "error", "latency_seconds": 300.0}]
    assert launcher.stall_seconds({"stop_reason": "wall_cap", "api_calls": {"agent": calls}}, 600) == 400
    # Not wall-capped: a slow LLM user is not a stall under the agent-time cap.
    assert launcher.stall_seconds({"stop_reason": "recommended", "trial_runtime_s": 900}, 600) is None
    assert launcher.stages.stalled_traces is launcher.stalled_traces


def test_global_concurrency_is_shared_across_waves():
    launcher = _launcher()
    assert launcher.wave_concurrency(48, 6, 36) == 8
    assert launcher.wave_concurrency(48, 1, 200) == 1


def test_plan_only_mode_writes_every_selected_stage(tmp_path):
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), *ARGS, "--output", str(tmp_path),
         "--stages", "reliability,simulator_sensitivity"],
        capture_output=True, text=True, check=True,
    )
    assert "no provider calls made" in result.stdout
    assert {p.name for p in tmp_path.glob("*_v4_state.json")} == {
        "reliability_v4_state.json", "simulator_sensitivity_v4_state.json"}


def test_execute_refuses_a_draft_plan(tmp_path):
    plan = json.loads((ROOT / "analysis/books-wikidata-v2/experiment_plan_v4.json").read_text())
    plan["status"] = "draft-not-frozen"
    draft = tmp_path / "draft_plan.json"
    draft.write_text(json.dumps(plan))
    args = [*ARGS[:-2], "--plan", str(draft)]
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), *args, "--output", str(tmp_path / "out"), "--stages", "smoke",
         "--execute", "--max-provider-cost-usd", "1"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "the plan is a draft" in result.stderr
    assert not (tmp_path / "out" / "smoke").exists()


def test_no_test_can_start_a_provider_call(monkeypatch, tmp_path):
    """The launch point refuses to start provider calls under pytest."""
    launcher = _launcher()

    def no_popen(*args, **kwargs):
        raise AssertionError("a test reached subprocess.Popen")

    monkeypatch.setattr(launcher.stages.subprocess, "Popen", no_popen)
    cell = {"status": "planned", "run_dir": None, "key": "k", "stall_retries": 0}
    args = argparse.Namespace(
        max_provider_cost_usd=1.0, output=tmp_path, wave_concurrency=1, runtime={},
        plan={"stall_retry_rule": {"max_retries": 1, "threshold_seconds": 240}},
    )
    with pytest.raises(SystemExit, match="refusing to launch provider calls from a test"):
        launcher.stages._run_wave(args, {"name": "smoke"}, [cell], {}, tmp_path / "state.json", {}, 1)


def test_uncommitted_check_uses_the_v4_paths(monkeypatch):
    launcher = _launcher()
    seen = []

    def fake_run(command, **kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(launcher.stages.subprocess, "run", fake_run)
    assert launcher.uncommitted_inputs() == []
    assert "analysis/books-wikidata-v2/experiment_plan_v4.json" in seen[0]
    assert "analysis/books-controlled/matrix_stages.py" in seen[0]


def test_plan_smokes_both_user_types_before_the_matrix():
    plan = json.loads((ROOT / "analysis/books-wikidata-v2/experiment_plan_v4.json").read_text())
    stages = {stage["name"]: stage for stage in plan["stages"]}
    assert stages["smoke_llm_user"]["simulator_mode"] == "llm"
    assert len(stages["smoke_llm_user"]["simulator_labels"]) == 2
    assert plan["trial_accounting"]["smoke_trials"] == stages["smoke"]["planned_trials"] + stages["smoke_llm_user"]["planned_trials"]


def test_agent_filter_selects_only_those_agents_waves():
    launcher = _launcher()
    plan = json.loads((ROOT / "analysis/books-wikidata-v2/experiment_plan_v4.json").read_text())
    stages = [stage for stage in plan["stages"] if stage["name"] in ("reliability", "scale_all")]
    loaded = {stage["name"]: ({"cells": launcher.stages.build_cells(plan, stage)}, None) for stage in stages}
    open_agents = ["deepseek_v41_flash_high", "glm53_flash_high", "qwen38_27b_high"]
    waves = launcher.select_waves(stages, loaded, open_agents)
    assert {cell["agent_label"] for _, cells, _, _ in waves for cell in cells} == set(open_agents)
    assert len(launcher.select_waves(stages, loaded, None)) == 4 + 3
    try:
        launcher.select_waves(stages, loaded, ["no_such_agent"])
    except SystemExit as error:
        assert "no_such_agent" in str(error)
    else:
        raise AssertionError("an unknown agent label was accepted")


def test_stage_prompt_variant_reaches_the_command_and_the_validation(tmp_path):
    launcher = _launcher()
    plan = json.loads((ROOT / "analysis/books-wikidata-v2/experiment_plan_v4.json").read_text())
    reliability = next(row for row in plan["stages"] if row["name"] == "reliability")
    stage = {**reliability, "agent_prompt_variant": "ask-requirements-v1"}
    cell = launcher.stages.build_cells(plan, stage)[0]
    args = argparse.Namespace(runtime=plan["fixed_runtime"], artifact=ROOT / "data/tracks/books-wikidata-v2",
                              output=tmp_path)
    command = launcher.stages.run_command(args, stage, cell, concurrency=1, ceiling=1.0)
    assert command[-2:] == ["--agent-prompt-variant", "ask-requirements-v1"]
    assert "--agent-prompt-variant" not in launcher.stages.run_command(args, reliability, cell, concurrency=1, ceiling=1.0)
    run = tmp_path / "run"
    (run / "traces").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({
        "harness_profile": {"id": "books-v4"}, "harness_commit": {"sha": "a" * 40, "dirty": False}}))
    result = launcher.validate(run, cell, stage=stage, plan=plan, runtime=plan["fixed_runtime"])
    assert any(error.startswith("agent_prompt_variant") for error in result["errors"])


def test_uncommitted_check_includes_the_plan_being_launched(monkeypatch):
    launcher = _launcher()
    seen = []
    monkeypatch.setattr(launcher.stages, "uncommitted", lambda paths: seen.append(paths) or [])
    launcher.uncommitted_inputs(Path("analysis/books-wikidata-v2/experiment_plan_v4_addendum.json"))
    assert "analysis/books-wikidata-v2/experiment_plan_v4_addendum.json" in seen[-1]
    assert "src/tau_rec" in seen[-1]


def _validated(tmp_path, launcher, commit, monkeypatch, same):
    plan = json.loads((ROOT / "analysis/books-wikidata-v2/experiment_plan_v4.json").read_text())
    stage = next(row for row in plan["stages"] if row["name"] == "reliability")
    cell = launcher.stages.build_cells(plan, stage)[0]
    run = tmp_path / "run"
    (run / "traces").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"harness_profile": {"id": plan["harness"]["profile"]["id"]},
                                                       "harness_commit": commit}))
    monkeypatch.setattr(launcher, "harness_same", lambda freeze, sha, paths=launcher.FROZEN_PATHS: same)
    return launcher.validate(run, cell, stage=stage, plan=plan, runtime=plan["fixed_runtime"])


def test_validate_accepts_a_repo_wide_dirty_flag_when_the_commit_runs_the_frozen_harness(tmp_path, monkeypatch):
    # harness_commit marks any modified tracked file; the launch guards keep the harness clean.
    launcher = _launcher()
    result = _validated(tmp_path, launcher, {"sha": "b" * 40, "dirty": True}, monkeypatch, same=True)
    assert not any(error.startswith("harness_commit") for error in result["errors"])
    assert any("working tree" in warning for warning in result["warnings"])


def test_validate_rejects_a_commit_whose_harness_differs_from_the_freeze(tmp_path, monkeypatch):
    launcher = _launcher()
    result = _validated(tmp_path, launcher, {"sha": "b" * 40, "dirty": False}, monkeypatch, same=False)
    assert any(error.startswith("harness_commit") for error in result["errors"])
    result = _validated(tmp_path / "x", launcher, {"sha": None, "dirty": False}, monkeypatch, same=True)
    assert any(error.startswith("harness_commit") for error in result["errors"])
