"""Follow-up plans after the v4 matrix: the prompt variant for GLM and Qwen, then ask-and-verify and protocol."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "analysis/books-wikidata-v2"
MAKER = ROOT / "analysis/books-controlled/make_v4_followup_plans.py"
CONFIG = A / "model_config_v4_followup.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make(tmp_path, *extra, env=None):
    prompt = tmp_path / "prompt.json"
    result = subprocess.run(
        [sys.executable, str(MAKER), "--prompt-output", str(prompt),
         "--verify-output", str(tmp_path / "verify.json"), "--protocol-output", str(tmp_path / "protocol.json"),
         *extra],
        capture_output=True, text=True, cwd=ROOT, env=env,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(prompt.read_text())


def test_followup_model_config_keeps_the_v4_agents():
    v4 = json.loads((A / "model_config_v4.json").read_text())
    followup = json.loads(CONFIG.read_text())
    assert [row["label"] for row in followup["agent_models"]] == [row["label"] for row in v4["agent_models"]]
    assert "standard" in followup["routing_note"]


def test_followup_plans_match_the_design(tmp_path):
    prompt = _make(tmp_path)
    base = json.loads((A / "experiment_plan_v4.json").read_text())
    reliability = next(s for s in base["stages"] if s["name"] == "reliability")["task_ids"]
    assert len(reliability) == 54

    assert [s["name"] for s in prompt["stages"]] == ["smoke_prompt_variant", "prompt_variant"]
    assert [s["planned_trials"] for s in prompt["stages"]] == [6, 864]
    assert all(s["agent_labels"] == ["glm53_flash_high", "qwen38_27b_high"] for s in prompt["stages"])
    assert prompt["budgets_usd"] == {"v4_hard_ceiling": 20.0}

    for plan in (prompt,):
        assert plan["status"] == "draft"
        assert plan["base_plan_sha256"] == _sha(A / "experiment_plan_v4.json")
        assert plan["artifact"] == {**base["artifact"], "model_config_sha256": _sha(CONFIG)}
        assert plan["fixed_runtime"] == base["fixed_runtime"]
        assert plan["stall_retry_rule"] == base["stall_retry_rule"]
        for stage in plan["stages"]:
            assert stage["retrieval_modes"] == ["structured"]
            assert stage["catalog_sizes"] == [20000]
            assert stage["simulator_mode"] == "symbolic"
            expected = reliability[:3] if stage["name"].startswith("smoke") else reliability
            assert stage["task_ids"] == expected
            assert stage["trials_per_task"] == (1 if stage["name"].startswith("smoke") else 8)
    assert all(s["agent_prompt_variant"] == "ask-requirements-v1" for s in prompt["stages"])
    assert prompt["api_parameters"] == base["api_parameters"]


def test_verify_plan_runs_the_ask_and_verify_prompt_for_all_four_agents(tmp_path):
    _make(tmp_path)
    verify = json.loads((tmp_path / "verify.json").read_text())
    agents = ["deepseek_v41_flash_high", "glm53_flash_high", "qwen38_27b_high", "gpt61_sol_high"]
    assert [s["name"] for s in verify["stages"]] == ["smoke_verify_variant", "verify_variant"]
    assert [s["planned_trials"] for s in verify["stages"]] == [12, 1728]
    assert all(s["agent_labels"] == agents for s in verify["stages"])
    assert all(s["agent_prompt_variant"] == "ask-and-verify-v1" for s in verify["stages"])
    assert [s["trials_per_task"] for s in verify["stages"]] == [1, 8]
    assert verify["budgets_usd"] == {"v4_hard_ceiling": 25.0}
    assert verify["version"] == "books-wikidata-v4-verify-v1"
    assert "ask-requirements" in verify["analysis"]["verify_variant"]


def test_protocol_plan_runs_the_protocol_prompt_for_all_four_agents_like_verify(tmp_path):
    _make(tmp_path)
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    verify = json.loads((tmp_path / "verify.json").read_text())
    agents = ["deepseek_v41_flash_high", "glm53_flash_high", "qwen38_27b_high", "gpt61_sol_high"]
    assert [s["name"] for s in protocol["stages"]] == ["smoke_protocol_variant", "protocol_variant"]
    assert [s["planned_trials"] for s in protocol["stages"]] == [12, 1728]
    assert all(s["agent_labels"] == agents for s in protocol["stages"])
    assert all(s["agent_prompt_variant"] == "ask-verify-protocol-v1" for s in protocol["stages"])
    # The same tasks and trials as the verify follow-up, so the two prompts differ only in the protocol sentences.
    assert [s["task_ids"] for s in protocol["stages"]] == [s["task_ids"] for s in verify["stages"]]
    assert [s["trials_per_task"] for s in protocol["stages"]] == [1, 8]
    assert protocol["budgets_usd"] == {"v4_hard_ceiling": 25.0}
    assert protocol["version"] == "books-wikidata-v4-protocol-v1"
    assert "ask-and-verify" in protocol["analysis"]["protocol_variant"]


def test_only_writes_the_selected_plans(tmp_path):
    result = subprocess.run(
        [sys.executable, str(MAKER), "--prompt-output", str(tmp_path / "p.json"), "--verify-output", str(tmp_path / "v.json"),
         "--protocol-output", str(tmp_path / "x.json"), "--only", "verify"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in tmp_path.iterdir()) == ["v.json"]


def test_followup_plans_are_launchable_by_the_v4_launcher(tmp_path):
    prompt = _make(tmp_path)
    launcher = _load(ROOT / "analysis/books-controlled/run_v4_matrix.py", "v4_launcher_followup")
    cells = launcher.stages.build_cells(prompt, prompt["stages"][1])
    assert [c["model"] for c in cells] == ["openrouter/z-ai/glm-5.3-flash", "openrouter/qwen/qwen3.8-27b"]


def test_freeze_refuses_dirty_inputs(monkeypatch, tmp_path):
    maker = _load(MAKER, "followup_maker")
    monkeypatch.setattr(maker, "_dirty_inputs", lambda: [" M src/tau_rec/agents/litellm_agent.py"])
    with pytest.raises(SystemExit, match="uncommitted"):
        maker.main(["--prompt-output", str(tmp_path / "p.json"), "--freeze"])


def test_draft_plans_build_without_a_git_checkout(tmp_path):
    # A release tarball has no .git; only --freeze needs the commit.
    import os
    env = {**os.environ, "GIT_DIR": str(tmp_path / "no-git")}
    prompt = _make(tmp_path, env=env)
    assert prompt["status"] == "draft"
