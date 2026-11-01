"""The reportable books protocol runner (scripts/run_books_protocol.py)."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts/run_books_protocol.py"
PANELS = json.loads((ROOT / "analysis/books-wikidata-v2/panels.json").read_text())["panels"]


def _runner():
    spec = importlib.util.spec_from_file_location("books_protocol_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_primary_block_runs_the_paper_protocol():
    runner = _runner()
    command = runner.build_command(runner.parse_args(["--model", "m", "--retrieval-mode", "structured"]))
    assert command[:4] == ["uv", "run", "tau-rec", "run"]
    assert _value(command, "--track") == "data/tracks/books-wikidata-v2"
    assert _value(command, "--simulator-mode") == "symbolic"
    assert _value(command, "--retrieval-mode") == "structured"
    assert _value(command, "--catalog-size") == "20000"
    assert _value(command, "--max-turns") == "20"
    assert _value(command, "--trial-timeout-seconds") == "600"
    assert _value(command, "--trials") == "4"
    assert _value(command, "--tasks-filter").split(",") == PANELS["primary"]["task_ids"]


def test_reliability_block_uses_its_panel_and_eight_trials():
    runner = _runner()
    args = runner.parse_args(["--model", "m", "--retrieval-mode", "keyword", "--block", "reliability"])
    command = runner.build_command(args)
    assert _value(command, "--tasks-filter").split(",") == PANELS["reliability"]["task_ids"]
    assert _value(command, "--trials") == "8"


def test_optional_agent_settings_pass_through():
    runner = _runner()
    command = runner.build_command(runner.parse_args([
        "--model", "m", "--retrieval-mode", "keyword", "--reasoning-effort", "high",
        "--provider-order", "deepinfra/fp8", "--max-provider-cost-usd", "5",
    ]))
    assert _value(command, "--reasoning-effort") == "high"
    assert _value(command, "--provider-order") == "deepinfra/fp8"
    assert _value(command, "--max-provider-cost-usd") == "5.0"


def test_ask_requirements_prompt_passes_through():
    runner = _runner()
    command = runner.build_command(runner.parse_args([
        "--model", "m", "--retrieval-mode", "structured", "--agent-prompt-variant", "ask-requirements-v1",
    ]))
    assert _value(command, "--agent-prompt-variant") == "ask-requirements-v1"
    default = runner.build_command(runner.parse_args(["--model", "m", "--retrieval-mode", "structured"]))
    assert "--agent-prompt-variant" not in default


def test_without_execute_it_only_prints(monkeypatch, capsys):
    runner = _runner()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not launch without --execute"))
    assert runner.main(["--model", "m", "--retrieval-mode", "keyword"]) == 0
    assert "tau-rec run" in capsys.readouterr().out


def test_execute_needs_a_cost_ceiling(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(SystemExit):
        runner.main(["--model", "m", "--retrieval-mode", "keyword", "--execute"])


def test_execute_refuses_inside_a_test(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("a test reached a paid launch"))
    with pytest.raises(SystemExit, match="refusing"):
        runner.main(["--model", "m", "--retrieval-mode", "keyword", "--execute", "--max-provider-cost-usd", "1"])
