from click.testing import CliRunner
from types import SimpleNamespace

from tau_rec.cli import _stratified_dry_run_sample, main

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "tau-rec" in result.output.lower() or "benchmark" in result.output.lower()

def test_cli_validate_help():
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--help"])
    assert result.exit_code == 0
    assert "catalog" in result.output.lower()

def test_cli_run_help():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "model" in result.output.lower()
    assert "policy" in result.output.lower()
    assert "--dry-run-output" in result.output

def test_cli_report_help():
    runner = CliRunner()
    result = runner.invoke(main, ["report", "--help"])
    assert result.exit_code == 0
    assert "results" in result.output.lower()


def test_dry_run_sample_round_robins_complexity_deterministically():
    tasks = [
        SimpleNamespace(id=f"{complexity}_{index}", complexity=complexity)
        for complexity in ("simple", "medium", "complex")
        for index in range(5)
    ]
    first = _stratified_dry_run_sample(tasks, 3)
    second = _stratified_dry_run_sample(tasks, 3)
    assert [task.id for task in first] == [task.id for task in second]
    assert {task.complexity for task in first} == {"simple", "medium", "complex"}

    larger = _stratified_dry_run_sample(tasks, 8)
    counts = {
        complexity: sum(task.complexity == complexity for task in larger)
        for complexity in ("simple", "medium", "complex")
    }
    assert max(counts.values()) - min(counts.values()) <= 1
