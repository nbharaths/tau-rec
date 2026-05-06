from click.testing import CliRunner
from tau_rec.cli import main

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

def test_cli_report_help():
    runner = CliRunner()
    result = runner.invoke(main, ["report", "--help"])
    assert result.exit_code == 0
    assert "results" in result.output.lower()
