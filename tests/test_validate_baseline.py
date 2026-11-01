"""`validate --baseline`: the committed task baselines match the tasks on disk."""
from pathlib import Path

from click.testing import CliRunner

from tau_rec.cli import main as cli

ROOT = Path(__file__).resolve().parent.parent


def test_full_validate_matches_committed_baselines():
    for track, baseline in (("data", "data/task_baseline.json"),
                            ("data/tracks/books-wikidata-v2", "data/tracks/books-wikidata-v2/task_baseline.json")):
        result = CliRunner().invoke(cli, ["validate", "--track", str(ROOT / track), "--baseline", str(ROOT / baseline)])
        assert result.exit_code == 0, result.output
        assert "Baseline OK" in result.output
