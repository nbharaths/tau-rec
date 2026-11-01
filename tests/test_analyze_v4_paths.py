"""analyze_v4.py reports missing traces instead of failing later with ZeroDivisionError."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _analyze():
    spec = importlib.util.spec_from_file_location("analyze_v4_paths", ROOT / "analysis/books-controlled/analyze_v4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_v4_without_traces_says_so(tmp_path):
    with pytest.raises(SystemExit, match="no traces found"):
        _analyze()._load_cells(tmp_path, "reliability")


def test_analyze_v4_with_unreachable_run_dirs_says_so(tmp_path):
    state = {"cells": [{"status": "complete", "key": "k", "run_dir": "out/books-wikidata-v4/reliability/x"}]}
    (tmp_path / "reliability_v4_state.json").write_text(json.dumps(state))
    with pytest.raises(SystemExit, match="no traces found"):
        _analyze()._load_cells(tmp_path, "reliability")
