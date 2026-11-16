"""Inspect is an optional adapter: nothing on the native or books paths may load it."""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent

BOOKS_SCRIPTS = [
    "scripts/run_books_protocol.py",
    "scripts/report_books_run.py",
    "scripts/verify_books_archive.py",
    "scripts/validate_books_controlled.py",
]


def test_inspect_ai_importable_and_pinned():
    assert Version(importlib.metadata.version("inspect_ai")) >= Version("0.3.290")
    assert Version(importlib.metadata.version("openai")) >= Version("3.4")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT
    )


def test_tau_rec_import_does_not_load_inspect():
    code = (
        "import sys\n"
        "import tau_rec, tau_rec.cli, tau_rec.simulator.user_simulator, "
        "tau_rec.orchestrator.orchestrator, tau_rec.agents.litellm_agent\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'inspect_ai')\n"
        "assert not loaded, loaded\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("script", BOOKS_SCRIPTS)
def test_books_entry_point_does_not_load_inspect(script):
    path = ROOT / script
    assert path.exists(), path
    code = (
        "import runpy, sys\n"
        f"runpy.run_path({str(path)!r}, run_name='not_main')\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'inspect_ai')\n"
        "assert not loaded, loaded\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
