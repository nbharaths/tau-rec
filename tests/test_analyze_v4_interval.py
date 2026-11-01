"""analyze_v4.interval: family and template intervals, and the exact template sign-flip test."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "analysis/books-controlled/analyze_v4.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_v4", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interval_adds_an_exact_sign_flip_p_over_template_sums():
    # Template sums are +3 (A), +1 (B) and -1 (C), observed |3 + 1 - 1| = 3; six of the eight sign patterns
    # reach 3 in absolute value.
    values = {"f1": 2.0, "f2": 1.0, "f3": 1.0, "f4": -1.0}
    clusters = {"f1": "A", "f2": "A", "f3": "B", "f4": "C"}
    row = _load().interval("test", values, clusters)
    assert row["template_sign_flip_p"] == 0.75
    assert {"family_ci95", "family_p", "template_ci95", "template_p"} <= set(row)
