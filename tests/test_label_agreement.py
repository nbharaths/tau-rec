"""Agreement between two blind labellers of the recogniser evaluation set (score_label_agreement.py)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "analysis/books-controlled/score_label_agreement.py"


def _load():
    spec = importlib.util.spec_from_file_location("score_label_agreement", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identical_labels_agree_perfectly():
    agreement = _load().agreement
    a = [{"uid": 0, "fields": ["genres"], "generic": False}, {"uid": 1, "fields": [], "generic": True}]
    result = agreement(a, a)
    assert result["field_kappa"] == 1.0 and result["field_agreement"] == 1.0
    assert result["exact_field_set"] == 1.0 and result["generic_kappa"] == 1.0


def test_kappa_corrects_for_chance():
    agreement = _load().agreement
    a = [{"uid": i, "fields": ["genres"] if i < 2 else [], "generic": False} for i in range(4)]
    b = [{"uid": i, "fields": ["genres"] if i in (0, 2) else [], "generic": False} for i in range(4)]
    result = agreement(a, b)
    # genres: 2 of 4 items agree, chance agreement 0.5 -> kappa 0; other fields all agree (no positives).
    assert result["per_field_kappa"]["genres"] == 0.0
    assert result["exact_field_set"] == 0.5


def test_labels_are_matched_by_uid_not_order():
    agreement = _load().agreement
    a = [{"uid": 0, "fields": ["genres"], "generic": False}, {"uid": 1, "fields": [], "generic": False}]
    b = list(reversed(a))
    assert agreement(a, b)["exact_field_set"] == 1.0
