"""Redacted traces score exactly like the originals, on real archived runs."""
import json
import os
from pathlib import Path

import pytest

from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.conversation import ConversationTrace
from tau_rec.data_model.task import Task
from tau_rec.evaluator.evaluator import CombinedEvaluator
from tau_rec.redact import REDACTED_TOOLS, redact_dir, redact_trace

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = Path(os.environ.get("TAU_REC_G1_TRACES", "/nonexistent"))


def test_redaction_removes_catalog_results_and_nothing_else():
    doc = json.loads((ROOT / "tests" / "fixtures" / "sample_trace.json").read_text())
    red = redact_trace(doc)
    for a, b in zip(doc["events"], red["events"]):
        if a.get("type") == "tool_call" and a["name"] in REDACTED_TOOLS:
            payload = json.loads(b["result"])
            assert payload["redacted"] is True and len(payload["sha256"]) == 64
            assert b["arguments"] == a["arguments"] and b["name"] == a["name"]
        else:
            assert a == b
    assert red["recommendations"] == doc["recommendations"] and red["stop_reason"] == doc["stop_reason"]
    assert any(e.get("type") == "tool_call" and e["name"] in REDACTED_TOOLS for e in doc["events"]), "fixture exercises redaction"


def test_metadata_redaction_preserves_only_success_state():
    doc = {
        "events": [
            {"type": "tool_call", "name": "get_metadata", "arguments": {"item_id": "wd_1"},
             "result": '{"id":"wd_1","title":"secret"}'},
            {"type": "tool_call", "name": "get_metadata", "arguments": {"item_id": "missing"},
             "result": '{"error":"not found"}'},
        ]
    }
    redacted = redact_trace(doc)
    success, failure = (json.loads(event["result"]) for event in redacted["events"])
    assert success["ok"] is True and failure["ok"] is False
    assert "secret" not in json.dumps(redacted)


def _score(evaluator, tasks, doc):
    trace = ConversationTrace.model_validate(doc)
    r = evaluator.evaluate(tasks[trace.task_id], trace)
    return (r.constraint_score, r.policy_score, tuple(r.policy_detail.violations), r.efficiency.total_tool_calls,
            r.efficiency.redundant_tool_call_rate)


@pytest.mark.skipif(not ARCHIVE.exists(), reason="published archive not extracted locally")
def test_redacted_archived_traces_rederive_identically(tmp_path):
    catalog = Catalog.from_json(str(ROOT / "data" / "catalog.json"))
    tasks = {p.stem: Task.from_json(str(p)) for p in (ROOT / "data" / "tasks").glob("*.json")}
    evaluator = CombinedEvaluator(catalog)
    n = redact_dir(ARCHIVE, tmp_path / "redacted")
    assert n == 240
    checked = 0
    for path in sorted(ARCHIVE.glob("*.json")):
        original = json.loads(path.read_text())
        redacted = json.loads((tmp_path / "redacted" / path.name).read_text())
        assert _score(evaluator, tasks, original) == _score(evaluator, tasks, redacted), path.name
        checked += 1
    assert checked == 240
