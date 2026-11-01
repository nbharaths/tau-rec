"""Per-agent levels under the base and ask-requirements prompts (prompt_headroom.py)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "analysis/books-controlled/prompt_headroom.py"


def _load():
    spec = importlib.util.spec_from_file_location("prompt_headroom", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(success: bool, rejections: int = 0) -> dict:
    return {"evaluation": {"primary_reward": 1.0 if success else 0.0},
            "simulator_decisions": [{"action": "candidate_verdict", "verdict": "rejected"}] * rejections}


def test_summary_splits_hidden_and_caps_rejections():
    summary = _load().cell_summary
    by_task = {
        "task_001_v": [_trace(True), _trace(True)],
        "task_001_m": [_trace(True), _trace(False)],
        "task_001_h": [_trace(True, 1), _trace(True, 5)],   # satisfiable: uncapped 1.0, within 3 rejections 0.5
        "task_002_h": [_trace(False), _trace(True)],        # NVR family
    }
    nvr = {"task_001_h": False, "task_002_h": True, "task_001_v": False, "task_001_m": False}
    row = summary(by_task, nvr)
    assert row["volunteer"] == 1.0 and row["mixed"] == 0.5
    assert row["hidden_satisfiable"] == 1.0 and row["hidden_satisfiable_within_3"] == 0.5
    assert row["hidden_nvr"] == 0.5


def _failed(final: str | None, verdicts: dict[str, str]) -> dict:
    events = [{"type": "tool_call", "name": "recommend", "arguments": {"item_id": final}}] if final else []
    return {"evaluation": {"primary_reward": 0.0}, "events": events,
            "simulator_decisions": [{"action": "candidate_verdict", "candidate_id": c, "verdict": v}
                                    for c, v in verdicts.items()]}


def test_summary_reports_the_untested_share_of_hidden_failures():
    by_task = {"task_001_h": [_trace(True), _failed("b1", {}), _failed("b2", {"b2": "rejected"}),
                              _failed(None, {}), _failed("b3", {"b9": "rejected"})]}
    row = _load().cell_summary(by_task, {"task_001_h": False})
    assert row["hidden_untested_of_failures"] == 0.5    # b1 and b3 never proposed; of 4 failures


def test_capped_difference_is_the_mean_of_family_differences():
    module = _load()
    verify = {"task_001_h": [_trace(True), _trace(True)], "task_002_h": [_trace(True), _trace(False)]}
    ask = {"task_001_h": [_trace(False), _trace(False)], "task_002_h": [_trace(True), _trace(False)]}
    row = module.capped_difference("t", verify, ask, {"task_001_h": False, "task_002_h": False},
                                   {"task_001_h": "f1", "task_002_h": "f2"}, {"f1": "a", "f2": "b"})
    assert row["estimate"] == 0.5 and row["families"] == 2 and len(row["family_ci95"]) == 2


def test_main_refuses_a_cell_without_traces(tmp_path, monkeypatch):
    import json
    import pytest
    module = _load()
    (tmp_path / "family_manifest.json").write_text(json.dumps({"families": []}))
    monkeypatch.setattr(module.addendum, "_cell", lambda root, stage, agent: {})
    monkeypatch.setattr("sys.argv", ["prompt_headroom.py", "--artifact", str(tmp_path), "--main-root", "m",
                                     "--addendum-root", "a", "--prompt-root", "p", "--output", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="no traces"):
        module.main()
    assert not (tmp_path / "o.json").exists()


def test_capped_difference_without_a_cap_uses_uncapped_hidden_success():
    module = _load()
    protocol = {"task_001_h": [_trace(True, 5), _trace(True, 5)]}
    verify = {"task_001_h": [_trace(False), _trace(False)]}
    args = ({"task_001_h": False}, {"task_001_h": "f1"}, {"f1": "a"})
    assert module.capped_difference("t", protocol, verify, *args, k=None)["estimate"] == 1.0
    assert module.capped_difference("t", protocol, verify, *args)["estimate"] == 0.0  # beyond three rejections


def test_main_adds_protocol_cells_and_contrasts_when_given_a_protocol_root(tmp_path, monkeypatch):
    import json
    module = _load()
    families = [{"family_id": f"f{i}", "no_valid_recommendation": False,
                 "constraint_signature": f"genres:contains:G{i}|countries:contains:US",
                 "task_ids": {"volunteer": f"task_00{i}_v", "mixed": f"task_00{i}_m", "hidden": f"task_00{i}_h"}}
                for i in (1, 2)]
    (tmp_path / "family_manifest.json").write_text(json.dumps({"families": families}))

    def cell(root, stage, agent):
        win = str(root) == "x"  # the protocol prompt wins every hidden trial, the others end without a book
        return {f"task_00{i}_{c}": [_trace(True) if c != "h" or win else _failed(None, {})] * 2
                for i in (1, 2) for c in "vmh"}

    monkeypatch.setattr(module.addendum, "_cell", cell)
    out_path = tmp_path / "o.json"
    monkeypatch.setattr("sys.argv", ["prompt_headroom.py", "--artifact", str(tmp_path), "--main-root", "m",
                                     "--addendum-root", "a", "--prompt-root", "p", "--verify-root", "v",
                                     "--protocol-root", "x", "--output", str(out_path)])
    assert module.main() == 0
    out = json.loads(out_path.read_text())
    for agent in module.AGENTS:
        assert out["agents"][agent]["ask_verify_protocol"]["hidden_satisfiable"] == 1.0
        assert out["agents"][agent]["ask_and_verify"]["hidden_satisfiable"] == 0.0
        assert out["contrasts"][f"protocol_minus_verify/{agent}/hidden_satisfiable"]["estimate"] == 1.0
    assert out["contrasts"]["protocol_minus_verify/model_averaged/hidden_satisfiable"]["estimate"] == 1.0
    # Every prompt cell gets a family interval on hidden satisfiable success.
    for agent in module.AGENTS:
        for prompt, level in (("base", 0.0), ("ask_requirements", 0.0), ("ask_and_verify", 0.0),
                              ("ask_verify_protocol", 1.0)):
            row = out["intervals"][f"{prompt}/{agent}/hidden_satisfiable"]
            assert row["estimate"] == level and len(row["family_ci95"]) == 2


def test_reference_difference_compares_an_agent_cell_with_a_deterministic_reference_by_family():
    module = _load()
    cell = {"task_001_h": [_trace(True), _trace(False)], "task_002_h": [_trace(True), _trace(True)],
            "task_003_h": [_trace(False), _trace(False)]}  # task_003 is NVR and left out
    rows = {"task_001_h": {"success": True}, "task_002_h": {"success": False}, "task_003_h": {"success": True}}
    nvr = {"task_001_h": False, "task_002_h": False, "task_003_h": True}
    row = module.reference_difference("t", cell, rows, nvr, {"task_001_h": "f1", "task_002_h": "f2", "task_003_h": "f3"},
                                      {"f1": "a", "f2": "b", "f3": "c"})
    # f1: 0.5 - 1 = -0.5; f2: 1 - 0 = +1; the mean over the two satisfiable families is 0.25.
    assert row["estimate"] == 0.25 and row["families"] == 2


def test_summary_reports_pass1_over_all_tasks():
    by_task = {"task_001_v": [_trace(True), _trace(True)], "task_001_m": [_trace(True), _trace(False)],
               "task_001_h": [_failed(None, {}), _failed(None, {})], "task_002_h": [_trace(True), _trace(True)]}
    nvr = {"task_001_h": False, "task_002_h": True}
    assert _load().cell_summary(by_task, nvr)["all"] == 0.625  # (1 + 0.5 + 0 + 1) / 4


def test_kendall_tau_over_agent_levels():
    tau = _load().kendall_tau
    assert tau({"a": 3, "b": 2, "c": 1}, {"a": 30, "b": 20, "c": 10}) == 1.0
    assert tau({"a": 3, "b": 2, "c": 1}, {"a": 1, "b": 2, "c": 3}) == -1.0
    assert tau({"a": 3, "b": 2, "c": 1, "d": 0}, {"a": 3, "b": 1, "c": 2, "d": 0}) == 4 / 6


def test_level_interval_is_the_family_mean_of_uncapped_hidden_satisfiable_success():
    module = _load()
    cell = {"task_001_h": [_trace(True), _trace(False)], "task_002_h": [_trace(True, 5), _trace(True, 5)],
            "task_003_h": [_trace(False), _trace(False)]}  # task_003 is NVR and left out
    nvr = {"task_001_h": False, "task_002_h": False, "task_003_h": True}
    row = module.level_interval("t", cell, nvr, {"task_001_h": "f1", "task_002_h": "f2", "task_003_h": "f3"},
                                {"f1": "a", "f2": "b", "f3": "c"})
    # f1: 0.5; f2: 1.0 (uncapped, so five rejections still count); the mean over the two families is 0.75.
    assert row["estimate"] == 0.75 and row["families"] == 2
    lo, hi = row["family_ci95"]
    assert 0.5 <= lo <= 0.75 <= hi <= 1.0
