"""Post hoc capped hidden success (posthoc_v4.py) and the follow-up analysis (analyze_v4_followups.py)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis/books-controlled"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hidden_trace(rejections, success):
    decisions = [{"action": "candidate_verdict", "verdict": "rejected"}] * rejections
    if success:
        decisions = decisions + [{"action": "candidate_verdict", "verdict": "accepted"}]
    return {"simulator_decisions": decisions, "evaluation": {"primary_reward": 1.0 if success else 0.0}}


def test_capped_success():
    posthoc = _load("posthoc_v4")
    traces = [_hidden_trace(0, True), _hidden_trace(2, True), _hidden_trace(4, True), _hidden_trace(1, False)]
    assert posthoc.capped_success(traces, 3) == 0.5
    assert posthoc.capped_success(traces, 4) == 0.75
    assert posthoc.capped_success(traces, None) == 0.75
    assert posthoc.capped_success(traces, 0) == 0.25


def test_incomplete_cell_is_refused(tmp_path):
    followups = _load("analyze_v4_followups")
    traces = {"task_001_m": [{}] * 4, "task_002_m": [{}] * 3}
    with pytest.raises(SystemExit, match="task_002_m"):
        followups.require_complete("prompt_variant/glm", traces, trials_per_task=4,
                                   expected={"task_001_m", "task_002_m"})
    followups.require_complete("ok", {"task_001_m": [{}] * 4}, trials_per_task=4, expected={"task_001_m"})


def test_cell_missing_whole_tasks_or_all_traces_is_refused():
    followups = _load("analyze_v4_followups")
    with pytest.raises(SystemExit, match="no traces found"):
        followups.require_complete("reliability/glm", {}, trials_per_task=8, expected={"task_001_m"})
    with pytest.raises(SystemExit, match="task_002_m"):
        followups.require_complete("reliability/glm", {"task_001_m": [{}] * 8}, trials_per_task=8,
                                   expected={"task_001_m", "task_002_m"})


def test_uncapped_success_ignores_wall_capped_trials():
    followups = _load("analyze_v4_followups")
    traces = [{"stop_reason": "recommended", "evaluation": {"primary_reward": 1.0}},
              {"stop_reason": "recommended", "evaluation": {"primary_reward": 0.0}},
              {"stop_reason": "wall_cap", "evaluation": {"primary_reward": 0.0}}]
    assert followups.uncapped_success(traces) == 0.5
    assert followups.uncapped_success([traces[2]]) is None


def test_group_levels_report_each_condition_on_satisfiable_families_too():
    posthoc = _load("posthoc_v4")
    family_of = {"f1_v": "f1", "f1_m": "f1", "f1_h": "f1", "f2_v": "f2", "f2_m": "f2", "f2_h": "f2"}
    nvr = {"f1": False, "f2": True}
    members = {"f1_v": 1.0, "f1_m": 0.5, "f1_h": 0.25, "f2_v": 1.0, "f2_m": 1.0, "f2_h": 0.0}
    assert posthoc.group_levels(members, family_of, nvr) == {
        "families": 2, "nvr_families": 1, "pass^1/volunteer": 1.0, "pass^1/mixed": 0.75, "pass^1/hidden": 0.125,
        "pass^1/volunteer_satisfiable": 1.0, "pass^1/mixed_satisfiable": 0.5, "pass^1/hidden_satisfiable": 0.25,
    }


def test_success_curve_counts_each_trial_once_it_succeeds_within_k_rejections():
    posthoc = _load("posthoc_v4")
    by_task = {"a": [_hidden_trace(0, True), _hidden_trace(2, True)], "b": [_hidden_trace(1, False), _hidden_trace(5, True)]}
    curve = posthoc.success_curve(by_task, ["a", "b"], max_k=3)
    # Task means, then the mean over tasks: a = (1 + 0) / 2 at k = 0, b = 0 until k = 5.
    assert curve == {"k0": 0.25, "k1": 0.25, "k2": 0.5, "k3": 0.5, "uncapped": 0.75}


def test_reference_curve_reads_proposals_from_the_scripted_rows():
    posthoc = _load("posthoc_v4")
    rows = {"a": {"success": True, "probes": 1}, "b": {"success": True, "probes": 3}, "c": {"success": False, "probes": 18}}
    assert posthoc.reference_curve(rows, ["a", "b", "c"], max_k=2) == {"k0": 1 / 3, "k1": 1 / 3, "k2": 2 / 3, "uncapped": 2 / 3}


def test_first_trials_keeps_the_lowest_trial_indices():
    posthoc = _load("posthoc_v4")
    traces = [{"trial": 2}, {"trial": 0}, {"trial": 1}]
    assert [trace["trial"] for trace in posthoc.first_trials(traces, 2)] == [0, 1]


def test_separated_pairs_counts_pairs_whose_family_interval_excludes_zero():
    posthoc = _load("posthoc_v4")
    family_of = {f"t{i}": f"f{i // 2}" for i in range(40)}
    high = {task: 1.0 for task in family_of}
    low = {task: 0.0 for task in family_of}
    out = posthoc.separated_pairs({"high": high, "low": low, "low_too": dict(low)}, family_of, "test")
    assert (out["pairs"], out["separated"]) == (3, 2)
    assert out["by_pair"]["high|low"]["separated"] and not out["by_pair"]["low|low_too"]["separated"]
    assert out["by_pair"]["high|low"]["estimate"] == 1.0


def test_order_flips_lists_pairs_whose_order_differs_between_two_sets_of_levels():
    posthoc = _load("posthoc_v4")
    assert posthoc.order_flips({"a": 0.5, "b": 0.6, "c": 0.7}, {"a": 0.65, "b": 0.6, "c": 0.7}) == ["a|b"]
    assert posthoc.order_flips({"a": 0.5, "b": 0.5}, {"a": 0.4, "b": 0.6}) == []  # a tie is not an order


def test_uncapped_level_drops_wall_capped_trials_before_averaging_tasks():
    posthoc = _load("posthoc_v4")
    win = {"evaluation": {"primary_reward": 1.0}, "stop_reason": "recommended"}
    loss = {"evaluation": {"primary_reward": 0.0}, "stop_reason": "recommended"}
    capped = {"evaluation": {"primary_reward": 0.0}, "stop_reason": "wall_cap"}
    assert posthoc.uncapped_level({"t1": [win, capped], "t2": [loss, win], "t3": [capped]}) == 0.75


def _constraint(field, reveal, op="contains", value="x"):
    return {"constraint": {"field": field, "op": op, "value": value}, "reveal": reveal}


def test_hidden_fields_lists_the_fields_of_hidden_predicates_once():
    posthoc = _load("posthoc_v4")
    task = {"constraints": [_constraint("genres", "volunteer"), _constraint("genres", "hidden"),
                            _constraint("publication_year", "hidden", op=">="),
                            _constraint("publication_year", "hidden", op="<="), _constraint("languages", "on_ask")]}
    assert posthoc.hidden_fields(task) == ("genres", "publication_year")


def test_group_means_averages_each_groups_tasks_and_skips_empty_groups():
    posthoc = _load("posthoc_v4")
    values = {"a": 1.0, "b": 0.0, "c": 0.5}
    out = posthoc.group_means(values, {"g1": ["a", "b"], "g2": ["b", "c"], "empty": []})
    assert out == {"g1": {"tasks": 2, "pass^1": 0.5}, "g2": {"tasks": 2, "pass^1": 0.25}}


def test_popularity_percentile_is_the_mid_rank_share_of_the_pool_below_the_item():
    posthoc = _load("posthoc_v4")
    links = {"a": 1, "b": 5, "c": 5, "d": 9}
    assert posthoc.popularity_percentile("d", ["a", "b", "c", "d"], links) == 0.875
    assert posthoc.popularity_percentile("b", ["a", "b", "c", "d"], links) == 0.5
    assert posthoc.popularity_percentile("a", ["a", "b", "c", "d"], links) == 0.125


def test_cue_split_counts_nvr_families_among_those_without_a_satisfiable_only_cue():
    posthoc = _load("posthoc_v4")
    families = [{"family_id": f, "no_valid_recommendation": nvr, "task_ids": {"volunteer": f"{f}_v"}}
                for f, nvr in (("f1", False), ("f2", False), ("f3", False), ("f4", True))]
    answers = {"f1_v": {"constraint_solutions": ["a", "b"], "read_ids": ["a"]},  # history holds a valid answer
               "f2_v": {"constraint_solutions": ["c"], "read_ids": []},
               "f3_v": {"constraint_solutions": ["d"], "read_ids": ["z"]},
               "f4_v": {"constraint_solutions": [], "read_ids": ["y"]}}
    tasks = {"f1_v": {"user_formats": []}, "f2_v": {"user_formats": ["ebook"]},  # one accepted format
             "f3_v": {"user_formats": ["ebook", "audiobook"]}, "f4_v": {"user_formats": []}}
    assert posthoc.cue_split(families, answers, tasks) == {"cued": {"families": 2, "nvr": 0},
                                                           "uncued": {"families": 2, "nvr": 1}}


def test_trial_cost_sums_agent_calls_preferring_the_provider_reported_cost():
    posthoc = _load("posthoc_v4")
    trace = {"api_calls": {"agent": [{"provider_reported_cost_usd": 0.25, "estimated_cost_usd": 9.0},
                                     {"provider_reported_cost_usd": None, "estimated_cost_usd": 0.5},
                                     {"estimated_cost_usd": None}],
                           "simulator": [{"provider_reported_cost_usd": 4.0}]}}
    assert posthoc.trial_cost(trace) == 0.75
    assert posthoc.trial_cost({}) == 0.0


def test_all_on_ask_disclosed_reads_every_disclosure_in_the_trial():
    posthoc = _load("posthoc_v4")
    task = {"constraints": [_constraint("genres", "volunteer"), _constraint("languages", "on_ask"),
                            _constraint("publication_year", "on_ask", op="<=")]}
    partial = {"simulator_decisions": [{"disclosed_constraint_indices": [0]}, {"disclosed_constraint_indices": [1]}]}
    full = {"simulator_decisions": [*partial["simulator_decisions"], {"disclosed_constraint_indices": [2]},
                                    {"action": "candidate_verdict"}]}
    assert not posthoc.all_on_ask_disclosed(partial, task)
    assert posthoc.all_on_ask_disclosed(full, task)
