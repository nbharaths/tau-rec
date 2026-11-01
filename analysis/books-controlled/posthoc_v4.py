#!/usr/bin/env python3
"""Post-hoc analyses of the v4 run; none is in the frozen plan.

- The 20k interface contrast by complexity tier, model-averaged and per agent.
- Randomization tests of the interface contrast: family-level sign flips
  (Monte Carlo, 100,000 draws) and an exact template-level test over every sign
  pattern of the field-operator templates on the panel.
- The minimum detectable effect of the primary contrast (two-sided alpha 0.05,
  80% power) from its family bootstrap.
- The primary contrast with the stalled first attempts in place of their re-runs.
- pass^1 by condition for families whose predicates name a language other than
  English or a country other than the US and UK, and for the other families,
  also over satisfiable families alone (the groups hold different NVR shares).
- Proposals rejected per successful hidden trial, and hidden success when a trial
  counts only after at most k rejected proposals (k = 1..5; k = 3 is reported),
  with the whole curve (k = 0..17) for each agent and the scripted references
  (the eliminating one with broad hypotheses, and with the generator's template shapes).
- pass^1 by condition on satisfiable families, overall and per agent (pooled over
  interfaces, with family-bootstrap intervals).
- Discriminative power: the agent-interface pairs whose paired family-bootstrap
  95% interval for the pass^1 difference excludes zero, on all tasks and on hidden
  satisfiable tasks, with all trials and with only the first one or two.
- Hidden success on families whose reading list can stand in for asking (a
  history and two hidden predicates) and on the others.
- Whether dropping trials that end at the wall-clock cap reorders the
  agent-interface configurations.
- pass^1 on satisfiable families by tier and condition, and on hidden
  satisfiable tasks by hidden field (a genre also by whether another genre is
  revealed).
- How popular successful recommendations are within their answer sets (the
  mid-rank sitelink percentile among unread answers, sets of five or more).
- NVR families among those whose history holds a valid answer or whose user
  accepts one format (cues that occur only in satisfiable families) and the rest.
- The agent API cost and runtime per trial, and for one agent's reportable run
  (the panel's 153 tasks, four trials each).
- Mixed success on satisfiable tasks split by whether the user disclosed every
  on_ask predicate during the trial.

Reuses analyze_v4's loaders and paired family bootstrap (B = 10,000, seed
20261014). No LLM is called.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

_spec = importlib.util.spec_from_file_location("analyze_v4", Path(__file__).with_name("analyze_v4.py"))
v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4)

TIERS = ("simple", "medium", "complex")
FLIPS = 100_000
ANGLO = {"languages": {"English"}, "countries": {"United States", "United Kingdom"}}
PRIMARY = "primary/interface/model_averaged/pass^1"
MAX_K = 17  # 18 proposals fit in the 20 turns: rejections before the last one


def _cells(root: Path, stage: str, *, quarantined: bool = False) -> dict[tuple[str, str], dict[str, list[dict]]]:
    state = json.loads((root / f"{stage}_v4_state.json").read_text())
    if any(cell["status"] != "complete" for cell in state["cells"]):
        raise SystemExit(f"{stage}: not every cell is complete")
    out = {}
    for cell in state["cells"]:
        if cell["catalog_size"] != 20000 or cell.get("simulator_label"):
            continue
        run_dir = Path(cell["run_dir"])
        by_task = v4._trials(run_dir)
        if quarantined:
            first = {}
            for path in sorted((run_dir / "quarantine").glob("*.json")):
                trace = json.loads(path.read_text())
                first[(trace["task_id"], trace["trial"])] = trace
            by_task = {task: [first.get((task, trace["trial"]), trace) for trace in traces]
                       for task, traces in by_task.items()}
        out[(cell["agent_label"], cell["retrieval_mode"])] = by_task
    return out


def _primary(root: Path, *, quarantined: bool = False) -> dict[tuple[str, str], dict[str, list[dict]]]:
    remainder = _cells(root, "primary_remainder", quarantined=quarantined)
    reliability = _cells(root, "reliability", quarantined=quarantined)
    return {key: {**remainder[key], **reliability[key]} for key in remainder}


def _effects(primary: dict, agents: list[str]) -> dict[str, dict[str, float]]:
    per_agent = {}
    for agent in agents:
        keyword, structured = primary[(agent, "keyword")], primary[(agent, "structured")]
        per_agent[agent] = {task: v4._pass(structured[task], 1) - v4._pass(keyword[task], 1)
                            for task in keyword if task in structured}
    shared = set.intersection(*(set(values) for values in per_agent.values()))
    per_agent["model_averaged"] = {task: sum(per_agent[a][task] for a in agents) / len(agents) for task in shared}
    return per_agent


def _sign_flip_family(name: str, family_values: dict[str, float]) -> float:
    rng = random.Random(f"{v4.SEED}:{name}:sign-flip")
    values = [family_values[family] for family in sorted(family_values)]
    observed = abs(sum(values))
    hits = sum(abs(sum(v if rng.random() < 0.5 else -v for v in values)) >= observed - 1e-12 for _ in range(FLIPS))
    return (hits + 1) / (FLIPS + 1)


def _sign_flip_template(family_values: dict[str, float], template_of: dict[str, str]) -> tuple[float, int]:
    sums: dict[str, float] = defaultdict(float)
    for family, value in family_values.items():
        sums[template_of[family]] += value
    values = [sums[template] for template in sorted(sums)]
    observed = abs(sum(values))
    hits = sum(
        abs(sum(v if mask >> i & 1 else -v for i, v in enumerate(values))) >= observed - 1e-12
        for mask in range(2 ** len(values))
    )
    return hits / 2 ** len(values), len(values)


def group_levels(members: dict[str, float], family_of: dict[str, str], nvr: dict[str, bool]) -> dict:
    """Family-averaged pass^1 per condition for one group of tasks, over all its families and its satisfiable ones."""
    def level(letter: str, satisfiable_only: bool = False) -> float | None:
        chosen = {t: v for t, v in members.items()
                  if t.endswith(f"_{letter}") and not (satisfiable_only and nvr[family_of[t]])}
        return v4._mean(v4._family_mean(chosen, family_of).values())
    return {
        "families": len({family_of[t] for t in members}),
        "nvr_families": len({family_of[t] for t in members if nvr[family_of[t]]}),
        **{f"pass^1/{reveal}": level(letter) for letter, reveal in v4.REVEALS.items()},
        **{f"pass^1/{reveal}_satisfiable": level(letter, satisfiable_only=True) for letter, reveal in v4.REVEALS.items()},
    }


def _rejections(trace: dict) -> int:
    return sum(d.get("action") == "candidate_verdict" and d.get("verdict") == "rejected"
               for d in trace.get("simulator_decisions", []))


def capped_success(traces: list[dict], k: int | None) -> float:
    """Share of traces that succeed after at most ``k`` rejected proposals (``None``: uncapped)."""
    return v4._mean(trace["evaluation"]["primary_reward"] >= 1 - 1e-9 and (k is None or _rejections(trace) <= k)
                    for trace in traces)


def _caps(max_k: int) -> dict[str, int | None]:
    return {f"k{k}": k for k in range(max_k + 1)} | {"uncapped": None}


def success_curve(by_task: dict, tasks: list, max_k: int = MAX_K) -> dict[str, float]:
    """Mean over tasks of the share of trials succeeding within k rejected proposals (k = 0..max_k, uncapped)."""
    return {name: v4._mean(capped_success(by_task[task], k) for task in tasks) for name, k in _caps(max_k).items()}


def reference_curve(rows: dict[str, dict], tasks: list[str], max_k: int = MAX_K) -> dict[str, float]:
    """The same curve for a deterministic scripted reference, whose rows count the proposals made."""
    return {name: v4._mean(rows[task]["success"] and (k is None or rows[task]["probes"] - 1 <= k) for task in tasks)
            for name, k in _caps(max_k).items()}


def first_trials(traces: list[dict], n: int) -> list[dict]:
    return sorted(traces, key=lambda trace: trace["trial"])[:n]


def separated_pairs(configs: dict[str, dict[str, float]], family_of: dict[str, str], name: str) -> dict:
    """Configuration pairs whose paired family-bootstrap 95% interval for the pass^1 difference excludes zero."""
    by_pair = {}
    for a, b in itertools.combinations(sorted(configs), 2):
        shared = sorted(set(configs[a]) & set(configs[b]))
        diff = v4._family_mean({task: configs[a][task] - configs[b][task] for task in shared}, family_of)
        boot = v4._bootstrap(f"{name}:{a}|{b}", diff, None)
        low, high = v4._percentile(boot, 0.025), v4._percentile(boot, 0.975)
        mean = sum(boot) / len(boot)
        se = (sum((value - mean) ** 2 for value in boot) / (len(boot) - 1)) ** 0.5
        by_pair[f"{a}|{b}"] = {"estimate": sum(diff.values()) / len(diff), "ci95": [low, high],
                               "standard_error": se, "separated": low > 0 or high < 0}
    return {"pairs": len(by_pair), "separated": sum(row["separated"] for row in by_pair.values()), "by_pair": by_pair}


def uncapped_level(by_task: dict[str, list[dict]]) -> float:
    """pass^1 over the trials that did not end at the wall-clock cap, averaged over tasks that keep any."""
    kept = [[trace for trace in traces if trace.get("stop_reason") != "wall_cap"] for traces in by_task.values()]
    return v4._mean(v4._pass(traces, 1) for traces in kept if traces)


def order_flips(first: dict[str, float], second: dict[str, float]) -> list[str]:
    """Configuration pairs whose order differs between two sets of levels (a tie is no order)."""
    return [f"{a}|{b}" for a, b in itertools.combinations(sorted(first), 2)
            if (first[a] - first[b]) * (second[a] - second[b]) < 0]


def hidden_fields(task: dict) -> tuple[str, ...]:
    """The fields of a task's hidden predicates."""
    return tuple(sorted({c["constraint"]["field"] for c in task["constraints"] if c["reveal"] == "hidden"}))


def group_means(values: dict[str, float], groups: dict[str, list[str]]) -> dict[str, dict]:
    """The mean of ``values`` over each group's tasks (a task may sit in several groups), with the count."""
    return {group: {"tasks": len(members), "pass^1": v4._mean(values[t] for t in members)}
            for group, members in groups.items() if members}


def popularity_percentile(item: str, pool: list[str], sitelinks: dict[str, int]) -> float:
    """The share of ``pool`` with fewer sitelinks than ``item``, ties counting half."""
    links = sitelinks[item]
    return (sum(sitelinks[x] < links for x in pool) + 0.5 * sum(sitelinks[x] == links for x in pool)) / len(pool)


def cue_split(families: list[dict], answers: dict, tasks: dict) -> dict:
    """Families, and NVR families, with and without a cue that occurs only in satisfiable families.

    The cues are a reading history that holds a valid answer and a user who accepts one format.
    """
    out = {"cued": {"families": 0, "nvr": 0}, "uncued": {"families": 0, "nvr": 0}}
    for row in families:
        task = row["task_ids"]["volunteer"]
        answer = answers[task]
        cued = bool(set(answer.get("read_ids", [])) & set(answer["constraint_solutions"])) or len(
            tasks[task]["user_formats"]) == 1
        group = out["cued" if cued else "uncued"]
        group["families"] += 1
        group["nvr"] += row["no_valid_recommendation"]
    return out


def all_on_ask_disclosed(trace: dict, task: dict) -> bool:
    """Whether the user disclosed every on_ask predicate of the task during the trial."""
    on_ask = {i for i, c in enumerate(task["constraints"]) if c["reveal"] == "on_ask"}
    disclosed = {i for d in trace.get("simulator_decisions", []) for i in d.get("disclosed_constraint_indices") or []}
    return on_ask <= disclosed


def trial_cost(trace: dict) -> float:
    """The agent's API cost for one trial in US dollars: the provider-reported cost, else the estimate."""
    return sum(call["provider_reported_cost_usd"] if call.get("provider_reported_cost_usd") is not None
               else call.get("estimated_cost_usd") or 0.0
               for call in (trace.get("api_calls") or {}).get("agent", []))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scripted", type=Path, help="scripted_reference_v4.json, for its capped hidden success")
    parser.add_argument("--eliminating", type=Path,
                        help="run_scripted_reference.py --order eliminate --hypotheses broad output (the reported eliminating agent)")
    parser.add_argument("--eliminating-template", type=Path, help="the same with the generator's template shapes")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    agents = [row["label"] for row in plan["models"]["agents"]]
    families = json.loads((args.artifact / "family_manifest.json").read_text())["families"]
    family_of, template_of, nvr = {}, {}, {}
    for row in families:
        template_of[row["family_id"]] = "|".join(
            sorted({":".join(part.split(":")[:2]) for part in row["constraint_signature"].split("|")}))
        nvr[row["family_id"]] = row["no_valid_recommendation"]
        for task_id in row["task_ids"].values():
            family_of[task_id] = row["family_id"]
    tasks = {path.stem: json.loads(path.read_text()) for path in (args.artifact / "tasks").glob("*.json")}
    out: dict = {"seed": v4.SEED, "iterations": v4.ITERATIONS, "sign_flips": FLIPS, "status": "post hoc"}

    primary = _primary(args.root)
    effects = _effects(primary, agents)

    # Interface contrast by complexity tier.
    tiers = {}
    for who, values in effects.items():
        for tier in TIERS:
            subset = {task: value for task, value in values.items() if tasks[task]["complexity"] == tier}
            name = f"tier/interface/{who}/{tier}/pass^1"
            row = v4.interval(name, v4._family_mean(subset, family_of), template_of)
            row["tasks"] = len(subset)
            tiers[name] = row
    out["interface_by_tier"] = tiers
    out["year_predicate_families_by_tier"] = {
        tier: {
            "families": len({family_of[t] for t, task in tasks.items() if task["complexity"] == tier
                             and family_of[t] in {family_of[x] for x in effects["model_averaged"]}}),
            "with_year_predicate": len({family_of[t] for t, task in tasks.items() if task["complexity"] == tier
                                        and family_of[t] in {family_of[x] for x in effects["model_averaged"]}
                                        and any(c["constraint"]["field"] == "publication_year"
                                                for c in task["constraints"])}),
        }
        for tier in TIERS
    }

    # Randomization tests and the minimum detectable effect.
    tests = {}
    for who in ("model_averaged", *agents, "without_qwen38_27b_high"):
        values = (
            {task: sum(effects[a][task] for a in agents if a != "qwen38_27b_high") / (len(agents) - 1)
             for task in effects["model_averaged"]}
            if who == "without_qwen38_27b_high" else effects[who]
        )
        family_values = v4._family_mean(values, family_of)
        template_p, clusters = _sign_flip_template(family_values, template_of)
        tests[who] = {
            "estimate": sum(family_values.values()) / len(family_values),
            "family_sign_flip_p": _sign_flip_family(who, family_values),
            "template_exact_sign_flip_p": template_p,
            "templates": clusters,
        }
    out["randomization_tests"] = tests
    boot = v4._bootstrap(PRIMARY, v4._family_mean(effects["model_averaged"], family_of), None)
    mean = sum(boot) / len(boot)
    se = (sum((value - mean) ** 2 for value in boot) / (len(boot) - 1)) ** 0.5
    out["primary_minimum_detectable_effect"] = {"standard_error": se, "mde_alpha05_power80": (1.959964 + 0.841621) * se}

    # Stalled first attempts in place of their re-runs.
    first = _effects(_primary(args.root, quarantined=True), agents)["model_averaged"]
    out["primary_with_stalled_first_attempts"] = sum(v4._family_mean(first, family_of).values()) / len(
        v4._family_mean(first, family_of))

    # Non-Anglo predicates.
    def non_anglo(task: dict) -> bool:
        return any(c["constraint"]["field"] in ANGLO and c["constraint"]["value"] not in ANGLO[c["constraint"]["field"]]
                   for c in task["constraints"])

    pooled = {task: v4._mean(v4._pass(primary[(a, arm)][task], 1) for a in agents for arm in v4.ARMS)
              for task in effects["model_averaged"]}
    split = {}
    for group, keep in (("non_anglo", True), ("anglo", False)):
        members = {t: v for t, v in pooled.items() if non_anglo(tasks[t]) is keep}
        split[group] = group_levels(members, family_of, nvr)
    out["non_anglo_split"] = split
    all_nvr = [row["family_id"] for row in families if row["no_valid_recommendation"]]
    out["nvr_families_with_non_anglo_predicates"] = {
        "all": sum(non_anglo(tasks[next(t for t, f in family_of.items() if f == family and t.endswith("_v"))])
                   for family in all_nvr),
        "of": len(all_nvr),
    }

    # Rejected proposals per successful hidden trial.
    burden = {}
    for agent in agents:
        wins = [trace for arm in v4.ARMS for task, traces in primary[(agent, arm)].items() if task.endswith("_h")
                for trace in traces if trace["evaluation"]["primary_reward"] >= 1 - 1e-9]
        rejected = [sum(d.get("action") == "candidate_verdict" and d.get("verdict") == "rejected"
                        for d in trace.get("simulator_decisions", [])) for trace in wins]
        burden[agent] = {
            "hidden_successes": len(wins),
            "rejections_per_success": v4._mean(rejected),
            "share_after_three_or_more_rejections": v4._mean(r >= 3 for r in rejected),
        }
    out["hidden_rejections_per_success"] = burden

    # Hidden success under a patience cap: a trial counts only if it succeeds
    # after at most k rejected proposals. Satisfiable hidden tasks of the
    # primary panel, pass^1 averaged over tasks; k = 3 is the reported cap.
    caps = {f"k{k}": k for k in range(1, 6)} | {"uncapped": None}
    hidden_sat = sorted({task for (agent, arm) in primary for task in primary[(agent, arm)]
                         if task.endswith("_h") and not nvr[family_of[task]]})
    capped = {}
    for agent in agents:
        capped[agent] = {name: v4._mean(capped_success(primary[(agent, arm)][task], k)
                                        for arm in v4.ARMS for task in hidden_sat if task in primary[(agent, arm)])
                         for name, k in caps.items()}
    capped["model_averaged"] = {name: v4._mean(capped[agent][name] for agent in agents) for name in caps}
    if args.scripted:
        rows = {row["task_id"]: row for row in json.loads(args.scripted.read_text())["rows"]}
        capped["scripted"] = {name: v4._mean(rows[task]["success"] and (k is None or rows[task]["probes"] - 1 <= k)
                                             for task in hidden_sat) for name, k in caps.items()}
    out["hidden_capped_success"] = capped

    # The whole curve, k = 0..MAX_K, for each agent (pooled over interfaces) and each scripted reference.
    curves = {agent: success_curve({(arm, task): primary[(agent, arm)][task] for arm in v4.ARMS for task in hidden_sat},
                                   [(arm, task) for arm in v4.ARMS for task in hidden_sat if task in primary[(agent, arm)]])
              for agent in agents}
    curves["model_averaged"] = {name: v4._mean(curves[agent][name] for agent in agents) for name in _caps(MAX_K)}
    references = (("scripted", args.scripted), ("eliminating", args.eliminating),
                  ("eliminating_template", args.eliminating_template))
    for label, path in references:
        if path:
            curves[label] = reference_curve({row["task_id"]: row for row in json.loads(path.read_text())["rows"]},
                                            hidden_sat)
    out["hidden_success_curve"] = curves
    # The references on the reliability block alone, to set beside the prompt follow-ups, which ran only there.
    reliability = {t for stage in plan["stages"] if stage["name"] == "reliability" for t in stage["task_ids"]}
    out["hidden_success_curve_reliability"] = {
        label: reference_curve({row["task_id"]: row for row in json.loads(path.read_text())["rows"]},
                               [t for t in hidden_sat if t in reliability])
        for label, path in references if path
    }


    # pass^1 by condition on satisfiable families: overall, and per agent with family intervals.
    out["levels"] = group_levels(pooled, family_of, nvr)
    per_agent = {}
    for agent in agents:
        values = {task: v4._mean(v4._pass(primary[(agent, arm)][task], 1) for arm in v4.ARMS)
                  for task in effects["model_averaged"]}
        for letter, reveal in v4.REVEALS.items():
            chosen = {t: v for t, v in values.items() if t.endswith(f"_{letter}") and not nvr[family_of[t]]}
            row = v4.interval(f"levels/{agent}/{reveal}", v4._family_mean(chosen, family_of), template_of)
            per_agent[f"{agent}/{reveal}_satisfiable"] = {"estimate": row["estimate"], "family_ci95": row["family_ci95"]}
    out["satisfiable_levels_by_agent"] = per_agent

    # Discriminative power over the agent-interface configurations.
    def configs(trials: int | None, keep) -> dict[str, dict[str, float]]:
        return {f"{agent}/{arm}": {task: v4._pass(first_trials(traces, trials) if trials else traces, 1)
                                   for task, traces in primary[(agent, arm)].items() if keep(task)}
                for agent in agents for arm in v4.ARMS}
    hidden_sat_set = set(hidden_sat)
    power = {}
    for subset, keep in (("all", lambda task: task in effects["model_averaged"]),
                         ("hidden_satisfiable", lambda task: task in hidden_sat_set)):
        for trials in (None, 2, 1):
            result = separated_pairs(configs(trials, keep), family_of, f"pairs/{subset}/{trials or 'all'}")
            ses = sorted(row["standard_error"] for row in result["by_pair"].values())
            power[f"{subset}/{'all_trials' if trials is None else f'first_{trials}'}"] = {
                "pairs": result["pairs"], "separated": result["separated"],
                "median_mde": (1.959964 + 0.841621) * ses[len(ses) // 2],
                **({"by_pair": result["by_pair"]} if trials is None else {}),
            }
    out["discriminative_power"] = power

    # Does the wall-clock cap reorder the agent-interface configurations?
    with_caps = {f"{agent}/{arm}": v4._mean(v4._pass(traces, 1) for traces in by_task.values())
                 for (agent, arm), by_task in primary.items()}
    without_caps = {f"{agent}/{arm}": uncapped_level(by_task) for (agent, arm), by_task in primary.items()}
    out["uncapped_ordering"] = {"levels": with_caps, "uncapped_levels": without_caps,
                                "flipped_pairs": order_flips(with_caps, without_caps)}

    # Families whose reading list can stand in for asking: a history and two hidden predicates.
    answers = json.loads((args.artifact / "answers.json").read_text())
    leaky = {row["family_id"] for row in families
             if answers[row["task_ids"]["volunteer"]].get("read_ids")
             and sum(c["reveal"] == "hidden" for c in tasks[row["task_ids"]["hidden"]]["constraints"]) == 2}
    out["history_two_hidden_split"] = {
        group: {"tasks": len(chosen), "pass^1/hidden_satisfiable": v4._mean(pooled[t] for t in chosen)}
        for group, chosen in (("history_two_hidden", [t for t in hidden_sat if family_of[t] in leaky]),
                              ("other", [t for t in hidden_sat if family_of[t] not in leaky]))
    }

    # Satisfiable families by tier and condition; hidden satisfiable tasks by hidden field.
    satisfiable = [t for t in pooled if not nvr[family_of[t]]]
    out["levels_by_tier"] = group_means(pooled, {
        f"{reveal}/{tier}": [t for t in satisfiable if t.endswith(f"_{letter}") and tasks[t]["complexity"] == tier]
        for letter, reveal in v4.REVEALS.items() for tier in TIERS})
    by_field = {f"hidden/{field}": [t for t in hidden_sat if field in hidden_fields(tasks[t])]
                for field in ("genres", "languages", "countries", "publication_year")}
    revealed_genre = {t for t in hidden_sat if any(c["constraint"]["field"] == "genres" and c["reveal"] != "hidden"
                                                   for c in tasks[t]["constraints"])}
    by_field["hidden/genres/another_genre_revealed"] = [t for t in by_field["hidden/genres"] if t in revealed_genre]
    by_field["hidden/genres/no_genre_revealed"] = [t for t in by_field["hidden/genres"] if t not in revealed_genre]
    out["hidden_levels_by_field"] = group_means(pooled, by_field)

    # How popular successful recommendations are within their answer sets, per interface.
    sitelinks = {row["id"]: row.get("sitelink_count") or 0
                 for row in json.loads((args.artifact / "catalog.json").read_text())}
    percentiles: dict[str, list[float]] = defaultdict(list)
    for (agent, arm), by_task in primary.items():
        for task, traces in by_task.items():
            pool = [x for x in answers[task]["constraint_solutions"] if x not in set(answers[task]["read_ids"])]
            if len(pool) < 5:
                continue
            for trace in traces:
                if trace["evaluation"]["primary_reward"] < 1 - 1e-9:
                    continue
                item = next(e["arguments"].get("item_id") for e in reversed(trace["events"])
                            if e.get("type") == "tool_call" and e.get("name") == "recommend")
                percentiles[arm].append(popularity_percentile(item, pool, sitelinks))
    out["recommendation_popularity"] = {
        arm: {"successes": len(values), "median_percentile": sorted(values)[len(values) // 2],
              "top_decile_share": v4._mean(value >= 0.9 for value in values)}
        for arm, values in sorted(percentiles.items())}

    # Cues that occur only in satisfiable families.
    volunteer_tasks = {row["task_ids"]["volunteer"]: tasks[row["task_ids"]["volunteer"]] for row in families}
    out["satisfiable_cues"] = cue_split(families, answers, volunteer_tasks)

    # Agent API cost and runtime per trial, and for one agent's reportable run (153 tasks, four trials each).
    panel_tasks = len(effects["model_averaged"])
    cost = {}
    for (agent, arm), by_task in sorted(primary.items()):
        traces = [trace for traces in by_task.values() for trace in traces]
        per_trial = v4._mean(trial_cost(trace) for trace in traces)
        runtime = v4._mean(trace.get("trial_runtime_s") or 0.0 for trace in traces)
        cost[f"{agent}/{arm}"] = {"trials": len(traces), "usd_per_trial": per_trial, "runtime_s_per_trial": runtime,
                                  "reportable_run_usd": per_trial * panel_tasks * 4,
                                  "reportable_run_trial_hours": runtime * panel_tasks * 4 / 3600}
    out["run_cost"] = cost

    # Mixed success on satisfiable tasks by whether every on_ask predicate was disclosed.
    split: dict[str, Counter] = {"complete": Counter(), "incomplete": Counter()}
    for by_task in primary.values():
        for task, traces in by_task.items():
            if not task.endswith("_m") or nvr[family_of[task]]:
                continue
            for trace in traces:
                group = split["complete" if all_on_ask_disclosed(trace, tasks[task]) else "incomplete"]
                group.update({"trials": 1, "successes": trace["evaluation"]["primary_reward"] >= 1 - 1e-9})
    out["mixed_success_by_elicitation"] = {group: dict(counts) for group, counts in split.items()}

    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    for who, row in tests.items():
        print(f"  {who}: {row['estimate']:+.3f}  family sign-flip p={row['family_sign_flip_p']:.4f}  "
              f"template exact p={row['template_exact_sign_flip_p']:.4f} ({row['templates']} templates)")
    for tier in TIERS:
        row = tiers[f"tier/interface/model_averaged/{tier}/pass^1"]
        print(f"  {tier}: {row['estimate']:+.3f} [{row['family_ci95'][0]:+.3f}, {row['family_ci95'][1]:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
