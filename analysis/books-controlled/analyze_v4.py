#!/usr/bin/env python3
"""Analyze the v4 books matrix exactly as experiment_plan_v4.json pre-specifies it.

Every number comes from traces on disk. The sampling unit is the task family;
trials are repeated measurements. Intervals are seeded paired bootstraps over
families (B = 10,000). The two-sided p-value is (k + 1) / (B + 1) with
k = 2 * min(#replicates <= 0, #replicates >= 0), so the smallest attainable p is
1 / (B + 1), as the plan's Holm floor note states. A template-clustered
bootstrap, resampling whole field templates, is reported alongside as a check.

- Primary: 20k structured-minus-keyword pass^1 on the primary panel, all
  three disclosure conditions, averaged over the four agents; tested alone.
- Secondary family, Holm-adjusted: exactly the eleven contrasts the plan lists.
- Everything else is exploratory or descriptive, with unadjusted intervals.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from tau_rec.metrics.pass_k import pass_at_k

SEED = 20261014
ITERATIONS = 10_000
REVEALS = {"v": "volunteer", "m": "mixed", "h": "hidden"}
SIZES = (5000, 10000, 20000)
ARMS = ("keyword", "structured")
STAGES = ("primary_remainder", "reliability", "scale_all", "simulator_sensitivity")
# Paraphrases an LLM user may use for a catalog label. England, Scotland and
# Wales are separate catalog countries, so they do not count for United Kingdom.
ALIASES = {
    "Children": ("children", "childrens", "kid"),
    "Graphic Novel": ("graphic novel", "comic"),
    "Historical Fiction": ("historical fiction", "historical novel", "historical-fiction"),
    "Humor": ("humor", "humour", "humorous", "funny", "comedy", "comedic"),
    "Nonfiction": ("nonfiction", "non-fiction", "non fiction"),
    "Poetry": ("poetry", "poem", "poet"),
    "Science Fiction": ("science fiction", "sci-fi", "sci fi", "scifi"),
    "Self-Help": ("self-help", "self help"),
    "Young Adult": ("young adult", "ya", "teen", "teenage"),
    "Memoir": ("memoir", "autobiography"),
    "United Kingdom": ("united kingdom", "uk", "u.k.", "britain", "great britain", "british"),
    "United States": ("united states", "u.s.", "usa", "u.s.a.", "america", "american"),
}
ID_PATTERN = re.compile(r'"id":\s*"(wd_Q\d+)"')
DISCLOSURE_NOTE = (
    "lexical: a user message contains the constraint value; the tolerant form also accepts the "
    "listed aliases and plural endings; years must appear as the number; negation is not parsed, "
    "so both forms are upper bounds on correct disclosure"
)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] * (1 - (position - lower)) + ordered[upper] * (position - lower)


def _bootstrap(name: str, family_values: dict[str, float], clusters: dict[str, str] | None) -> list[float]:
    rng = random.Random(f"{SEED}:{name}:{'cluster' if clusters else 'family'}")
    if not clusters:
        values = [family_values[family] for family in sorted(family_values)]
        return [sum(rng.choice(values) for _ in values) / len(values) for _ in range(ITERATIONS)]
    groups: dict[str, list[float]] = defaultdict(list)
    for family in sorted(family_values):
        groups[clusters[family]].append(family_values[family])
    members = [groups[template] for template in sorted(groups)]
    out = []
    for _ in range(ITERATIONS):
        draw = [value for _ in members for value in rng.choice(members)]
        out.append(sum(draw) / len(draw))
    return out


def template_sign_flip_p(family_values: dict[str, float], clusters: dict[str, str]) -> float:
    """Exact two-sided sign-flip test over every sign pattern of the per-template sums (post hoc)."""
    sums: dict[str, float] = defaultdict(float)
    for family, value in family_values.items():
        sums[clusters[family]] += value
    values = [sums[template] for template in sorted(sums)]
    observed = abs(sum(values))
    hits = sum(
        abs(sum(v if mask >> i & 1 else -v for i, v in enumerate(values))) >= observed - 1e-12
        for mask in range(2 ** len(values))
    )
    return hits / 2 ** len(values)


def interval(name: str, family_values: dict[str, float], clusters: dict[str, str]) -> dict:
    estimate = sum(family_values.values()) / len(family_values)
    result = {"estimate": estimate, "families": len(family_values)}
    for label, grouping in (("family", None), ("template", clusters)):
        boot = _bootstrap(name, family_values, grouping)
        k = 2 * min(sum(value <= 0 for value in boot), sum(value >= 0 for value in boot))
        result[f"{label}_ci95"] = [_percentile(boot, 0.025), _percentile(boot, 0.975)]
        result[f"{label}_p"] = min(1.0, (k + 1) / (ITERATIONS + 1))
    result["template_sign_flip_p"] = template_sign_flip_p(family_values, clusters)
    return result


def _family_mean(task_values: dict[str, float], family_of: dict[str, str]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for task, value in task_values.items():
        grouped[family_of[task]].append(value)
    return {family: sum(values) / len(values) for family, values in grouped.items()}


def _trials(run_dir: Path) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((run_dir / "traces").glob("*.json")):
        trace = json.loads(path.read_text())
        by_task[trace["task_id"]].append(trace)
    return by_task


def _pass(traces: list[dict], k: int) -> float | None:
    n = len(traces)
    c = sum(trace["evaluation"]["primary_reward"] >= 1 - 1e-9 for trace in traces)
    return pass_at_k(n, c, k) if n >= k else None


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _load_cells(root: Path, stage: str) -> list[dict]:
    state_path = root / f"{stage}_v4_state.json"
    if not state_path.is_file():
        raise SystemExit(f"no traces found: {state_path} does not exist")
    state = json.loads(state_path.read_text())
    if any(cell["status"] != "complete" for cell in state["cells"]):
        raise SystemExit(f"{stage}: not every cell is complete")
    # Run directories are stored relative to the directory the run (or the
    # extracted archive) lives in, so analyses run from there.
    unreachable = [cell["run_dir"] for cell in state["cells"]
                   if not any((Path(cell["run_dir"]) / "traces").glob("*.json"))]
    if unreachable:
        raise SystemExit(f"no traces found for {len(unreachable)} {stage} cell(s), e.g. {unreachable[0]}; run from "
                         "the directory that contains out/ (run directories are relative to it)")
    return state["cells"]


def _provider_failed(trace: dict, role: str) -> bool:
    """The role's provider never answered: no successful call after the harness's retries."""
    calls = trace.get("api_calls", {}).get(role, [])
    return bool(calls) and not any(call.get("status") == "ok" for call in calls)


def _user_messages(trace: dict) -> list[str]:
    return [
        event["content"].lower() for event in trace["events"]
        if event.get("type") == "message" and event["role"] == "user"
    ]


def _stated(text: str, value, *, tolerant: bool) -> bool:
    spellings = [str(value).lower()]
    if tolerant and isinstance(value, str):
        spellings += list(ALIASES.get(value, ()))
    ending = r"(?:s|es)?" if tolerant else ""
    return any(re.search(r"(?<!\w)" + re.escape(spelling) + ending + r"(?!\w)", text) for spelling in spellings)


def _values(task: dict, reveal: str) -> list:
    return [tagged["constraint"]["value"] for tagged in task["constraints"] if tagged["reveal"] == reveal]


def _decided_on_ask(trace: dict, task: dict) -> tuple[int, int]:
    """(askable constraints, askable constraints the symbolic controller disclosed)."""
    on_ask = {i for i, tagged in enumerate(task["constraints"]) if tagged["reveal"] == "on_ask"}
    disclosed = {
        index for decision in trace.get("simulator_decisions", [])
        for index in decision.get("disclosed_constraint_indices", [])
    }
    return len(on_ask), len(on_ask & disclosed)


def _exposed(trace: dict) -> set[str]:
    """Catalog ids returned to the agent by search or metadata calls."""
    found: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"id", "item_id"} and isinstance(child, str):
                    found.add(child)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for event in trace["events"]:
        if event.get("type") == "tool_call" and event.get("name") in {"search_catalog", "get_metadata"}:
            try:
                walk(json.loads(event.get("result") or ""))
            except json.JSONDecodeError:
                pass
    return found


def _max_step_tool_calls(trace: dict) -> int:
    """The most tool calls the agent made between two user messages."""
    most = current = 0
    for event in trace["events"]:
        if event.get("type") == "tool_call":
            current += 1
            most = max(most, current)
        elif event.get("type") == "message" and event["role"] == "user":
            current = 0
    return most


def _searches(trace: dict) -> int:
    return sum(event.get("type") == "tool_call" and event["name"] == "search_catalog" for event in trace["events"])


def _density_terciles(artifact: Path, scripted_path: Path, task_ids: list[str], pooled: dict[str, float],
                      answers: dict) -> dict:
    """Hidden satisfiable tasks in thirds of answer density.

    Density is the number of policy-eligible answers per catalog work that
    matches every predicate the hidden user ever states.
    """
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import RevealTag, Task
    from tau_rec.domain import BOOKS

    catalog = Catalog.from_json(str(artifact / "catalog.json"), BOOKS)
    scripted = {row["task_id"]: row["success"] for row in json.loads(scripted_path.read_text())["rows"]}
    density = {}
    for task_id in task_ids:
        task = Task.from_json(str(artifact / "tasks" / f"{task_id}.json"))
        revealed = [tagged.constraint for tagged in task.constraints if tagged.reveal != RevealTag.HIDDEN]
        matches = sum(all(constraint.evaluate(item) for constraint in revealed) for item in catalog.items)
        density[task_id] = len(answers[task_id]["policy_eligible_solutions"]) / max(1, matches)
    ordered = sorted(task_ids, key=density.get)
    thirds = [ordered[i * len(ordered) // 3:(i + 1) * len(ordered) // 3] for i in range(3)]
    return {
        name: {
            "tasks": len(group),
            "median_density": statistics.median(density[t] for t in group),
            "scripted_success": sum(scripted[t] for t in group) / len(group),
            "agents_pass^1": sum(pooled[t] for t in group) / len(group),
        }
        for name, group in zip(("sparsest", "middle", "densest"), thirds)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scripted", type=Path, help="scripted reference rows, for the density breakdown")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    agents = [row["label"] for row in plan["models"]["agents"]]
    scale_agents = plan["models"]["scale_agents"]
    sensitivity_agent = plan["models"]["sensitivity_agent"]
    llm_users = [row["label"] for row in plan["models"]["llm_simulators"]]

    families = json.loads((args.artifact / "family_manifest.json").read_text())["families"]
    family_of, template_of, nvr_family = {}, {}, {}
    for row in families:
        template_of[row["family_id"]] = "|".join(
            sorted({":".join(part.split(":")[:2]) for part in row["constraint_signature"].split("|")})
        )
        nvr_family[row["family_id"]] = row["no_valid_recommendation"]
        for task_id in row["task_ids"].values():
            family_of[task_id] = row["family_id"]
    tasks = {path.stem: json.loads(path.read_text()) for path in (args.artifact / "tasks").glob("*.json")}
    answers = json.loads((args.artifact / "answers.json").read_text())

    confirmatory: dict[str, dict] = {}
    exploratory: dict[str, dict] = {}
    sensitivities: dict[str, dict] = {}
    levels: dict[str, dict] = {}

    def estimate(store: dict, name: str, task_values: dict[str, float]) -> dict:
        store[name] = interval(name, _family_mean(task_values, family_of), template_of)
        store[name]["tasks"] = len(task_values)
        return store[name]

    def level(name: str, task_values: dict[str, float]) -> None:
        row = interval(name, _family_mean(task_values, family_of), template_of)
        levels[name] = {key: value for key, value in row.items() if not key.endswith("_p")}
        levels[name]["tasks"] = len(task_values)

    # Load every cell, then fix provider-failure exclusions before looking at any outcome.
    cells = {stage: _load_cells(args.root, stage) for stage in STAGES}
    data: dict[tuple, dict[str, list[dict]]] = defaultdict(dict)
    for stage, rows in cells.items():
        for row in rows:
            key = (stage, row["agent_label"], row.get("simulator_label"), row["retrieval_mode"], row["catalog_size"])
            data[key].update(_trials(Path(row["run_dir"])))
    failed: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for (stage, agent, simulator, _, _), by_task in data.items():
        for task, traces in by_task.items():
            for trace in traces:
                failed[(agent, family_of[task])].append(_provider_failed(trace, "agent"))
                if simulator:
                    failed[(simulator, family_of[task])].append(_provider_failed(trace, "simulator"))
    excluded = sorted({pair for pair, flags in failed.items() if all(flags)})

    def cell(stage, agent, arm, size=20000, simulator=None) -> dict[str, list[dict]]:
        return {
            task: traces for task, traces in data[(stage, agent, simulator, arm, size)].items()
            if (agent, family_of[task]) not in excluded and (simulator, family_of[task]) not in excluded
        }

    result: dict = {
        "plan_version": plan["version"],
        "seed": SEED,
        "iterations": ITERATIONS,
        "p_value": "two-sided (k + 1) / (B + 1), k = 2 min(#<=0, #>=0)",
        "provider_failure_exclusions": [{"model": model, "family": family} for model, family in excluded],
        "stall_quarantine": {
            stage: {
                "quarantined_traces": sum(len(row.get("quarantined_traces") or []) for row in rows),
                "cells_with_stall_retries": sum(bool(row.get("stall_retries")) for row in rows),
            }
            for stage, rows in cells.items()
        },
        "run_accounting": {
            stage: {
                "trials": sum(row["trace_pairs"] for row in rows),
                "provider_spend_usd": json.loads((args.root / f"{stage}_v4_state.json").read_text())["provider_spend_usd"],
                "git_commit": json.loads((args.root / f"{stage}_v4_state.json").read_text())["git_commit"],
                "plan_sha256": json.loads((args.root / f"{stage}_v4_state.json").read_text())["plan_sha256"],
            }
            for stage, rows in cells.items()
        },
        # Trials that crashed without a trace; the launcher re-ran each until every planned trial had one.
        "crashed_trials_rerun": dict(Counter(
            "network" if "Cannot connect" in line else "malformed_tool_json" if "JSONDecodeError" in line
            else "provider_error"
            for log in sorted(args.root.glob("*launcher*.log")) if not log.name.startswith("smoke")
            for line in log.read_text(errors="replace").splitlines() if line.startswith("  FAIL  ")
        )),
        "stop_classes_per_cell": {
            "/".join(str(part) for part in key if part is not None): dict(sorted(Counter(
                trace["stop_reason"] for traces in by_task.values() for trace in traces
            ).items()))
            for key, by_task in sorted(data.items(), key=lambda item: str(item[0]))
        },
    }

    # Primary panel at 20k: the K=4 remainder plus the nested K=8 reliability block.
    primary = {
        (agent, arm): {**cell("primary_remainder", agent, arm), **cell("reliability", agent, arm)}
        for agent in agents for arm in ARMS
    }
    reliability = {(agent, arm): cell("reliability", agent, arm) for agent in agents for arm in ARMS}
    table = {}
    for (agent, arm), by_task in primary.items():
        row = {f"pass^{k}": _mean(_pass(t, k) for t in by_task.values()) for k in (1, 2, 4)}
        row.update({f"reliability/pass^{k}": _mean(_pass(t, k) for t in reliability[(agent, arm)].values())
                    for k in (1, 2, 4, 8)})
        for letter, reveal in REVEALS.items():
            subset = {task: t for task, t in by_task.items() if task.endswith(f"_{letter}")}
            row[f"pass^1/{reveal}"] = _mean(_pass(t, 1) for t in subset.values())
            for status in ("satisfiable", "nvr"):
                row[f"pass^1/{reveal}/{status}"] = _mean(
                    _pass(t, 1) for task, t in subset.items() if nvr_family[family_of[task]] == (status == "nvr")
                )
        mixed = [_decided_on_ask(trace, tasks[task]) for task, traces in by_task.items() if task.endswith("_m")
                 for trace in traces]
        row["mixed_complete_elicitation"] = _mean(asked == got for asked, got in mixed)
        row["mixed_no_elicitation"] = _mean(got == 0 for _, got in mixed)
        row["mixed_askable_share_elicited"] = sum(got for _, got in mixed) / sum(asked for asked, _ in mixed)
        mixed_trials = [(task, trace) for task, traces in by_task.items() if task.endswith("_m") for trace in traces]
        row["mixed_valid_exposure"] = _mean(
            bool(_exposed(trace) & set(answers[task]["policy_eligible_solutions"]))
            for task, trace in mixed_trials if not nvr_family[family_of[task]]
        )
        row["mixed_question_turns"] = _mean(
            sum(decision.get("intent") in {"field", "generic"} for decision in trace["simulator_decisions"])
            for _, trace in mixed_trials
        )
        asked = [
            {field for decision in trace["simulator_decisions"] if decision.get("intent") == "field"
             for field in decision.get("requested_fields", [])}
            for _, trace in mixed_trials
        ]
        for field in ("genres", "publication_year", "countries", "languages", "formats"):
            row[f"mixed_asked_{field}"] = _mean(field in fields for fields in asked)
        row["mixed_asked_year_country_or_language"] = _mean(
            bool(fields & {"publication_year", "countries", "languages"}) for fields in asked)
        complete = [trace for task, trace in mixed_trials
                    if (lambda pair: pair[0] == pair[1])(_decided_on_ask(trace, tasks[task]))]
        row["mixed_success_given_complete_elicitation"] = _mean(
            trace["evaluation"]["primary_reward"] >= 1 - 1e-9 for trace in complete)
        row["mixed_complete_elicitation_trials"] = len(complete)
        complete_tasks = [task for task, trace in mixed_trials
                          if (lambda pair: pair[0] == pair[1])(_decided_on_ask(trace, tasks[task]))]
        row["mixed_complete_elicitation_single_fact_trials"] = sum(
            len(_values(tasks[task], "on_ask")) == 1 for task in complete_tasks)
        wins = [(task, trace) for task, trace in mixed_trials
                if not nvr_family[family_of[task]] and trace["evaluation"]["primary_reward"] >= 1 - 1e-9]
        row["mixed_successes_without_complete_elicitation_share"] = _mean(
            (lambda pair: pair[0] != pair[1])(_decided_on_ask(trace, tasks[task])) for task, trace in wins)
        row["hidden_wall_cap_share"] = _mean(
            trace["stop_reason"] == "wall_cap" for task, traces in by_task.items() if task.endswith("_h")
            for trace in traces)
        row["wall_cap_share"] = _mean(trace["stop_reason"] == "wall_cap" for t in by_task.values() for trace in t)
        block = reliability[(agent, arm)]
        row["reliability/all_or_nothing_share"] = _mean(
            sum(trace["evaluation"]["primary_reward"] >= 1 - 1e-9 for trace in t) in {0, len(t)} for t in block.values())
        violations: dict[str, dict[str, int]] = {reveal: {} for reveal in REVEALS.values()}
        policy_only: Counter = Counter()
        for task, traces in by_task.items():
            for trace in traces:
                evaluation = trace["evaluation"]
                for name in evaluation["policy_detail"]["violations"]:
                    violations[REVEALS[task[-1]]][name] = violations[REVEALS[task[-1]]].get(name, 0) + 1
                    if evaluation["constraint_score"] >= 1 - 1e-9:
                        policy_only[name] += 1
        row["policy_violations"] = violations
        row["policy_only_failures"] = dict(policy_only)
        row["tool_calls_per_trial"] = _mean(
            trace["evaluation"]["efficiency"]["total_tool_calls"] for t in by_task.values() for trace in t
        )
        row["stop_classes"] = dict(Counter(trace["stop_reason"] for t in by_task.values() for trace in t))
        row["trials"] = sum(len(t) for t in by_task.values())
        row["tasks"] = len(by_task)
        table[f"{agent}/{arm}"] = row
        for letter, reveal in REVEALS.items():
            level(f"primary/{agent}/{arm}/{reveal}/pass^1",
                  {task: _pass(t, 1) for task, t in by_task.items() if task.endswith(f"_{letter}")})
    result["primary_table"] = table

    def interface_effects(agent: str, k: int, keep=lambda trace: True) -> dict[str, float]:
        keyword, structured = primary[(agent, "keyword")], primary[(agent, "structured")]
        out = {}
        for task in keyword:
            a = [trace for trace in keyword[task] if keep(trace)]
            b = [trace for trace in structured.get(task, []) if keep(trace)]
            if _pass(a, k) is not None and _pass(b, k) is not None:
                out[task] = _pass(b, k) - _pass(a, k)
        return out

    def averaged(effects: dict[str, dict[str, float]], over: list[str]) -> dict[str, float]:
        shared = set.intersection(*(set(effects[agent]) for agent in over))
        return {task: sum(effects[agent][task] for agent in over) / len(over) for task in shared}

    per_agent = {agent: interface_effects(agent, 1) for agent in agents}
    primary_name = "primary/interface/model_averaged/pass^1"
    estimate(confirmatory, primary_name, averaged(per_agent, agents))
    confirmatory[primary_name]["role"] = "primary"

    secondary = []
    for agent in agents:
        name = f"secondary/interface/{agent}/pass^1"
        estimate(confirmatory, name, per_agent[agent])
        secondary.append(name)
    pooled = {
        task: _mean(_pass(primary[(agent, arm)][task], 1) for agent in agents for arm in ARMS)
        for task in set.intersection(*(set(primary[key]) for key in primary))
    }
    result["disclosure_levels"] = {
        reveal: _mean(v for t, v in pooled.items() if t.endswith(f"_{letter}")) for letter, reveal in REVEALS.items()
    }
    pooled4 = {task: _mean(_pass(primary[(agent, arm)][task], 4) for agent in agents for arm in ARMS) for task in pooled}
    for k, values in ((1, pooled), (4, pooled4)):
        for letter, reveal in REVEALS.items():
            level(f"disclosure/{reveal}/pass^{k}", {t: v for t, v in values.items() if t.endswith(f"_{letter}")})
    name = "secondary/disclosure/mixed_minus_volunteer/pass^1"
    estimate(confirmatory, name, {t[:-1] + "v": v - pooled[t[:-1] + "v"] for t, v in pooled.items() if t.endswith("_m")})
    secondary.append(name)
    name = "secondary/disclosure/hidden_minus_volunteer/satisfiable/pass^1"
    estimate(confirmatory, name, {
        t[:-1] + "v": v - pooled[t[:-1] + "v"] for t, v in pooled.items()
        if t.endswith("_h") and not nvr_family[family_of[t]]
    })
    secondary.append(name)

    # Scale study: every satisfiable family, volunteer and mixed, three sizes, scale agents.
    scale = {(agent, arm, size): cell("scale_all", agent, arm, size)
             for agent in scale_agents for arm in ARMS for size in SIZES}
    result["scale_curves"] = {
        f"{agent}/{arm}/{size}": {
            "pass^1": _mean(_pass(t, 1) for t in by_task.values()),
            "pass^4": _mean(_pass(t, 4) for t in by_task.values()),
            "stop_classes": dict(Counter(trace["stop_reason"] for t in by_task.values() for trace in t)),
            "tool_calls_per_trial": _mean(
                trace["evaluation"]["efficiency"]["total_tool_calls"] for t in by_task.values() for trace in t
            ),
        }
        for (agent, arm, size), by_task in sorted(scale.items())
    }

    def scale_drop(agent: str, arm: str) -> dict[str, float]:
        small, large = scale[(agent, arm, 5000)], scale[(agent, arm, 20000)]
        return {task: _pass(small[task], 1) - _pass(large[task], 1) for task in small if task in large}

    drop = {arm: averaged({agent: scale_drop(agent, arm) for agent in scale_agents}, scale_agents) for arm in ARMS}
    for arm in ARMS:
        name = f"secondary/scale/{arm}/5k_minus_20k/pass^1"
        estimate(confirmatory, name, drop[arm])
        secondary.append(name)
    name = "secondary/scale/interaction/structured_drop_minus_keyword_drop/pass^1"
    estimate(confirmatory, name, {
        task: drop["structured"][task] - drop["keyword"][task] for task in set(drop["keyword"]) & set(drop["structured"])
    })
    secondary.append(name)

    # LLM users against the symbolic user: sensitivity agent, structured, 20k, reliability block.
    users = {"symbolic": cell("reliability", sensitivity_agent, "structured")}
    for label in llm_users:
        users[label] = cell("simulator_sensitivity", sensitivity_agent, "structured", simulator=label)
        name = f"secondary/simulator/{label}_minus_symbolic/pass^1"
        estimate(confirmatory, name, {
            task: _pass(t, 1) - _pass(users["symbolic"][task], 1)
            for task, t in users[label].items() if task in users["symbolic"]
        })
        secondary.append(name)

    if len(secondary) != 11:
        raise SystemExit(f"expected the 11 pre-specified secondary contrasts, built {len(secondary)}")
    ordered = sorted(secondary, key=lambda n: confirmatory[n]["family_p"])
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, confirmatory[name]["family_p"] * (len(ordered) - rank)))
        confirmatory[name]["holm_p"] = running
        confirmatory[name]["role"] = "secondary"
    # Exploratory robustness: the same Holm step-down on the template-clustered p-values.
    for field in ("template_p", "template_sign_flip_p"):
        running = 0.0
        for rank, name in enumerate(sorted(secondary, key=lambda n: confirmatory[n][field])):
            running = max(running, min(1.0, confirmatory[name][field] * (len(secondary) - rank)))
            confirmatory[name][f"holm_{field}"] = running

    # Pre-specified sensitivity: primary and per-agent interface without overflow and budget stops.
    keep = lambda trace: trace["stop_reason"] not in {"context_overflow", "tool_budget"}  # noqa: E731
    without = {agent: interface_effects(agent, 1, keep) for agent in agents}
    suffix = "excluding_context_overflow_and_tool_budget"
    estimate(sensitivities, f"{primary_name}/{suffix}", averaged(without, agents))
    for agent in agents:
        estimate(sensitivities, f"secondary/interface/{agent}/pass^1/{suffix}", without[agent])

    # Descriptive: user behaviour on the 54 sensitivity tasks, the same lexical measure for every user.
    simulator_table = {}
    for user, by_task in users.items():
        row = {f"pass^1/{reveal}": _mean(_pass(t, 1) for task, t in by_task.items() if task.endswith(f"_{letter}"))
               for letter, reveal in REVEALS.items()}
        row["pass^1"] = _mean(_pass(t, 1) for t in by_task.values())
        row["stop_classes"] = dict(Counter(trace["stop_reason"] for t in by_task.values() for trace in t))
        mixed = [(tasks[task], trace) for task, traces in by_task.items() if task.endswith("_m") for trace in traces]
        for form, tolerant in (("verbatim", False), ("tolerant", True)):
            all_stated, shares, first, hidden = [], [], [], []
            for task, trace in mixed:
                messages = _user_messages(trace)
                text = " ".join(messages)
                stated = [_stated(text, value, tolerant=tolerant) for value in _values(task, "on_ask")]
                all_stated.append(all(stated))
                shares.append(sum(stated) / len(stated))
                first.append(bool(messages) and any(
                    _stated(messages[0], value, tolerant=tolerant) for value in _values(task, "on_ask")))
            for task_id, traces in by_task.items():
                values = _values(tasks[task_id], "hidden")
                for trace in traces if values else []:
                    text = " ".join(_user_messages(trace))
                    hidden.append(any(_stated(text, value, tolerant=tolerant) for value in values))
            row[f"mixed/all_askable_stated/{form}"] = _mean(all_stated)
            row[f"mixed/askable_share_stated/{form}"] = _mean(shares)
            row[f"mixed/first_message_states_askable/{form}"] = _mean(first)
            row[f"hidden_value_stated/{form}"] = _mean(hidden)
            row["trials_with_hidden_constraints"] = len(hidden)
        if user == "symbolic":
            decided = [_decided_on_ask(trace, task) for task, trace in mixed]
            row["mixed/all_askable_disclosed/controller_decisions"] = _mean(a == g for a, g in decided)
            row["lexical_vs_decisions_agreement"] = _mean(
                (a == g) == lexical for (a, g), lexical in zip(decided, [
                    all(_stated(" ".join(_user_messages(trace)), value, tolerant=True)
                        for value in _values(task, "on_ask")) for task, trace in mixed
                ])
            )
        simulator_table[user] = row
    result["simulator_table"] = simulator_table
    result["disclosure_measure"] = DISCLOSURE_NOTE

    # Exploratory: unadjusted, never confirmatory.
    effects = {agent: interface_effects(agent, 4) for agent in agents}
    estimate(exploratory, "exploratory/interface/model_averaged/pass^4", averaged(effects, agents))
    overall = averaged(per_agent, agents)
    for letter, reveal in REVEALS.items():
        estimate(exploratory, f"exploratory/interface/model_averaged/{reveal}/pass^1",
                 {task: value for task, value in overall.items() if task.endswith(f"_{letter}")})
    for size in SIZES:
        estimate(exploratory, f"exploratory/scale/interface_at_{size}/scale_agents_averaged/pass^1", averaged({
            agent: {task: _pass(t, 1) - _pass(scale[(agent, "keyword", size)][task], 1)
                    for task, t in scale[(agent, "structured", size)].items()
                    if task in scale[(agent, "keyword", size)]}
            for agent in scale_agents
        }, scale_agents))
    for agent in scale_agents:
        for arm in ARMS:
            estimate(exploratory, f"exploratory/scale/{agent}/{arm}/5k_minus_20k/pass^1", scale_drop(agent, arm))
    for status in ("satisfiable", "nvr"):
        subset = {t: v for t, v in pooled.items() if t.endswith("_h") and nvr_family[family_of[t]] == (status == "nvr")}
        if subset:
            level(f"descriptive/hidden/{status}/model_and_arm_averaged/pass^1", subset)
    estimate(exploratory, "exploratory/disclosure/mixed_minus_volunteer/pass^4",
             {t[:-1] + "v": v - pooled4[t[:-1] + "v"] for t, v in pooled4.items() if t.endswith("_m")})
    estimate(exploratory, "exploratory/disclosure/hidden_minus_volunteer/satisfiable/pass^4", {
        t[:-1] + "v": v - pooled4[t[:-1] + "v"] for t, v in pooled4.items()
        if t.endswith("_h") and not nvr_family[family_of[t]]
    })
    for agent in agents:
        keyword, structured = reliability[(agent, "keyword")], reliability[(agent, "structured")]
        estimate(exploratory, f"exploratory/reliability/interface/{agent}/pass^8",
                 {t: _pass(structured[t], 8) - _pass(keyword[t], 8) for t in keyword if t in structured})
    # Test-retest: the scale study's 20k cells rerun the panel's satisfiable volunteer and mixed tasks.
    replication = {}
    for agent in scale_agents:
        for arm in ARMS:
            again, first = scale[(agent, arm, 20000)], primary[(agent, arm)]
            replication[f"{agent}/{arm}"] = {t: _pass(again[t], 1) - _pass(first[t], 1) for t in again if t in first}
            estimate(exploratory, f"exploratory/replication/{agent}/{arm}/20000/pass^1", replication[f"{agent}/{arm}"])
    estimate(exploratory, "exploratory/replication/model_averaged/20000/pass^1",
             averaged(replication, list(replication)))
    for (agent, arm, size), by_task in sorted(scale.items()):
        level(f"scale/{agent}/{arm}/{size}/pass^1", {t: _pass(v, 1) for t, v in by_task.items()})
    for user, by_task in users.items():
        for letter, reveal in REVEALS.items():
            level(f"simulator/{user}/{reveal}/pass^1",
                  {t: _pass(v, 1) for t, v in by_task.items() if t.endswith(f"_{letter}")})
        for other, letter in (("mixed", "m"), ("hidden", "h")):
            estimate(exploratory, f"exploratory/simulator/{user}/{other}_minus_volunteer/satisfiable/pass^4", {
                t[:-1] + "v": _pass(v, 4) - _pass(by_task[t[:-1] + "v"], 4) for t, v in by_task.items()
                if t.endswith(f"_{letter}") and not nvr_family[family_of[t]] and t[:-1] + "v" in by_task
            })
    steps = [(_max_step_tool_calls(trace), trace["stop_reason"])
             for by_task in data.values() for traces in by_task.values() for trace in traces]
    others = sorted(most for most, stop in steps if stop != "tool_budget")
    result["tool_budget"] = {
        "budget_per_step": 150,
        "tool_budget_stops": sum(stop == "tool_budget" for _, stop in steps),
        "other_trials_over_budget_in_a_step": sum(most > 150 for most in others),
        "other_trials_max_step_calls": {
            "p50": others[len(others) // 2], "p99": others[int(len(others) * 0.99)],
            "p999": others[int(len(others) * 0.999)], "max": others[-1],
        },
    }
    result["primary_panel_families"] = {
        "total": len({family_of[t] for t in pooled}),
        "nvr": len({family_of[t] for t in pooled if nvr_family[family_of[t]]}),
    }

    # Robustness and descriptive measures; exploratory throughout.
    for agent in agents:
        others = [other for other in agents if other != agent]
        estimate(exploratory, f"exploratory/interface/without_{agent}/pass^1", averaged(per_agent, others))
    no_caps = {agent: interface_effects(agent, 1, lambda trace: trace["stop_reason"] != "wall_cap") for agent in agents}
    estimate(exploratory, "exploratory/interface/model_averaged/pass^1/excluding_wall_caps", averaged(no_caps, agents))

    def failure_kind(task_id: str, trace: dict) -> str | None:
        evaluation = trace["evaluation"]
        if evaluation["primary_reward"] >= 1 - 1e-9:
            return None
        stop, nvr = trace["stop_reason"], nvr_family[family_of[task_id]]
        if stop == "recommended":
            if nvr:
                return "nvr_recommended"
            return "wrong_item" if evaluation["constraint_score"] < 1 - 1e-9 else "policy_only"
        if stop == "abstained":
            return "policy_only" if nvr else "false_abstention"
        return "no_recommendation"

    kinds = Counter(
        kind for by_task in primary.values() for task, traces in by_task.items() for trace in traces
        if (kind := failure_kind(task, trace))
    )
    result["primary_failure_kinds"] = {
        "counts": dict(kinds), "shares": {kind: count / sum(kinds.values()) for kind, count in kinds.items()},
    }
    continued = Counter()
    for by_task in data.values():
        for traces in by_task.values():
            for trace in traces:
                if _max_step_tool_calls(trace) > 150 and trace["stop_reason"] != "tool_budget":
                    continued[trace["stop_reason"]] += 1
                    continued["succeeded"] += trace["evaluation"]["primary_reward"] >= 1 - 1e-9
    result["tool_budget"]["continued_after_budget_error"] = dict(continued)

    def outcome(trace: dict) -> str:
        if trace["evaluation"]["primary_reward"] >= 1 - 1e-9:
            return "success"
        stop = trace["stop_reason"]
        if stop == "recommended":
            return "wrong_item"
        if stop == "abstained":
            return "abstained_after_budget_error" if _max_step_tool_calls(trace) > 150 else "abstained"
        return "overflow_or_budget_stop" if stop in {"context_overflow", "tool_budget"} else stop

    result["scale_outcomes"] = {
        f"{agent}/{arm}/{size}": dict(Counter(outcome(trace) for t in scale[(agent, arm, size)].values() for trace in t))
        for agent in scale_agents for arm in ARMS for size in (5000, 20000)
    }
    pooled_hidden = {t: v for t, v in pooled.items() if t.endswith("_h") and not nvr_family[family_of[t]]}
    for label, has_history in (("history", True), ("no_history", False)):
        subset = {t: v for t, v in pooled_hidden.items() if bool(answers[t]["read_ids"]) == has_history}
        level(f"descriptive/hidden/satisfiable/{label}/pass^1", subset)
    recommits = {}
    for user, by_task in users.items():
        flags = []
        for traces in by_task.values():
            for trace in traces:
                rejected = {decision.get("candidate_id") for decision in trace.get("simulator_decisions", [])
                            if decision.get("action") == "candidate_verdict" and decision.get("verdict") == "rejected"}
                final = (trace.get("recommendations") or [None])[-1]
                flags.append(final is not None and final in rejected)
        recommits[user] = _mean(flags)
    result["recommitted_to_rejected_candidate"] = recommits
    # Interface effect split by whether the family constrains publication year (keyword search
    # does not index the year field), and by condition without Qwen.
    has_year = {row["family_id"]: "publication_year:" in row["constraint_signature"] for row in families}
    for label, wanted in (("year_predicate", True), ("no_year_predicate", False)):
        estimate(exploratory, f"exploratory/interface/model_averaged/{label}/pass^1",
                 {t: v for t, v in overall.items() if has_year[family_of[t]] == wanted})
    without_qwen = averaged(per_agent, [a for a in agents if a != "qwen38_27b_high"])
    for letter, reveal in REVEALS.items():
        estimate(exploratory, f"exploratory/interface/without_qwen38_27b_high/{reveal}/pass^1",
                 {t: v for t, v in without_qwen.items() if t.endswith(f"_{letter}")})
    # What GPT-6.1 Sol asked in mixed trials that elicited nothing (controller decisions).
    nothing = Counter()
    for arm in ARMS:
        for task, traces in primary[("gpt61_sol_high", arm)].items():
            if not task.endswith("_m"):
                continue
            for trace in traces:
                if _decided_on_ask(trace, tasks[task])[1]:
                    continue
                fields = {f for d in trace["simulator_decisions"] if d.get("intent") == "field"
                          for f in d.get("requested_fields", [])}
                nothing["formats" if "formats" in fields else "genre_only" if fields == {"genres"}
                        else "other_fields" if fields else "no_field_question"] += 1
    result["gpt_mixed_nothing_elicited"] = {"counts": dict(nothing), "trials": sum(nothing.values())}
    # Users asked about a field: did they state its askable value? (recognizer applied to agent turns)
    from tau_rec.simulator.user_simulator import classify_book_question
    conditional = {}
    for user, by_task in users.items():
        asked_stated = asked_total = never_asked = facts = 0
        for task_id, traces in by_task.items():
            if not task_id.endswith("_m"):
                continue
            askable = [t["constraint"] for t in tasks[task_id]["constraints"] if t["reveal"] == "on_ask"]
            for trace in traces:
                asked_fields = {
                    field for event in trace["events"]
                    if event.get("type") == "message" and event["role"] == "agent"
                    for field in classify_book_question(event["content"]).fields
                }
                text = " ".join(_user_messages(trace))
                for constraint in askable:
                    facts += 1
                    if constraint["field"] in asked_fields:
                        asked_total += 1
                        asked_stated += _stated(text, constraint["value"], tolerant=True)
                    else:
                        never_asked += 1
        conditional[user] = {
            "stated_when_asked": asked_stated / max(1, asked_total),
            "never_asked_share": never_asked / max(1, facts),
            "askable_facts": facts,
        }
    result["simulator_conditional_disclosure"] = conditional
    for agent in agents:
        for arm in ARMS:
            block = reliability[(agent, arm)]
            mixed_block = [t for task, t in block.items() if task.endswith("_m")]
            result["primary_table"][f"{agent}/{arm}"]["reliability/mixed_all_or_nothing_share"] = _mean(
                sum(trace["evaluation"]["primary_reward"] >= 1 - 1e-9 for trace in t) in {0, len(t)}
                for t in mixed_block)
    # Effort: search calls per trial, by scale cell and by primary-panel cell.
    for key, by_task in scale.items():
        counts = sorted(_searches(trace) for t in by_task.values() for trace in t)
        result["scale_curves"]["/".join(map(str, key))]["search_calls"] = {
            "mean": _mean(counts), "median": counts[len(counts) // 2], "p90": counts[int(len(counts) * 0.9)]}
    for (agent, arm), by_task in primary.items():
        result["primary_table"][f"{agent}/{arm}"]["search_calls_per_trial"] = _mean(
            _searches(trace) for t in by_task.values() for trace in t)
    # Final recommendations whose identifier no search result showed earlier in the trial.
    unseen = Counter()
    for by_task in data.values():
        for traces in by_task.values():
            for trace in traces:
                final = (trace.get("recommendations") or [None])[-1]
                if final is None:
                    continue
                shown = {item for event in trace["events"] if event.get("type") == "tool_call"
                         and event["name"] == "search_catalog" for item in ID_PATTERN.findall(event.get("result") or "")}
                unseen["recommendations"] += 1
                if final not in shown:
                    unseen["unseen"] += 1
                    unseen["unseen_succeeded"] += trace["evaluation"]["primary_reward"] >= 1 - 1e-9
    result["recommendations_never_searched"] = dict(unseen)
    # Qwen's interface effect where its two arms ran together (remainder) and on the reliability block.
    qwen = per_agent["qwen38_27b_high"]
    block_tasks = set(reliability[("qwen38_27b_high", "keyword")])
    for label, inside in (("reliability_block", True), ("primary_remainder", False)):
        estimate(exploratory, f"exploratory/interface/qwen38_27b_high/{label}/pass^1",
                 {t: v for t, v in qwen.items() if (t in block_tasks) == inside})
    # When each cell ran: first and last agent call.
    spans = {}
    for key, by_task in data.items():
        starts = sorted(call["started_at"] for traces in by_task.values() for trace in traces
                        for call in (trace.get("api_calls") or {}).get("agent", []) if call.get("started_at"))
        if starts:
            spans["/".join(str(part) for part in key if part is not None)] = [starts[0], starts[-1]]
    result["cell_time_spans_utc"] = dict(sorted(spans.items()))
    # Share of each cell's trials whose first agent call fell on the run's second UTC day.
    first_day = min(span[0] for span in spans.values())[:10]
    result["second_day_share"] = {
        "/".join(str(part) for part in key if part is not None): _mean(
            min(call["started_at"] for call in trace["api_calls"]["agent"])[:10] > first_day
            for traces in by_task.values() for trace in traces if (trace.get("api_calls") or {}).get("agent"))
        for key, by_task in sorted(data.items(), key=lambda item: str(item[0]))
    }
    if args.scripted:
        result["hidden_density_terciles"] = _density_terciles(
            args.artifact, args.scripted, sorted(pooled_hidden), pooled_hidden, answers)
        scripted_rows = {row["task_id"]: row for row in json.loads(args.scripted.read_text())["rows"]}
        hidden_panel = sorted(t for t in pooled if t.endswith("_h"))
        failed = [t for t in hidden_panel if not scripted_rows[t]["success"]]
        result["scripted_hidden_failures"] = {
            "panel_hidden_tasks": len(hidden_panel),
            "scripted_failed": len(failed),
            "solved_by_an_agent_at_least_once": sum(
                any(trace["evaluation"]["primary_reward"] >= 1 - 1e-9
                    for by_task in primary.values() for trace in by_task.get(t, [])) for t in failed),
        }
        result["scripted_hidden_history"] = {
            label: _mean(scripted_rows[t]["success"] for t in pooled_hidden
                         if bool(answers[t]["read_ids"]) == has_history)
            for label, has_history in (("history", True), ("no_history", False))
        }

    result["confirmatory"] = confirmatory
    result["exploratory"] = exploratory
    result["sensitivities"] = sensitivities
    result["levels"] = levels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    def show(name: str, row: dict, p: str) -> None:
        low, high = row["family_ci95"]
        print(f"  {name:70} {row['estimate']:+.3f} [{low:+.3f}, {high:+.3f}] {p}")

    show(primary_name, confirmatory[primary_name], f"p={confirmatory[primary_name]['family_p']:.4f}")
    for name in ordered:
        show(name, confirmatory[name], f"holm p={confirmatory[name]['holm_p']:.4f}")
    for name, row in sensitivities.items():
        show(name, row, f"p={row['family_p']:.4f}")
    print(f"provider-failure exclusions: {len(excluded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
