#!/usr/bin/env python3
"""Per-agent success under the base, ask-requirements, ask-and-verify and protocol prompts on the reliability block.

Collects the four agents' structured 20k reliability cells under the base
prompt (v4 matrix), under the ask-requirements prompt (the follow-up that
ran GPT-6.1 Sol and DeepSeek, and the one that ran GLM and Qwen) and, with
--verify-root, under the ask-and-verify prompt and, with --protocol-root, under
ask-and-verify plus what declines and rejections mean. Reports pass^1 on volunteer
and mixed tasks, hidden success on satisfiable tasks with and without a cap
of three rejected proposals, hidden success on NVR tasks, and the share of
satisfiable hidden failures whose final book was never proposed. Levels are
task-averaged and descriptive; verify minus ask-requirements on capped hidden
success, and protocol minus verify on uncapped hidden satisfiable success (per
agent and averaged over agents), get a paired family bootstrap (B = 10,000,
seed 20261014), and every cell's uncapped hidden satisfiable success gets a
family bootstrap interval.  Post hoc, except the protocol contrasts its frozen plan names.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


addendum = _module("analyze_v4_addendum")
posthoc = _module("posthoc_v4")
failures = _module("hidden_failures")
v4 = addendum.v4

AGENTS = ("deepseek_v41_flash_high", "glm53_flash_high", "qwen38_27b_high", "gpt61_sol_high")
FIRST_FOLLOWUP = ("gpt61_sol_high", "deepseek_v41_flash_high")


def cell_summary(by_task: dict[str, list[dict]], nvr: dict[str, bool]) -> dict[str, float]:
    def mean_over(tasks: list[str], value) -> float | None:
        mean = v4._mean(value(by_task[t]) for t in tasks)
        return None if mean is None else round(mean, 4)

    def untested_share(traces: list[dict]) -> float | None:
        kinds = [failures.classify(trace) for trace in traces]
        failed = [kind for kind in kinds if kind != "success"]
        return round(sum(kind == "untested" for kind in failed) / len(failed), 4) if failed else None

    def tasks(letter: str, status: bool | None = None) -> list[str]:
        return sorted(t for t in by_task if t.endswith(f"_{letter}") and (status is None or nvr[t] == status))

    return {
        "all": mean_over(sorted(by_task), lambda traces: v4._pass(traces, 1)),
        "volunteer": mean_over(tasks("v"), lambda traces: v4._pass(traces, 1)),
        "mixed": mean_over(tasks("m"), lambda traces: v4._pass(traces, 1)),
        "hidden_satisfiable": mean_over(tasks("h", False), lambda traces: posthoc.capped_success(traces, None)),
        "hidden_satisfiable_within_3": mean_over(tasks("h", False), lambda traces: posthoc.capped_success(traces, 3)),
        "hidden_nvr": mean_over(tasks("h", True), lambda traces: v4._pass(traces, 1)),
        "hidden_untested_of_failures": untested_share(
            [trace for t in tasks("h", False) for trace in by_task[t]]),
    }


def kendall_tau(first: dict[str, float], second: dict[str, float]) -> float:
    """Kendall's tau-a between two sets of levels over the same agents (ties count as discordant-free zeros)."""
    agents = sorted(first)
    pairs = [(a, b) for i, a in enumerate(agents) for b in agents[i + 1:]]
    sign = lambda x: (x > 0) - (x < 0)  # noqa: E731
    return sum(sign(first[a] - first[b]) * sign(second[a] - second[b]) for a, b in pairs) / len(pairs)


def _cell(root: Path, stage: str, agent: str) -> dict[str, list[dict]]:
    by_task = addendum._cell(root, stage, agent)
    if not by_task:
        raise SystemExit(f"no traces for {agent} in {root}/{stage}; run from the directory the run paths are "
                         "relative to")
    return by_task


def capped_difference(name: str, first: dict[str, list[dict]], second: dict[str, list[dict]],
                      nvr: dict[str, bool], family_of: dict[str, str], template_of: dict[str, str],
                      k: int | None = 3) -> dict:
    """First minus second on hidden success within k rejections (uncapped if None), satisfiable tasks, by family."""
    tasks = sorted(t for t in first if t.endswith("_h") and not nvr[t])
    values = {t: posthoc.capped_success(first[t], k) - posthoc.capped_success(second[t], k) for t in tasks}
    return v4.interval(name, v4._family_mean(values, family_of), template_of)


def reference_difference(name: str, cell: dict[str, list[dict]], rows: dict[str, dict], nvr: dict[str, bool],
                         family_of: dict[str, str], template_of: dict[str, str]) -> dict:
    """An agent cell minus a deterministic scripted reference on uncapped hidden satisfiable success, by family."""
    tasks = sorted(t for t in cell if t.endswith("_h") and not nvr[t])
    values = {t: posthoc.capped_success(cell[t], None) - float(rows[t]["success"]) for t in tasks}
    return v4.interval(name, v4._family_mean(values, family_of), template_of)


def level_interval(name: str, cell: dict[str, list[dict]], nvr: dict[str, bool], family_of: dict[str, str],
                   template_of: dict[str, str]) -> dict:
    """One cell's uncapped hidden satisfiable success, averaged by family, with its family bootstrap interval."""
    tasks = sorted(t for t in cell if t.endswith("_h") and not nvr[t])
    values = {t: posthoc.capped_success(cell[t], None) for t in tasks}
    return v4.interval(name, v4._family_mean(values, family_of), template_of)


def averaged_difference(name: str, first: dict[str, dict], second: dict[str, dict], nvr: dict[str, bool],
                        family_of: dict[str, str], template_of: dict[str, str]) -> dict:
    """First minus second on uncapped hidden satisfiable success, averaged over agents per task, by family."""
    tasks = sorted(t for t in next(iter(first.values())) if t.endswith("_h") and not nvr[t])
    values = {t: v4._mean(posthoc.capped_success(first[a][t], None) - posthoc.capped_success(second[a][t], None)
                          for a in first) for t in tasks}
    return v4.interval(name, v4._family_mean(values, family_of), template_of)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--addendum-root", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--verify-root", type=Path, help="the ask-and-verify follow-up's output root")
    parser.add_argument("--protocol-root", type=Path, help="the protocol follow-up's output root (needs --verify-root)")
    parser.add_argument("--references", nargs=2, type=Path, metavar=("POPULARITY", "ELIMINATING"),
                        help="run_scripted_reference.py outputs; each prompt cell is compared with both")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.protocol_root and not args.verify_root:
        raise SystemExit("--protocol-root needs --verify-root, the cells it is compared with")
    families = json.loads((args.artifact / "family_manifest.json").read_text())["families"]
    nvr = {task: row["no_valid_recommendation"] for row in families for task in row["task_ids"].values()}
    family_of = {task: row["family_id"] for row in families for task in row["task_ids"].values()}
    template_of = {row["family_id"]: "|".join(sorted({":".join(part.split(":")[:2])
                                                      for part in row["constraint_signature"].split("|")}))
                   for row in families}
    out: dict = {"status": "post hoc; levels descriptive", "agents": {}, "contrasts": {}, "intervals": {}}
    base_cells, verify_cells, protocol_cells = {}, {}, {}
    for agent in AGENTS:
        variant_root = args.addendum_root if agent in FIRST_FOLLOWUP else args.prompt_root
        ask = _cell(variant_root, "prompt_variant", agent)
        base = base_cells[agent] = _cell(args.main_root, "reliability", agent)
        out["agents"][agent] = {"base": cell_summary(base, nvr), "ask_requirements": cell_summary(ask, nvr)}
        for prompt, cell in (("base", base), ("ask_requirements", ask)):
            name = f"{prompt}/{agent}/hidden_satisfiable"
            out["intervals"][name] = level_interval(name, cell, nvr, family_of, template_of)
        if args.verify_root:
            verify = _cell(args.verify_root, "verify_variant", agent)
            out["agents"][agent]["ask_and_verify"] = cell_summary(verify, nvr)
            name = f"verify_minus_ask/{agent}/hidden_satisfiable_within_3"
            out["contrasts"][name] = capped_difference(name, verify, ask, nvr, family_of, template_of)
            verify_cells[agent] = verify
            name = f"ask_and_verify/{agent}/hidden_satisfiable"
            out["intervals"][name] = level_interval(name, verify, nvr, family_of, template_of)
        if args.protocol_root:
            protocol = _cell(args.protocol_root, "protocol_variant", agent)
            out["agents"][agent]["ask_verify_protocol"] = cell_summary(protocol, nvr)
            name = f"protocol_minus_verify/{agent}/hidden_satisfiable"
            out["contrasts"][name] = capped_difference(name, protocol, verify, nvr, family_of, template_of, k=None)
            protocol_cells[agent] = protocol
            name = f"ask_verify_protocol/{agent}/hidden_satisfiable"
            out["intervals"][name] = level_interval(name, protocol, nvr, family_of, template_of)
    prompts = [p for p in ("ask_requirements", "ask_and_verify", "ask_verify_protocol")
               if all(p in row for row in out["agents"].values())]
    out["rank_correlation"] = {
        f"base_vs_{prompt}/all": kendall_tau({a: row["base"]["all"] for a, row in out["agents"].items()},
                                             {a: row[prompt]["all"] for a, row in out["agents"].items()})
        for prompt in prompts}
    if args.references:
        references = {label: {row["task_id"]: row for row in json.loads(path.read_text())["rows"]}
                      for label, path in zip(("popularity", "eliminating"), args.references)}
        cells = {"base": base_cells, "ask_and_verify": verify_cells, "ask_verify_protocol": protocol_cells}
        for prompt, by_agent in cells.items():
            for agent, cell in by_agent.items():
                for label, rows in references.items():
                    name = f"{prompt}_minus_{label}/{agent}/hidden_satisfiable"
                    out["contrasts"][name] = reference_difference(name, cell, rows, nvr, family_of, template_of)
    if protocol_cells:
        name = "protocol_minus_verify/model_averaged/hidden_satisfiable"
        out["contrasts"][name] = averaged_difference(name, protocol_cells, verify_cells, nvr, family_of, template_of)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
