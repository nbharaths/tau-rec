#!/usr/bin/env python3
"""Freeze the v4 books matrix: stages, model identities, pins, harness and analysis.

v4 runs once, on one commit, on the books-wikidata-v2 track and the
books-v4 harness. The stages: the 20k primary panel in both
retrieval arms for every agent (K=4, with a nested K=8 reliability block), the
scale study on every satisfiable family (volunteer and mixed, 5k/10k/20k), and
two LLM users on the reliability block. The plan records the track, the
harness versions, the model upstreams, and an analysis section fixed before
any trial runs (the confirmatory family, multiplicity, stop classes and
exclusion rules).

``--draft`` writes the plan without an accepted hard ceiling; a frozen plan
needs the authors' ceiling in the model config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tau_rec.leaderboard.hashing import hash_task_dir

VERSION = "books-wikidata-experiment-plan-v4"
SIZES = [5_000, 10_000, 20_000]
BOTH = ["keyword", "structured"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(
    name: str, task_ids: list[str], *, role: str, agents: list[str], catalog_sizes: list[int],
    retrieval_modes: list[str], trials_per_task: int, simulator_mode: str = "symbolic",
    simulators: list[str] | None = None,
) -> dict:
    simulator_slots = len(simulators) if simulators else 1
    return {
        "name": name,
        "role": role,
        "task_ids": task_ids,
        "tasks_filter": ",".join(task_ids),
        "task_count": len(task_ids),
        "agent_labels": agents,
        "simulator_mode": simulator_mode,
        "simulator_labels": simulators or [],
        "catalog_sizes": catalog_sizes,
        "retrieval_modes": retrieval_modes,
        "trials_per_task": trials_per_task,
        "planned_trials": (
            len(task_ids) * len(agents) * simulator_slots * len(catalog_sizes)
            * len(retrieval_modes) * trials_per_task
        ),
    }


def _validate_config(config: dict, *, draft: bool) -> None:
    agents = config["agent_models"]
    labels = [row["label"] for row in agents]
    if len(set(labels)) != len(labels) or len({row["model"] for row in agents}) != len(agents):
        raise SystemExit("agent labels and model IDs must be distinct")
    if not set(config["scale_agent_labels"]) <= set(labels):
        raise SystemExit("scale agents must be configured agents")
    if config["sensitivity_agent_label"] not in labels:
        raise SystemExit("sensitivity agent must be a configured agent")
    simulators = config["simulator_models"]
    if len({row["provider_family"] for row in simulators}) != len(simulators):
        raise SystemExit("LLM simulators must come from distinct provider families")
    for row in agents + simulators:
        if row["gateway"] == "openrouter" and not row.get("provider_order"):
            raise SystemExit(f"{row['label']}: OpenRouter models must pin an upstream")
    if not draft and not isinstance(config["budgets_usd"].get("v4_hard_ceiling"), (int, float)):
        raise SystemExit("model config needs the authors' v4 hard ceiling (or pass --draft)")


ANALYSIS = {
    "sampling_unit": "task family",
    "trial_role": "repeated measurement, not an independent task sample",
    "pass_k": [1, 2, 4, 8],
    "confidence_interval": "seeded paired bootstrap over task families",
    "bootstrap_seed": 20261014,
    "bootstrap_iterations": 10_000,
    "bootstrap_p_value": "(k + 1) / (B + 1), two-sided",
    "primary_contrast": (
        "model-averaged 20k structured-minus-keyword pass^1 on the primary panel, all three "
        "disclosure conditions; tested alone at alpha 0.05"
    ),
    "secondary_family": {
        "multiplicity": "Holm over exactly these contrasts at familywise alpha 0.05",
        "contrasts": [
            "structured-minus-keyword pass^1 at 20k on the primary panel, per agent (4)",
            "mixed-minus-volunteer pass^1 at 20k on the primary panel, model-averaged over arms",
            "hidden-minus-volunteer pass^1 at 20k on the primary panel, satisfiable families, model-averaged over arms",
            "5k-minus-20k pass^1 in the scale study, keyword arm, scale agents averaged",
            "5k-minus-20k pass^1 in the scale study, structured arm, scale agents averaged",
            "interface x scale interaction: the structured arm's 5k-minus-20k drop minus the keyword arm's",
            "LLM-user-minus-symbolic pass^1 on the reliability block, structured 20k, per LLM user (2)",
        ],
        "holm_floor_note": "with B = 10,000 and 11 contrasts the smallest attainable adjusted p is 11/10,001",
    },
    "exploratory": "every other contrast; reported with unadjusted intervals and never as confirmatory",
    "stop_classes": {
        "classes": ["recommended", "abstained", "wall_cap", "turn_limit", "context_overflow", "tool_budget"],
        "scoring": "as scored: a trial that stops without recommend() has no recommendation",
        "sensitivity": "the primary contrast and per-agent interface contrasts are repeated excluding context_overflow and tool_budget trials",
    },
    "exclusions": {
        "infrastructure_stalls": "the stall rule below: quarantined, re-run once, kept and counted",
        "provider_failures": (
            "a family is excluded for one agent or LLM user only when that model's provider fails "
            "every trial of the family (blank completions or refusals after the harness's retries), "
            "identified from api_calls before outcomes are analysed; exclusions are listed with the "
            "results and apply to every contrast that uses that model"
        ),
        "no_other_exclusions": True,
    },
    "descriptive": [
        "hidden satisfiable and hidden no-valid-answer families reported separately",
        "elicitation measured from controller decisions, and for LLM users with a paraphrase-tolerant disclosure measure",
        "stop-class distribution per cell",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()

    import litellm
    from tau_rec.agents.litellm_agent import request_parameter_manifest
    from tau_rec.catalog.books_tasks import REVEAL_GUARD_VERSION
    from tau_rec.domain import BOOKS
    from tau_rec.harness import harness_commit
    from tau_rec.simulator.user_simulator import (
        books_user_versions,
        request_parameter_manifest as simulator_parameter_manifest,
    )
    from tau_rec.track import load_track

    track = load_track(args.artifact)
    family_manifest_path = args.artifact / "family_manifest.json"
    task_manifest_path = args.artifact / "task_manifest.json"
    family_manifest = json.loads(family_manifest_path.read_text())
    task_manifest = json.loads(task_manifest_path.read_text())
    panels_payload = json.loads(args.panels.read_text())
    panels = panels_payload["panels"]
    if panels_payload.get("family_manifest_sha256") != _sha256(family_manifest_path):
        raise SystemExit("panels do not belong to the frozen family manifest")
    if task_manifest.get("reveal_guard_version") != REVEAL_GUARD_VERSION:
        raise SystemExit("artifact does not declare the current reveal guard")
    if track.harness_profile != "books-v4":
        raise SystemExit("the v4 matrix runs only on a track declaring the books-v4 harness")
    config = json.loads(args.model_config.read_text())
    _validate_config(config, draft=args.draft)

    satisfiable = sorted(
        (row for row in family_manifest["families"] if not row["no_valid_recommendation"]),
        key=lambda row: row["family_id"],
    )
    scale_tasks = [row["task_ids"][reveal] for row in satisfiable for reveal in ("volunteer", "mixed")]
    agents = [row["label"] for row in config["agent_models"]]
    stages = [
        _stage("smoke", panels["smoke"]["task_ids"], role="interface, pin, and cost gate; excluded from estimates",
               agents=agents, catalog_sizes=[20_000], retrieval_modes=BOTH, trials_per_task=1),
        _stage("smoke_llm_user", panels["smoke"]["task_ids"],
               role="the two LLM users on the smoke tasks: prompt and pin gate; excluded from estimates",
               agents=[config["sensitivity_agent_label"]], catalog_sizes=[20_000],
               retrieval_modes=["structured"], trials_per_task=1, simulator_mode="llm",
               simulators=[row["label"] for row in config["simulator_models"]]),
        _stage("primary_remainder", panels["primary_remainder"]["task_ids"],
               role="K=4 primary block outside the nested reliability families",
               agents=agents, catalog_sizes=[20_000], retrieval_modes=BOTH, trials_per_task=4),
        _stage("reliability", panels["reliability"]["task_ids"], role="nested primary block run at K=8",
               agents=agents, catalog_sizes=[20_000], retrieval_modes=BOTH, trials_per_task=8),
        _stage("scale_all", scale_tasks,
               role="every satisfiable family, volunteer and mixed, at 5k/10k/20k, sizes and arms run together",
               agents=config["scale_agent_labels"], catalog_sizes=SIZES, retrieval_modes=BOTH,
               trials_per_task=4),
        _stage("simulator_sensitivity", panels["reliability"]["task_ids"],
               role="two LLM users; the symbolic comparison cell comes from the reliability stage",
               agents=[config["sensitivity_agent_label"]], catalog_sizes=[20_000],
               retrieval_modes=["structured"], trials_per_task=4, simulator_mode="llm",
               simulators=[row["label"] for row in config["simulator_models"]]),
    ]
    smoke = sum(stage["planned_trials"] for stage in stages if stage["name"].startswith("smoke"))
    full = sum(stage["planned_trials"] for stage in stages if not stage["name"].startswith("smoke"))
    policy_flags = sorted({
        flag for path in (args.artifact / "tasks").glob("*.json")
        for flag in json.loads(path.read_text())["policy_flags"]
    })
    payload = {
        "version": VERSION,
        "status": "draft-not-frozen" if args.draft else "frozen-pending-smoke",
        "artifact": {
            "track_id": track.id,
            "catalog_sha256": _sha256(args.artifact / "catalog.json"),
            "tasks_sha256": hash_task_dir(args.artifact / "tasks"),
            "task_manifest_sha256": _sha256(task_manifest_path),
            "family_manifest_sha256": _sha256(family_manifest_path),
            "policy_sha256": _sha256(args.artifact / "policy.md"),
            "panels_sha256": _sha256(args.panels),
            "model_config_sha256": _sha256(args.model_config),
        },
        "harness": {
            "commit_at_freeze": harness_commit(),
            "profile": track.harness.manifest(),
            "symbolic_user": books_user_versions("symbolic"),
            "llm_user": books_user_versions("llm"),
            "reveal_guard": REVEAL_GUARD_VERSION,
            "scored_policy_flags": policy_flags,
            "structured_profile": track.structured_profile,
            "search_profile": track.search_profile,
            "taxonomy": task_manifest.get("taxonomy_version"),
            "litellm_version": getattr(litellm, "__version__", None) or _litellm_version(),
            "litellm_local_model_cost_map": config["litellm"]["local_model_cost_map"],
        },
        "models": {
            "agents": config["agent_models"],
            "llm_simulators": config["simulator_models"],
            "scale_agents": config["scale_agent_labels"],
            "sensitivity_agent": config["sensitivity_agent_label"],
        },
        "api_parameters": {
            "agents": {
                row["label"]: request_parameter_manifest(
                    row["model"], row.get("reasoning_effort"), row.get("provider_order"), track.harness,
                )
                for row in config["agent_models"]
            },
            "symbolic_simulator": simulator_parameter_manifest(BOOKS, interaction_mode="symbolic"),
            "llm_simulators": {
                row["label"]: simulator_parameter_manifest(BOOKS, provider_order=row.get("provider_order"))
                for row in config["simulator_models"]
            },
        },
        "fixed_runtime": config["fixed_runtime"],
        "stall_retry_rule": config["stall_retry_rule"],
        "budgets_usd": config["budgets_usd"],
        "stages": stages,
        "trial_accounting": {
            "smoke_trials": smoke,
            "full_matrix_trials_excluding_smoke": full,
            "all_planned_trials": smoke + full,
        },
        "analysis": ANALYSIS,
        "launch_rules": {
            "one_commit": "every v4 trial runs on the commit recorded at freeze; the launcher refuses a dirty tree",
            "smoke": (
                "run every agent once per smoke task and arm, and both LLM users once per smoke task; review "
                "the traces by hand; re-project cost"
            ),
            "full_matrix": "requires a passing smoke and the authors' v4 hard ceiling",
            "time_matching": "all cells of all stages run concurrently, so no contrast spans different hours",
            "outcome_blind": "do not change panels, models, contrasts or exclusions in response to smoke outcomes",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {VERSION} ({payload['status']}): {smoke} smoke + {full:,} full-matrix trials")
    for stage in stages:
        print(f"  {stage['name']:22} {stage['planned_trials']:6,} trials")
    return 0


def _litellm_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("litellm")
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
