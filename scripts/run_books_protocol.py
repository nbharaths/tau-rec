#!/usr/bin/env python3
"""Run one agent under the reportable books-wikidata-v2 protocol.

The protocol is the one behind the reported results: the primary panel (51
families, 153 tasks) or its nested reliability block (18 families, 54 tasks),
the 20,000-work catalog, the rule-based user, 20 agent turns, and 600 s of agent
time per trial (the track's harness profile counts agent time only). Four
trials per task on the primary panel and eight on the reliability block match
the reported results. ``--agent-prompt-variant ask-and-verify-v1`` gives the
recommended second prompt condition.

Without ``--execute`` the script prints the ``tau-rec run`` command and exits;
with it, the command runs and needs ``--max-provider-cost-usd``.

    uv run python scripts/run_books_protocol.py --model openrouter/deepseek/deepseek-v4.1-flash \\
        --retrieval-mode structured --reasoning-effort high
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACK = "data/tracks/books-wikidata-v2"
PANELS = ROOT / "analysis/books-wikidata-v2/panels.json"
DEFAULT_TRIALS = {"primary": 4, "reliability": 8}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="agent model (litellm name)")
    parser.add_argument("--retrieval-mode", required=True, choices=("keyword", "structured"))
    parser.add_argument("--block", choices=tuple(DEFAULT_TRIALS), default="primary")
    parser.add_argument("--trials", type=int, help="trials per task (default: 4 primary, 8 reliability)")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--provider-order", help="comma-separated OpenRouter upstreams to pin")
    parser.add_argument("--agent-prompt-variant", help="opt-in prompt variant, e.g. ask-and-verify-v1")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-provider-cost-usd", type=float)
    parser.add_argument("--output", default="out/books-reportable")
    parser.add_argument("--execute", action="store_true", help="run the command instead of printing it")
    return parser.parse_args(argv)


def build_command(args: argparse.Namespace) -> list[str]:
    task_ids = json.loads(PANELS.read_text())["panels"][args.block]["task_ids"]
    command = [
        "uv", "run", "tau-rec", "run",
        "--track", TRACK,
        "--model", args.model,
        "--simulator-mode", "symbolic",
        "--retrieval-mode", args.retrieval_mode,
        "--catalog-size", "20000",
        "--tasks-filter", ",".join(task_ids),
        "--trials", str(args.trials or DEFAULT_TRIALS[args.block]),
        "--max-turns", "20",
        "--trial-timeout-seconds", "600",
        "--concurrency", str(args.concurrency),
        "--output", args.output,
    ]
    if args.reasoning_effort:
        command += ["--reasoning-effort", args.reasoning_effort]
    if args.provider_order:
        command += ["--provider-order", args.provider_order]
    if args.agent_prompt_variant:
        command += ["--agent-prompt-variant", args.agent_prompt_variant]
    if args.max_provider_cost_usd is not None:
        command += ["--max-provider-cost-usd", str(args.max_provider_cost_usd)]
    return command


def launch(command: list[str]) -> int:
    # The launch point: no test may start provider calls.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise SystemExit("refusing to launch provider calls from a test")
    return subprocess.run(command, cwd=ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    if not args.execute:
        print(shlex.join(command))
        return 0
    if args.max_provider_cost_usd is None:
        raise SystemExit("--execute needs --max-provider-cost-usd")
    return launch(command)


if __name__ == "__main__":
    raise SystemExit(main())
