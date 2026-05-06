"""Manual play mode: you act as the agent, the user is a simulated LLM.

Usage:
    uv run tau-rec play \\
        --task data/tasks/task_017.json \\
        --catalog data/catalog.json \\
        --simulator-model openai/gpt-4o-mini

At each turn you choose what to do:
    [m] send a text message to the user
    [s] call search_catalog(query)
    [g] call get_metadata(item_id)
    [a] call check_availability(item_id, services)
    [h] call get_user_history(user_id)
    [r] call recommend(item_id)   — ends the trial
    [x] abstain via recommend(null) — ends the trial
    [q] quit without evaluating

The trial is scored via the same CombinedEvaluator used in automated runs.
"""

from __future__ import annotations
import asyncio
import json
from pathlib import Path

import click

from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.data_model.conversation import (
    ConversationTrace, Message, Role, ToolCall, StopReason,
)
from tau_rec.catalog.search import CatalogSearch
from tau_rec.environment.tools import ToolKit
from tau_rec.simulator.user_simulator import UserSimulator
from tau_rec.orchestrator.orchestrator import OPENING_GREETING
from tau_rec.evaluator.evaluator import CombinedEvaluator


def _print_header(title: str) -> None:
    click.echo()
    click.secho(f"─── {title} " + "─" * (60 - len(title)), fg="cyan")


def _print_task_brief(task: Task) -> None:
    _print_header(f"Task {task.id}")
    click.echo(f"  complexity:        {task.complexity}")
    click.echo(f"  reveal_difficulty: {task.reveal_difficulty}")
    click.echo(f"  no_valid_rec:      {task.no_valid_recommendation}")
    click.echo(f"  policy_flags:      {task.policy_flags}")
    click.echo(f"  user_id:           {task.user_id}")
    click.echo("  persona:           (hidden from you — you see what the user tells you)")
    click.echo("  constraints:       (hidden from you — the scorer knows them)")


def _print_tool_help() -> None:
    click.echo()
    click.secho("Choose an action:", fg="yellow")
    click.echo("  [m] send a text message to the user")
    click.echo("  [s] search_catalog(query)")
    click.echo("  [g] get_metadata(item_id)")
    click.echo("  [a] check_availability(item_id, services)")
    click.echo("  [h] get_user_history(user_id)")
    click.echo("  [r] recommend(item_id)     — ends trial")
    click.echo("  [x] abstain (recommend())  — ends trial")
    click.echo("  [q] quit without evaluating")


def _pretty_print_result(result: str) -> None:
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        click.echo(result)
        return
    if isinstance(parsed, list):
        for i, row in enumerate(parsed):
            click.echo(f"  [{i}] {json.dumps(row)}")
    else:
        click.echo(json.dumps(parsed, indent=2))


async def _play(
    task_path: str,
    catalog_path: str,
    simulator_model: str,
) -> None:
    task = Task.from_json(task_path)
    catalog = Catalog.from_json(catalog_path)
    search = CatalogSearch(catalog)
    toolkit = ToolKit(
        catalog=catalog,
        search=search,
        user_histories=task.user_history or {},
    )
    simulator = UserSimulator(model=simulator_model, task=task)
    evaluator = CombinedEvaluator(catalog)

    trace = ConversationTrace(task_id=task.id, model="human", trial=0)
    conversation: list[Message] = []

    _print_task_brief(task)

    # Seed greeting (same as orchestrator)
    greeting = Message(role=Role.AGENT, content=OPENING_GREETING)
    trace.add_message(greeting)
    conversation.append(greeting)
    _print_header("Agent (seeded greeting)")
    click.echo(greeting.content)

    # User's first turn
    _print_header("User")
    user_msg = await simulator.respond(conversation)
    trace.add_message(user_msg)
    conversation.append(user_msg)
    click.echo(user_msg.content)

    # Main loop — you drive
    while True:
        _print_tool_help()
        choice = click.prompt(">", type=str, default="m", show_default=False).strip().lower()

        if choice == "q":
            click.secho("Exiting without scoring.", fg="red")
            return

        if choice == "m":
            text = click.prompt("Your message", type=str)
            agent_msg = Message(role=Role.AGENT, content=text)
            trace.add_message(agent_msg)
            conversation.append(agent_msg)
            _print_header("User")
            user_msg = await simulator.respond(conversation)
            trace.add_message(user_msg)
            conversation.append(user_msg)
            click.echo(user_msg.content)
            continue

        if choice == "s":
            q = click.prompt("  query", type=str)
            result = toolkit.call("search_catalog", {"query": q})
            trace.add_tool_call(ToolCall(name="search_catalog", arguments={"query": q}, result=result))
            _print_header("search_catalog result")
            _pretty_print_result(result)
            continue

        if choice == "g":
            iid = click.prompt("  item_id", type=str)
            result = toolkit.call("get_metadata", {"item_id": iid})
            trace.add_tool_call(ToolCall(name="get_metadata", arguments={"item_id": iid}, result=result))
            _print_header("get_metadata result")
            _pretty_print_result(result)
            continue

        if choice == "a":
            iid = click.prompt("  item_id", type=str)
            svcs_raw = click.prompt("  services (comma-separated)", type=str)
            services = [s.strip() for s in svcs_raw.split(",") if s.strip()]
            result = toolkit.call("check_availability", {"item_id": iid, "services": services})
            trace.add_tool_call(ToolCall(
                name="check_availability",
                arguments={"item_id": iid, "services": services},
                result=result,
            ))
            _print_header("check_availability result")
            _pretty_print_result(result)
            continue

        if choice == "h":
            uid = click.prompt("  user_id", type=str, default=task.user_id)
            result = toolkit.call("get_user_history", {"user_id": uid})
            trace.add_tool_call(ToolCall(name="get_user_history", arguments={"user_id": uid}, result=result))
            _print_header("get_user_history result")
            _pretty_print_result(result)
            continue

        if choice == "r":
            iid = click.prompt("  item_id to recommend", type=str)
            result = toolkit.call("recommend", {"item_id": iid})
            trace.add_tool_call(ToolCall(name="recommend", arguments={"item_id": iid}, result=result))
            trace.add_recommendation(iid)
            _print_header("recommend result")
            _pretty_print_result(result)
            trace.stop_reason = StopReason.RECOMMENDED
            break

        if choice == "x":
            result = toolkit.call("recommend", {})
            trace.add_tool_call(ToolCall(name="recommend", arguments={}, result=result))
            _print_header("recommend(None) — abstained")
            _pretty_print_result(result)
            trace.stop_reason = StopReason.ABSTAINED
            break

        click.secho(f"Unknown choice: {choice!r}", fg="red")

    # Score
    result = evaluator.evaluate(task, trace)
    _print_header("Scoring")
    click.echo(f"  constraint_score = {result.constraint_score}")
    click.echo(f"  policy_score     = {result.policy_score}")
    click.secho(f"  primary_reward   = {result.primary_reward}",
                fg="green" if result.primary_reward >= 1.0 else "red", bold=True)
    if result.policy_detail.violations:
        click.echo(f"  policy violations: {result.policy_detail.violations}")
    # Show per-constraint detail (useful for learning what you missed)
    if result.constraint_detail.per_constraint:
        click.echo()
        click.secho("  per-constraint breakdown:", fg="cyan")
        for pc in result.constraint_detail.per_constraint:
            mark = "✓" if pc.satisfied else "✗"
            click.echo(f"    {mark} {pc.field} {pc.op}")


@click.command()
@click.option("--task", "task_path", required=True, type=click.Path(exists=True),
              help="Path to a single task JSON file")
@click.option("--catalog", "catalog_path", required=True, type=click.Path(exists=True),
              help="Path to catalog JSON")
@click.option("--simulator-model", default="openai/gpt-4o-mini",
              help="Model for the simulated user")
def play(task_path: str, catalog_path: str, simulator_model: str) -> None:
    """Manual play mode — you act as the agent against a simulated user."""
    asyncio.run(_play(task_path, catalog_path, simulator_model))
