from __future__ import annotations
import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
import click

# Load .env from CWD or repo root before anything that might need API keys.
from dotenv import load_dotenv
load_dotenv()

import litellm
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

@click.group()
def main():
    """τ-Rec: A verifiable benchmark for LLM-based CRS."""
    pass


from tau_rec import __version__
from tau_rec.play import play as _play_cmd
main.add_command(_play_cmd, name="play")

@main.command()
@click.option("--track", default=None, type=click.Path(exists=True, file_okay=False),
              help="Track directory; sets --catalog and --tasks unless given explicitly.")
@click.option("--catalog", default=None, type=click.Path(exists=True), help="Path to catalog JSON")
@click.option("--tasks", default=None, type=click.Path(exists=True), help="Path to tasks directory")
@click.option("--baseline", type=click.Path(), default=None,
              help="Snapshot file path. If it doesn't exist, write the current "
                   "task signatures there. If it exists, diff current vs. snapshot "
                   "and report any drift. Use to guard against silent regressions "
                   "when editing tasks.")
@click.option("--strict", is_flag=True, default=False,
              help="Promote warnings to errors (e.g., near-retrieval, untestable "
                   "policy flags). Use in CI to keep task quality from drifting.")
def validate(track: str | None, catalog: str | None, tasks: str | None, baseline: str | None, strict: bool):
    """Validate tasks against catalog."""
    import json as _json
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import Task
    from tau_rec.catalog.validator import CatalogValidator, task_signature, grid_summary
    from tau_rec.track import load_track, track_for_catalog

    if track:
        trk = load_track(track)
        catalog = catalog or str(trk.require_catalog())
        tasks = tasks or str(trk.tasks_path)
    elif not (catalog and tasks):
        raise click.UsageError("pass --track, or both --catalog and --tasks")
    else:
        trk = track_for_catalog(catalog)
    spec = trk.spec

    cat = Catalog.from_json(catalog, spec)
    validator = CatalogValidator(cat, spec)

    task_dir = Path(tasks)
    task_files = sorted(task_dir.glob("*.json"))
    click.echo(f"Validating {len(task_files)} tasks against catalog ({len(cat)} {spec.item_plural}, track {trk.id})...\n")

    all_valid = True
    total_warnings = 0
    total_errors = 0
    loaded_tasks: list[Task] = []
    signatures: dict[str, dict] = {}
    for tf in task_files:
        task = Task.from_json(str(tf))
        loaded_tasks.append(task)
        result = validator.validate_task(task)
        signatures[task.id] = task_signature(task, result)

        warns = list(result.warnings)
        errs = list(result.errors)
        if strict and warns:
            errs.extend(f"(strict) {w}" for w in warns)
            warns = []

        status = "OK" if result.valid_as_designed and not errs else "FAIL"
        if not result.valid_as_designed or errs:
            all_valid = False
        suffix = ""
        if warns:
            suffix += f"  WARN: {'; '.join(warns)}"
            total_warnings += len(warns)
        if errs:
            suffix += f"  ERR: {'; '.join(errs)}"
            total_errors += len(errs)
        click.echo(f"  {status}  {task.id}  solutions={result.solution_set_size}{suffix}")

    click.echo(f"\n{len(task_files)} tasks checked. {total_errors} errors, {total_warnings} warnings.")

    grid = grid_summary(loaded_tasks)
    click.echo("\nGrid (complexity × reveal_difficulty):")
    click.echo(f"  {'tier':>8} | {'volunteer':>9} | {'mixed':>5} | {'hidden':>6} | total")
    click.echo("  " + "-" * 48)
    for cx in ("simple", "medium", "complex"):
        row = grid[cx]
        total = sum(row.values())
        click.echo(f"  {cx:>8} | {row['volunteer']:>9} | {row['mixed']:>5} | {row['hidden']:>6} | {total:>5}")

    if baseline:
        baseline_path = Path(baseline)
        if not baseline_path.exists():
            from tau_rec.leaderboard.hashing import hash_file
            # _meta records which catalog these signatures were computed against.
            snapshot = dict(signatures)
            snapshot["_meta"] = {"catalog_sha256": hash_file(catalog), "track": trk.id, "tau_rec_version": __version__}
            baseline_path.write_text(_json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
            click.echo(f"\nBaseline snapshot written to {baseline_path}.")
        else:
            prev = _json.loads(baseline_path.read_text())
            prev.pop("_meta", None)
            drift = _diff_signatures(prev, signatures)
            if drift:
                click.secho(f"\nBaseline drift detected ({len(drift)} task(s) changed):", fg="yellow")
                for tid, changes in drift.items():
                    click.echo(f"  {tid}:")
                    for field, (old, new) in changes.items():
                        click.echo(f"    {field}: {old!r} -> {new!r}")
                all_valid = False
            else:
                click.echo(f"\nBaseline OK — task signatures match {baseline_path}.")

    if all_valid:
        click.echo("\nAll tasks valid.")
    else:
        click.echo("\nSome tasks have issues. See above.")
        raise SystemExit(1)


def _diff_signatures(prev: dict, curr: dict) -> dict[str, dict]:
    """Return {task_id: {field: (old, new)}} for any drift."""
    drift: dict[str, dict] = {}
    all_ids = {k for k in set(prev) | set(curr) if not k.startswith("_")}
    for tid in sorted(all_ids):
        if tid not in prev:
            drift[tid] = {"__status__": ("(absent)", "added")}
            continue
        if tid not in curr:
            drift[tid] = {"__status__": ("present", "(removed)")}
            continue
        diffs: dict = {}
        for k in set(prev[tid]) | set(curr[tid]):
            if prev[tid].get(k) != curr[tid].get(k):
                diffs[k] = (prev[tid].get(k), curr[tid].get(k))
        if diffs:
            drift[tid] = diffs
    return drift


@main.command()
@click.option("--model", default="deepseek/deepseek-chat", help="LLM model name (litellm format)")
@click.option("--track", default=None, type=click.Path(exists=True, file_okay=False),
              help="Track directory holding track.json, catalog, tasks/ and policy.md. "
                   "Sets --catalog/--tasks/--policy unless they are given explicitly.")
@click.option("--catalog", default=None, type=click.Path(exists=True), help="Path to catalog JSON")
@click.option("--tasks", default=None, type=click.Path(exists=True), help="Path to tasks directory")
@click.option("--policy", default=None, type=click.Path(exists=True), help="Path to policy.md")
@click.option("--trials", default=16, help="Number of trials per task")
@click.option(
    "--catalog-size",
    default=None,
    type=click.IntRange(min=1),
    help="Use the first N catalog items (nested catalog-scale experiments).",
)
@click.option("--resume", default=None, type=click.Path(exists=True, file_okay=False),
              help="An existing run directory (the timestamped one holding run_manifest.json "
                   "and traces/). Runs only the (task, trial) pairs that have no trace yet, "
                   "writing into the same directory, and rebuilds the results from every "
                   "trace on disk. Model, simulator, trials and track come from the manifest.")
@click.option("--output", default=None, type=click.Path(), help="Output directory for results")
@click.option("--simulator-model", default="gpt-5-mini", help="Simulator LLM model")
@click.option(
    "--simulator-mode",
    default="llm",
    type=click.Choice(["llm", "symbolic"]),
    show_default=True,
    help="llm: prompted user simulator. symbolic: deterministic books disclosure controller.",
)
@click.option("--max-turns", default=20, help="Maximum turns per conversation")
@click.option(
    "--trial-timeout-seconds",
    default=None,
    type=click.FloatRange(min=1.0),
    help="Optional wall-clock cap per trial. Timed-out trials retain their partial trace and score as failures.",
)
@click.option(
    "--max-provider-cost-usd",
    default=None,
    type=click.FloatRange(min=0.01),
    help=(
        "Hard ceiling on provider-reported spend. The run fails closed if a "
        "successful response omits its provider cost."
    ),
)
@click.option("--concurrency", default=16, help="Max trials running in parallel")
@click.option("--mode", default="tools", type=click.Choice(["tools", "memory"]), show_default=True,
              help="tools: the full harness. memory: one recommend(title) tool and no catalog — "
                   "a memorization probe scored on constraints only. Uses the track's "
                   "policy_memory.md.")
@click.option(
    "--retrieval-mode",
    default="keyword",
    type=click.Choice(["keyword", "structured"]),
    show_default=True,
    help="keyword: BM25 search. structured: typed metadata filtering (books/tools only).",
)
@click.option("--no-tools", is_flag=True, default=False, hidden=True,
              help="Deprecated alias for --mode memory.")
@click.option("--tasks-limit", default=None, type=int, help="Only run the first N tasks (for quick smoke tests)")
@click.option("--tasks-filter", default=None, type=str, help="Comma-separated list of task IDs to run (e.g. task_002,task_004)")
@click.option("--dry-run", is_flag=True, default=False,
              help="Estimate experiment cost from a small calibration sample; no run traces/results are written.")
@click.option("--dry-run-samples", default=9, type=int,
              help="Number of trials to run for calibration under --dry-run (default 9). "
                   "Per-trial cost varies ~18x across the task set, so small samples give "
                   "wide intervals; the estimate reports its own 95% interval.")
@click.option(
    "--dry-run-output", default=None, type=click.Path(),
    help="Write calibration tasks, token counts, parameters, and estimate to JSON.",
)
@click.option("--provider-order", default=None,
              help="Comma-separated OpenRouter upstreams to pin for the agent, e.g. "
                   "deepinfra/fp8. Fallbacks are disabled; an unservable pin fails the call.")
@click.option("--simulator-provider-order", default=None,
              help="Comma-separated OpenRouter upstreams to pin for an LLM user simulator.")
@click.option("--reasoning-effort", default=None, type=click.Choice(["minimal", "low", "medium", "high", "xhigh", "max"]),
              help="Reasoning effort for the agent model. Passed through to litellm, or via "
                   "OpenRouter's `reasoning` field for models that reject the parameter. "
                   "DeepSeek v4 maps low/medium→high and xhigh→max in thinking mode. Not every "
                   "model honours every level — some ignore an unrecognised one and run at their "
                   "default, so verify a level changes behaviour before labelling a run with it.")
@click.option("--agent-prompt-variant", default=None,
              type=click.Choice(["ask-requirements-v1", "ask-and-verify-v1", "ask-verify-protocol-v1",
                                 "keyword-field-list-v1"]),
              help="Opt-in exploratory variant of the agent system prompt (tools mode; see "
                   "AGENT_PROMPT_VARIANTS and KEYWORD_TOOL_VARIANTS). Recorded in the run "
                   "manifest; without it the prompt is unchanged.")
def run(model: str, track: str | None, catalog: str | None, tasks: str | None, policy: str | None, trials: int, catalog_size: int | None, resume: str | None, output: str | None, simulator_model: str, simulator_mode: str, max_turns: int, trial_timeout_seconds: float | None, max_provider_cost_usd: float | None, concurrency: int, mode: str, retrieval_mode: str, no_tools: bool, tasks_limit: int | None, tasks_filter: str | None, dry_run: bool, dry_run_samples: int, dry_run_output: str | None, reasoning_effort: str | None, provider_order: str | None, simulator_provider_order: str | None, agent_prompt_variant: str | None):
    """Run benchmark evaluation."""
    if no_tools:
        # The old flag removed recommend() with everything else, so nothing it
        # ran could register a recommendation. It now means memory mode.
        click.echo("--no-tools is deprecated and now means --mode memory.")
        mode = "memory"
    agent_pins = _provider_order(provider_order)
    simulator_pins = _provider_order(simulator_provider_order)
    if resume:
        # The manifest is what the run was; the flags are what someone typed
        # now. A resumed pass must be the same run, so the manifest wins for
        # everything that defines it, and the content digests must still match.
        from tau_rec.leaderboard.hashing import hash_file, hash_task_dir
        from tau_rec.track import find_track, load_track

        run_dir = Path(resume)
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists() or not (run_dir / "traces").is_dir():
            raise click.UsageError(f"{resume} is not a run directory (needs run_manifest.json and traces/)")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("no_tools") and "mode" not in manifest:
            raise click.UsageError("this run used the old --no-tools ablation; it cannot be resumed, only re-run")
        trk = load_track(track) if track else find_track(manifest.get("track", "movies-v1"))
        mode = manifest.get("mode", "tools")
        catalog = manifest.get("catalog_path") or str(trk.require_catalog())
        tasks = manifest.get("tasks_path") or str(trk.tasks_path)
        policy = manifest.get("policy_path") or str(trk.policy_for(mode))
        drifted = [name for name, recorded, current in (
            ("policy", manifest.get("policy_sha256"), hash_file(policy)),
            ("tasks", manifest.get("tasks_sha256"), hash_task_dir(tasks)),
            ("catalog", manifest.get("catalog_sha256"), hash_file(catalog)),
        ) if recorded and recorded != current]
        if drifted:
            raise click.ClickException(f"{', '.join(drifted)} changed since this run started; a resumed pass would mix content")
        recorded_search = manifest.get("search_profile", "bm25-v1")
        if recorded_search != trk.search_profile:
            raise click.ClickException(
                f"search profile changed since this run started ({recorded_search} -> {trk.search_profile}); "
                "a resumed pass would mix retrieval behavior"
            )
        asyncio.run(_run_benchmark(
            manifest["model"], catalog, tasks, policy, manifest["trials_per_task"], str(run_dir.parent),
            manifest.get("simulator_model", simulator_model), manifest.get("max_turns", max_turns),
            concurrency, mode, manifest.get("tasks_limit"),
            manifest.get("tasks_filter"), manifest.get("reasoning_effort"), track=trk, resume_dir=run_dir,
            trial_timeout_seconds=manifest.get("trial_timeout_seconds"),
            simulator_interaction_mode=manifest.get("simulator_interaction_mode", "llm"),
            retrieval_mode=manifest.get("retrieval_mode", "keyword"),
            catalog_size=manifest.get("catalog_size"),
            # A resume may tighten (or explicitly raise) the operational
            # ceiling after accounting for other matrix cells. Content-defining
            # settings still come exclusively from the manifest.
            max_provider_cost_usd=(
                max_provider_cost_usd
                if max_provider_cost_usd is not None
                else manifest.get("max_provider_cost_usd")
            ),
            provider_order=manifest.get("provider_order"),
            simulator_provider_order=manifest.get("simulator_provider_order"),
            agent_prompt_variant=manifest.get("agent_prompt_variant"),
        ))
        return
    if output is None:
        raise click.UsageError("pass --output (or --resume an existing run directory)")
    trk, catalog, tasks, policy = _resolve_track(track, catalog, tasks, policy)
    if simulator_mode == "symbolic" and trk.domain != "books":
        raise click.UsageError("--simulator-mode symbolic requires a books track")
    if retrieval_mode == "structured" and (mode != "tools" or trk.domain != "books"):
        raise click.UsageError("--retrieval-mode structured requires a books track in tools mode")
    if mode == "memory" and policy == str(trk.policy_path):
        # Memory mode reads the track's tool-free policy rendering unless one
        # was named explicitly.
        policy = str(trk.policy_for("memory"))
    if dry_run_output and not dry_run:
        raise click.UsageError("--dry-run-output requires --dry-run")
    if agent_prompt_variant and (mode != "tools" or dry_run):
        raise click.UsageError("--agent-prompt-variant requires tools mode and a real run")
    if agent_prompt_variant == "keyword-field-list-v1" and retrieval_mode != "keyword":
        raise click.UsageError("--agent-prompt-variant keyword-field-list-v1 requires --retrieval-mode keyword")
    if dry_run:
        asyncio.run(_dry_run_estimate(
            model, catalog, tasks, policy, simulator_model, max_turns, mode,
            tasks_limit, trials, dry_run_samples, reasoning_effort,
            track=trk, tasks_filter=tasks_filter,
            trial_timeout_seconds=trial_timeout_seconds,
            simulator_interaction_mode=simulator_mode,
            retrieval_mode=retrieval_mode,
            catalog_size=catalog_size,
            output_path=dry_run_output,
            provider_order=agent_pins,
            simulator_provider_order=simulator_pins,
        ))
        return
    asyncio.run(_run_benchmark(
        model, catalog, tasks, policy, trials, output, simulator_model, max_turns,
        concurrency, mode, tasks_limit, tasks_filter, reasoning_effort, track=trk,
        trial_timeout_seconds=trial_timeout_seconds,
        simulator_interaction_mode=simulator_mode,
        retrieval_mode=retrieval_mode,
        catalog_size=catalog_size,
        max_provider_cost_usd=max_provider_cost_usd,
        provider_order=agent_pins,
        simulator_provider_order=simulator_pins,
        agent_prompt_variant=agent_prompt_variant,
    ))


def _provider_order(value: str | None) -> list[str] | None:
    """Parse a comma-separated OpenRouter upstream pin list."""
    order = [part.strip() for part in (value or "").split(",") if part.strip()]
    return order or None


def _existing_trials(traces_dir: Path) -> set[tuple[str, int]]:
    """(task_id, trial) pairs that already have a trace on disk."""
    pairs = set()
    for p in traces_dir.glob("*_trial*.json"):
        task_id, _, idx = p.stem.rpartition("_trial")
        if task_id and idx.isdigit():
            pairs.add((task_id, int(idx)))
    return pairs


def _results_from_traces(traces_dir: Path, task_map: dict, evaluator, mode: str = "tools") -> tuple[dict, list[dict]]:
    """Rebuild task_results and trial rows from every trace on disk.

    A trial that died wrote no trace, and a pass that was killed wrote traces
    but no rows, so the files on disk are the only honest record. Scoring is
    deterministic, so re-evaluating reproduces what the run would have written.
    Memory mode counts constraint success: four of the seven policies need
    tools the agent does not have there.
    """
    from tau_rec.data_model.conversation import ConversationTrace

    task_results = {tid: {"n": 0, "c": 0} for tid in task_map}
    rows = []
    for path in sorted(traces_dir.glob("*_trial*.json")):
        trace = ConversationTrace.model_validate_json(path.read_text())
        if trace.task_id not in task_map:
            continue
        result = evaluator.evaluate(task_map[trace.task_id], trace)
        task_results[trace.task_id]["n"] += 1
        score = result.constraint_score if mode == "memory" else result.primary_reward
        if score >= 1.0 - 1e-6:
            task_results[trace.task_id]["c"] += 1
        rows.append(json.loads(json.dumps(result.model_dump(), default=str)))
    rows.sort(key=lambda r: (r["task_id"], r["trial"]))
    return task_results, rows


def _resolve_track(track: str | None, catalog: str | None, tasks: str | None, policy: str | None):
    """Which content a command scores against, from --track and/or explicit paths.

    Explicit paths win over the track's defaults, so a subset of tasks can be
    run against a track. Without --track the paths must all be given, and the
    catalog's own directory decides the track — a directory with no track.json
    is movies-v1, which is what every pre-track invocation meant.
    """
    from tau_rec.track import load_track, track_for_catalog

    if track:
        trk = load_track(track)
        catalog = catalog or str(trk.require_catalog())
        tasks = tasks or str(trk.tasks_path)
        policy = policy or str(trk.policy_path)
    elif not (catalog and tasks and policy):
        raise click.UsageError("pass --track, or all three of --catalog, --tasks and --policy")
    else:
        trk = track_for_catalog(catalog)
    return trk, catalog, tasks, policy


def _stratified_dry_run_sample(tasks: list, sample_size: int) -> list:
    """Deterministically round-robin a calibration sample across complexity."""
    import random

    by_complexity: dict[str, list] = {}
    for task in tasks:
        by_complexity.setdefault(
            getattr(task, "complexity", "unknown"), []
        ).append(task)
    rng = random.Random("dry-run")
    for group in by_complexity.values():
        rng.shuffle(group)

    sampled = []
    strata = sorted(by_complexity)
    while len(sampled) < min(sample_size, len(tasks)):
        progressed = False
        for name in strata:
            if by_complexity[name] and len(sampled) < sample_size:
                sampled.append(by_complexity[name].pop())
                progressed = True
        if not progressed:
            break
    return sampled


async def _dry_run_estimate(
    model: str, catalog_path: str, tasks_path: str, policy_path: str,
    simulator_model: str, max_turns: int, mode: str, tasks_limit: int | None,
    full_trials: int, sample_size: int, reasoning_effort: str | None = None,
    track=None, tasks_filter: str | None = None,
    trial_timeout_seconds: float | None = None,
    simulator_interaction_mode: str = "llm",
    retrieval_mode: str = "keyword",
    catalog_size: int | None = None,
    output_path: str | None = None,
    provider_order: list[str] | None = None,
    simulator_provider_order: list[str] | None = None,
) -> None:
    """Run `sample_size` trials (one per randomly selected task) to measure token
    usage, then extrapolate to the full experiment and print a cost estimate.
    No results or traces are written to disk."""
    calibration_started = datetime.now().astimezone()
    from tau_rec.track import track_for_catalog
    spec = (track or track_for_catalog(catalog_path)).spec
    if simulator_interaction_mode == "symbolic" and spec.id != "books":
        raise click.UsageError("symbolic simulator mode requires a books track")
    if retrieval_mode == "structured" and (mode != "tools" or spec.id != "books"):
        raise click.UsageError("structured retrieval requires a books track in tools mode")
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import Task
    from tau_rec.catalog.search import CatalogSearch
    from tau_rec.environment.tools import ToolKit
    from tau_rec.agents.litellm_agent import LiteLLMAgent, build_agent_system_prompt
    from tau_rec.simulator.user_simulator import UserSimulator
    from tau_rec.orchestrator.orchestrator import Orchestrator
    from tau_rec import cost as costlib

    catalog = Catalog.from_json(catalog_path, spec)
    if catalog_size is not None:
        if catalog_size > len(catalog):
            raise click.UsageError(
                f"--catalog-size {catalog_size:,} exceeds the {len(catalog):,}-item catalog"
            )
        catalog = Catalog(items=catalog.items[:catalog_size])
    search = (
        CatalogSearch(
            catalog,
            spec,
            profile=(track or track_for_catalog(catalog_path)).search_profile,
        )
        if retrieval_mode == "keyword" else None
    )
    policy_text = Path(policy_path).read_text()
    task_files = sorted(Path(tasks_path).glob("*.json"))
    all_tasks = [Task.from_json(str(tf)) for tf in task_files]
    if tasks_filter:
        allowed = set(tasks_filter.split(","))
        all_tasks = [task for task in all_tasks if task.id in allowed]
    elif tasks_limit:
        all_tasks = all_tasks[:tasks_limit]

    total_trials_full = len(all_tasks) * full_trials
    sample_size = min(sample_size, len(all_tasks))
    # Trial cost spans roughly 18x and grows faster than conversation length,
    # because every call resends history. Equal round-robin coverage prevents a
    # small calibration from accidentally sampling one complexity repeatedly.
    sampled = _stratified_dry_run_sample(all_tasks, sample_size)
    sample_size = len(sampled)

    click.echo(f"Dry run: calibrating on {sample_size} trials...")
    click.echo(f"  agent:     {model}" + (f" (reasoning_effort={reasoning_effort})" if reasoning_effort else ""))
    simulator_label = (
        "symbolic books controller"
        if simulator_interaction_mode == "symbolic" else simulator_model
    )
    click.echo(f"  simulator: {simulator_label}")
    click.echo(f"  sample:    {', '.join(f'{t.id}({getattr(t, 'complexity', '?')})' for t in sampled)}")

    costlib.reset()

    completed = 0
    per_trial_tokens: list[dict[str, int]] = []
    per_trial_provider_costs: list[dict] = []

    async def run_one(task):
        nonlocal completed
        toolkit = ToolKit(
            catalog=catalog,
            search=search,
            user_histories=task.user_history or {},
            spec=spec,
            mode=mode,
            retrieval_mode=retrieval_mode,
            structured_profile=(track or track_for_catalog(catalog_path)).structured_profile,
        )
        system_prompt = build_agent_system_prompt(
            policy_text,
            task.user_id,
            spec=spec,
            mode=mode,
            retrieval_mode=retrieval_mode,
            structured_profile=(track or track_for_catalog(catalog_path)).structured_profile,
        )
        harness = (track or track_for_catalog(catalog_path)).harness
        agent = LiteLLMAgent(model=model, system_prompt=system_prompt, tool_definitions=toolkit.tool_definitions(), reasoning_effort=reasoning_effort, provider_order=provider_order, harness=harness)
        simulator = UserSimulator(
            model=simulator_model,
            task=task,
            spec=spec,
            catalog=(
                catalog
                if mode == "tools" or simulator_interaction_mode == "symbolic"
                else None
            ),
            interaction_mode=simulator_interaction_mode,
            provider_order=simulator_provider_order,
            harness=harness,
        )
        orch = Orchestrator(
            agent=agent,
            simulator=simulator,
            toolkit=toolkit,
            max_turns=max_turns,
            spec=spec,
            trial_timeout_seconds=trial_timeout_seconds,
            harness=harness,
        )
        before = costlib.snapshot()
        before_provider = costlib.provider_cost_snapshot()
        failure = None
        try:
            await orch.run(task_id=task.id, model=model, trial=0)
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            failure = e
        after = costlib.snapshot()
        after_provider = costlib.provider_cost_snapshot()
        delta = {k: after[k] - before[k] for k in after}
        provider_delta = {
            "agent_usd": (
                after_provider["agent"]["usd"]
                - before_provider["agent"]["usd"]
            ),
            "simulator_usd": (
                after_provider["simulator"]["usd"]
                - before_provider["simulator"]["usd"]
            ),
            "reported_calls": (
                after_provider["reported_calls"]
                - before_provider["reported_calls"]
            ),
            "successful_response_calls": (
                after_provider["successful_response_calls"]
                - before_provider["successful_response_calls"]
            ),
        }
        provider_delta["total_usd"] = (
            provider_delta["agent_usd"] + provider_delta["simulator_usd"]
        )
        provider_delta["coverage"] = (
            provider_delta["reported_calls"]
            / provider_delta["successful_response_calls"]
            if provider_delta["successful_response_calls"] else None
        )
        completed += 1
        if failure is None:
            per_trial_tokens.append(delta)
            per_trial_provider_costs.append(provider_delta)
            hit = 100 * delta["agent_cached"] / delta["agent_input"] if delta["agent_input"] else 0
            click.echo(f"  [{completed}/{sample_size}] {task.id}  agent tokens in={delta['agent_input']:,} "
                       f"({hit:.0f}% cached) out={delta['agent_output']:,}")
        else:
            click.echo(f"  [{completed}/{sample_size}] {task.id} FAILED: {type(failure).__name__}: {failure!s:.80s}")

    # Sequentially, so each trial's token delta is attributable to that trial.
    # A concurrent sample can only give an average, which hides the spread that
    # makes a small sample untrustworthy in the first place.
    for task in sampled:
        await run_one(task)

    failed = sample_size - len(per_trial_tokens)
    if failed:
        raise click.ClickException(
            f"cost calibration invalid: {failed}/{sample_size} sampled trial(s) failed. "
            "Fix provider authentication/model availability and rerun; do not use a "
            "partial or zero-token estimate."
        )

    est = costlib.estimate_cost(
        model,
        simulator_model,
        sample_size,
        total_trials_full,
        simulator_enabled=simulator_interaction_mode != "symbolic",
    )

    click.echo()
    click.echo("=== Per-trial averages (from calibration) ===")
    p = est["per_trial"]
    click.echo(f"  agent     input={p['agent_input']:>8.0f}  output={p['agent_output']:>8.0f}  tokens"
               f"  (cached input={p['agent_cached']:>8.0f})")
    click.echo(f"  simulator input={p['sim_input']:>8.0f}  output={p['sim_output']:>8.0f}  tokens")

    click.echo()
    click.echo(f"=== Full experiment extrapolation ({len(all_tasks)} tasks × {full_trials} trials = {total_trials_full} trials) ===")
    t = est["estimated_total_tokens"]
    hit = 100 * t["agent_cached"] / t["agent_input"] if t["agent_input"] else 0
    click.echo(f"  agent     input={t['agent_input']:>10.0f}  output={t['agent_output']:>10.0f}  tokens"
               f"  ({hit:.0f}% of input served from cache)")
    click.echo(f"  simulator input={t['sim_input']:>10.0f}  output={t['sim_output']:>10.0f}  tokens")

    click.echo()
    click.echo("=== Cost estimate ===")
    ap = est["prices"]["agent"]
    sp = est["prices"]["sim"]
    if ap is None:
        click.echo(f"  agent {model}: PRICING UNKNOWN in litellm.model_cost — cost unavailable")
    if sp is None:
        click.echo(f"  simulator {simulator_model}: PRICING UNKNOWN in litellm.model_cost — cost unavailable")
    c = est["estimated_cost_usd"]
    if c["agent"] is not None:
        click.echo(f"  agent cost:     ${c['agent']:>8.2f}")
    if c["sim"] is not None:
        click.echo(f"  simulator cost: ${c['sim']:>8.2f}")
    if c["total"] is not None:
        click.secho(f"  TOTAL:          ${c['total']:>8.2f}", fg="green", bold=True)
    else:
        click.secho("  TOTAL: n/a (see missing pricing above)", fg="yellow")

    # A single number invites more confidence than a handful of trials can
    # support. Per-trial cost is heavy-tailed here, so report the spread and a
    # 95% interval on the mean and let the reader see how loose it really is.
    costs = [costlib.price_tokens(model, simulator_model, d)["total"] for d in per_trial_tokens]
    costs = [x for x in costs if x is not None]
    if len(costs) >= 2:
        import statistics
        mean = statistics.mean(costs)
        sem = statistics.stdev(costs) / (len(costs) ** 0.5)
        lo, hi = (mean - 1.96 * sem) * total_trials_full, (mean + 1.96 * sem) * total_trials_full
        click.echo()
        click.echo(f"  per-trial cost: min ${min(costs):.3f}  median ${statistics.median(costs):.3f}  max ${max(costs):.3f}")
        click.secho(f"  95% interval on the full run: ${max(lo, 0):.2f} – ${hi:.2f}", fg="yellow")
        if hi > 0 and hi / max(lo, 1e-9) > 2:
            click.echo("  Interval spans more than 2x. Raise --dry-run-samples before trusting the midpoint.")

    provider_calls = sum(
        row["successful_response_calls"] for row in per_trial_provider_costs
    )
    provider_reported_calls = sum(
        row["reported_calls"] for row in per_trial_provider_costs
    )
    provider_costs = [row["total_usd"] for row in per_trial_provider_costs]
    provider_coverage = (
        provider_reported_calls / provider_calls if provider_calls else None
    )
    provider_projection = (
        sum(provider_costs) / sample_size * total_trials_full
        if provider_calls and provider_reported_calls == provider_calls else None
    )
    if provider_projection is not None:
        click.echo()
        click.echo(
            f"  provider-reported sample spend: ${sum(provider_costs):.4f}; "
            f"linear projection: ${provider_projection:.2f} "
            f"({provider_reported_calls}/{provider_calls} calls covered)"
        )

    click.echo()
    click.echo(
        f"Note: estimate is based on {sample_size} sampled trial(s) of "
        f"{total_trials_full}. Per-task cost spans roughly 18x across the task "
        f"set, so trust the interval above rather than the midpoint. A completed "
        f"run writes its real usage to usage.json — calibrate against that."
    )

    if output_path:
        from tau_rec import __version__ as tau_rec_version
        from tau_rec.agents.litellm_agent import (
            request_parameter_manifest as agent_parameter_manifest,
        )
        from tau_rec.leaderboard.hashing import hash_file, hash_task_dir
        from tau_rec.simulator.user_simulator import (
            request_parameter_manifest as simulator_parameter_manifest,
        )

        serializable_estimate = {
            **est,
            "prices": {
                role: prices._asdict() if prices is not None else None
                for role, prices in est["prices"].items()
            },
        }
        record = {
            "schema_version": "tau-rec-dry-run-v2",
            "started_at": calibration_started.isoformat(timespec="seconds"),
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "tau_rec_version": tau_rec_version,
            "track": (track or track_for_catalog(catalog_path)).id,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "simulator_model": simulator_model,
            "simulator_mode": simulator_interaction_mode,
            "retrieval_mode": retrieval_mode,
            "catalog_size": len(catalog),
            "trials_per_task": full_trials,
            "task_count": len(all_tasks),
            "total_trials": total_trials_full,
            "sampled_task_ids": [task.id for task in sampled],
            "sampled_trial_tokens": per_trial_tokens,
            "sampled_trial_provider_costs": per_trial_provider_costs,
            "observed_sample_cost_usd": (
                sum(costs) if len(costs) == sample_size else None
            ),
            "observed_sample_cost_basis": "local token-price estimate",
            "observed_sample_provider_reported_cost_usd": (
                sum(provider_costs)
                if provider_calls and provider_reported_calls == provider_calls
                else None
            ),
            "provider_reported_cost": {
                "reported_calls": provider_reported_calls,
                "successful_response_calls": provider_calls,
                "coverage": provider_coverage,
                "projected_total_usd": provider_projection,
            },
            "estimate": serializable_estimate,
            "harness_profile": (track or track_for_catalog(catalog_path)).harness.manifest(),
            "api_parameters": {
                "agent": agent_parameter_manifest(
                    model, reasoning_effort, provider_order,
                    (track or track_for_catalog(catalog_path)).harness,
                ),
                "simulator": simulator_parameter_manifest(
                    spec, mode, simulator_interaction_mode, simulator_provider_order
                ),
            },
            "artifact": {
                "catalog_sha256": hash_file(catalog_path),
                "tasks_sha256": hash_task_dir(tasks_path),
                "policy_sha256": hash_file(policy_path),
            },
        }
        record_path = Path(output_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        click.echo(f"Calibration record written to {record_path}")


async def _run_benchmark(
    model: str, catalog_path: str, tasks_path: str, policy_path: str,
    trials: int, output_dir: str, simulator_model: str, max_turns: int,
    concurrency: int, mode: str = "tools", tasks_limit: int | None = None,
    tasks_filter: str | None = None, reasoning_effort: str | None = None,
    track=None, resume_dir: Path | None = None,
    trial_timeout_seconds: float | None = None,
    simulator_interaction_mode: str = "llm",
    retrieval_mode: str = "keyword",
    catalog_size: int | None = None,
    max_provider_cost_usd: float | None = None,
    provider_order: list[str] | None = None,
    simulator_provider_order: list[str] | None = None,
    agent_prompt_variant: str | None = None,
):
    from tau_rec.track import track_for_catalog
    track = track or track_for_catalog(catalog_path)
    spec = track.spec
    effective_structured_profile = track.structured_profile
    effective_harness = track.harness
    if simulator_interaction_mode == "symbolic" and spec.id != "books":
        raise click.UsageError("symbolic simulator mode requires a books track")
    if retrieval_mode == "structured" and (mode != "tools" or spec.id != "books"):
        raise click.UsageError("structured retrieval requires a books track in tools mode")
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import Task
    from tau_rec.catalog.search import CatalogSearch
    from tau_rec.environment.tools import ToolKit
    from tau_rec.agents.litellm_agent import (
        LiteLLMAgent, build_agent_system_prompt,
        request_parameter_manifest as agent_parameter_manifest,
    )
    from tau_rec.simulator.user_simulator import (
        UserSimulator, books_user_versions,
        request_parameter_manifest as simulator_parameter_manifest,
    )
    from tau_rec.orchestrator.orchestrator import Orchestrator
    from tau_rec.evaluator.evaluator import CombinedEvaluator
    from tau_rec.metrics.pass_k import pass_at_k, aggregate_pass_k
    from tau_rec import __version__ as tau_rec_version
    from tau_rec import cost as costlib

    # Runs normally execute in fresh CLI processes, but tests and library users
    # can invoke this coroutine more than once. Never inherit another run's
    # accounting state.
    costlib.reset()

    catalog = Catalog.from_json(catalog_path, spec)
    full_catalog_size = len(catalog)
    if catalog_size is not None:
        if catalog_size > full_catalog_size:
            raise click.UsageError(
                f"--catalog-size {catalog_size:,} exceeds the "
                f"{full_catalog_size:,}-item catalog"
            )
        catalog = Catalog(items=catalog.items[:catalog_size])
    search = (
        CatalogSearch(catalog, spec, profile=track.search_profile)
        if retrieval_mode == "keyword" else None
    )
    evaluator = CombinedEvaluator(catalog, spec)
    out = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resume_dir if resume_dir else out / timestamp
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    prior_provider_spend = 0.0
    if resume_dir:
        usage_paths = sorted(run_dir.glob("usage*.json"))
        cumulative_budget_totals: list[float] = []
        legacy_usage_total = 0.0
        for usage_path in usage_paths:
            try:
                usage_payload = json.loads(usage_path.read_text())
                legacy_usage_total += float(
                    usage_payload.get("provider_reported_cost_usd", {}).get(
                        "total_usd", 0.0
                    ) or 0.0
                )
                cumulative = usage_payload.get("provider_budget", {}).get(
                    "total_usd"
                )
                if isinstance(cumulative, (int, float)) and not isinstance(
                    cumulative, bool
                ):
                    cumulative_budget_totals.append(float(cumulative))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                raise click.ClickException(
                    f"cannot recover prior provider spend from {usage_path}"
                )
        prior_provider_spend = (
            max(cumulative_budget_totals)
            if cumulative_budget_totals else legacy_usage_total
        )
        # A killed process may have complete trace files but no usage file.
        # Recover spend from the per-call records before accepting a resume
        # ceiling; otherwise interruption silently resets the budget counter.
        if not usage_paths:
            missing_cost_calls = 0
            for trace_path in sorted((run_dir / "traces").glob("*.json")):
                trace_payload = json.loads(trace_path.read_text())
                for calls in trace_payload.get("api_calls", {}).values():
                    for call in calls:
                        if call.get("status") not in {"ok", "empty"}:
                            continue
                        reported = call.get("provider_reported_cost_usd")
                        if isinstance(reported, (int, float)) and not isinstance(
                            reported, bool
                        ):
                            prior_provider_spend += float(reported)
                        else:
                            missing_cost_calls += 1
            if missing_cost_calls and max_provider_cost_usd is not None:
                raise click.ClickException(
                    "cannot enforce resume budget: "
                    f"{missing_cost_calls} prior successful calls lack provider cost"
                )
    if max_provider_cost_usd is not None:
        try:
            costlib.configure_provider_budget(
                max_provider_cost_usd,
                already_spent_usd=prior_provider_spend,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    policy_text = Path(policy_path).read_text()

    # Written before any trial runs, so a killed run still records what it was.
    # Nothing else on disk captures reasoning_effort: a run without this is
    # unlabelable after the fact and has to be thrown away. A resumed pass
    # keeps the original manifest and only notes that it happened.
    from tau_rec.harness import harness_commit
    from tau_rec.leaderboard.hashing import hash_file, hash_task_dir

    if resume_dir:
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        # A resume keeps the structured and harness profiles the run started
        # under.
        effective_structured_profile = manifest.get(
            "structured_profile", effective_structured_profile
        )
        from tau_rec.harness import LEGACY, get_harness_profile
        effective_harness = get_harness_profile(
            manifest.get("harness_profile", {}).get("id", LEGACY.id)
        )
        # And the agent prompt variant, which defines the prompt.
        agent_prompt_variant = manifest.get("agent_prompt_variant")
        manifest.setdefault("resumed_at", []).append(datetime.now().astimezone().isoformat(timespec="seconds"))
        if max_provider_cost_usd is not None:
            manifest["max_provider_cost_usd"] = max_provider_cost_usd
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "model": model,
                "reasoning_effort": reasoning_effort,
                # A symbolic user calls no model.
                "simulator_model": simulator_model if simulator_interaction_mode == "llm" else None,
                "simulator_interaction_mode": simulator_interaction_mode,
                "trials_per_task": trials,
                "max_turns": max_turns,
                "trial_timeout_seconds": trial_timeout_seconds,
                "max_provider_cost_usd": max_provider_cost_usd,
                # Which content set and agent configuration this run scored
                # under. make-entry reads these in preference to its flags.
                # policy_sha256 below is of the policy this mode actually used.
                "track": track.id,
                "domain": track.domain,
                "catalog_path": str(Path(catalog_path)),
                "tasks_path": str(Path(tasks_path)),
                "policy_path": str(Path(policy_path)),
                "mode": mode,
                "retrieval_mode": retrieval_mode,
                "catalog_size": len(catalog),
                "catalog_full_size": full_catalog_size,
                "catalog_prefix_id_sha256": hashlib.sha256(
                    "\n".join(item.id for item in catalog.items).encode()
                ).hexdigest(),
                "search_profile": track.search_profile,
                "structured_profile": track.structured_profile,
                "harness_profile": effective_harness.manifest(),
                "harness_commit": harness_commit(),
                "tasks_filter": tasks_filter,
                "tasks_limit": tasks_limit,
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "tau_rec_version": tau_rec_version,
                "api_parameters": {
                    "agent": agent_parameter_manifest(
                        model, reasoning_effort, provider_order, effective_harness
                    ),
                    "simulator": simulator_parameter_manifest(
                        spec, mode, simulator_interaction_mode, simulator_provider_order
                    ),
                },
                "provider_order": provider_order,
                "simulator_provider_order": simulator_provider_order,
                **({"agent_prompt_variant": agent_prompt_variant} if agent_prompt_variant else {}),
                **(
                    {"books_user_versions": books_user_versions(simulator_interaction_mode)}
                    if spec.id == "books" else {}
                ),
                "policy_sha256": hash_file(policy_path),
                "tasks_sha256": hash_task_dir(tasks_path),
                "catalog_sha256": hash_file(catalog_path),
            },
            indent=2,
        )
        + "\n"
    )

    task_dir = Path(tasks_path)
    task_files = sorted(task_dir.glob("*.json"))
    all_tasks = [Task.from_json(str(tf)) for tf in task_files]
    if tasks_filter:
        allowed = set(tasks_filter.split(","))
        all_tasks = [t for t in all_tasks if t.id in allowed]
    elif tasks_limit:
        all_tasks = all_tasks[:tasks_limit]

    # A resumed pass runs only the pairs with no trace yet; a fresh run has none.
    existing = _existing_trials(traces_dir) if resume_dir else set()
    pending = [(task, i) for task in all_tasks for i in range(trials) if (task.id, i) not in existing]

    tag = " [MEMORY MODE]" if mode == "memory" else ""
    if simulator_interaction_mode == "symbolic":
        tag += " [SYMBOLIC USER]"
    if retrieval_mode == "structured":
        tag += " [STRUCTURED RETRIEVAL]"
    if max_provider_cost_usd is not None:
        tag += f" [PROVIDER COST CAP ${max_provider_cost_usd:.2f}]"
    if resume_dir:
        click.echo(f"Resuming {run_dir}: {len(existing)} trials on disk, {len(pending)} to run{tag}...")
    else:
        click.echo(f"Running {model} on {track.id}: {len(all_tasks)} tasks x {trials} trials (concurrency={concurrency}){tag}...")

    sem = asyncio.Semaphore(concurrency)
    total_trials = len(pending)
    completed_count = 0

    failed_count = 0
    budget_stopped = False
    budget_message: str | None = None

    async def run_trial(task, trial_idx, toolkit, system_prompt):
        nonlocal completed_count, failed_count, budget_stopped, budget_message
        async with sem:
            if budget_stopped:
                return None
            try:
                costlib.check_provider_budget()
                agent = LiteLLMAgent(model=model, system_prompt=system_prompt, tool_definitions=toolkit.tool_definitions(), reasoning_effort=reasoning_effort, provider_order=provider_order, harness=effective_harness)
                simulator = UserSimulator(
                    model=simulator_model,
                    task=task,
                    spec=spec,
                    catalog=(
                        catalog
                        if mode == "tools" or simulator_interaction_mode == "symbolic"
                        else None
                    ),
                    interaction_mode=simulator_interaction_mode,
                    provider_order=simulator_provider_order,
                    harness=effective_harness,
                )
                orch = Orchestrator(
                    agent=agent,
                    simulator=simulator,
                    toolkit=toolkit,
                    max_turns=max_turns,
                    spec=spec,
                    trial_timeout_seconds=trial_timeout_seconds,
                    harness=effective_harness,
                )
                trace = await orch.run(task_id=task.id, model=model, trial=trial_idx)
                result = evaluator.evaluate(task, trace)
                trace_doc = trace.model_dump(mode="json")
                trace_doc["evaluation"] = result.model_dump(mode="json")
                if simulator.decision_log:
                    trace_doc["simulator_decisions"] = simulator.decision_log
                api_calls = {
                    "agent": agent.call_log,
                    "simulator": simulator.call_log,
                }
                if api_calls["agent"] or api_calls["simulator"]:
                    trace_doc["api_calls"] = api_calls
                (traces_dir / f"{task.id}_trial{trial_idx}.json").write_text(
                    json.dumps(trace_doc, indent=2, default=str)
                )
                completed_count += 1
                pct = completed_count / total_trials
                bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
                click.echo(f"  [{bar}] {pct:5.1%} ({completed_count}/{total_trials})  {task.id} trial {trial_idx}: primary={result.primary_reward}")
                return task.id, result
            except costlib.ProviderBudgetError as e:
                if not budget_stopped:
                    budget_message = str(e)
                    click.echo(f"  BUDGET STOP: {budget_message}")
                budget_stopped = True
                return None
            except Exception as e:
                failed_count += 1
                click.echo(f"  FAIL  {task.id} trial {trial_idx}: {type(e).__name__}: {e!s:.120s}")
                return None

    coros = []
    per_task_setup: dict[str, tuple] = {}
    for task, trial_idx in pending:
        if task.id not in per_task_setup:
            toolkit = ToolKit(
                catalog=catalog,
                search=search,
                user_histories=task.user_history or {},
                spec=spec,
                mode=mode,
                retrieval_mode=retrieval_mode,
                structured_profile=effective_structured_profile,
            )
            per_task_setup[task.id] = (
                toolkit,
                build_agent_system_prompt(
                    policy_text,
                    task.user_id,
                    spec=spec,
                    mode=mode,
                    retrieval_mode=retrieval_mode,
                    structured_profile=effective_structured_profile,
                    prompt_variant=agent_prompt_variant,
                ),
            )
        toolkit, system_prompt = per_task_setup[task.id]
        coros.append(run_trial(task, trial_idx, toolkit, system_prompt))

    await asyncio.gather(*coros)

    # Results come from the traces on disk, never from this pass's return
    # values: a failed trial wrote no trace, a resumed pass adds to earlier
    # ones, and scoring is deterministic either way.
    task_results, all_trial_results = _results_from_traces(traces_dir, {t.id: t for t in all_tasks}, evaluator, mode)

    if failed_count:
        click.echo(f"\n{failed_count} trial(s) failed and wrote no trace. Fill them with: tau-rec run --resume {run_dir}")
    short = sorted(tid for tid, r in task_results.items() if r["n"] < trials)
    if short:
        click.echo(f"{len(short)} task(s) have fewer than {trials} trials on disk; pass^k counts a task with n < k as 0.")

    for k in [1, 2, 4]:
        score = aggregate_pass_k(task_results, k=k)
        click.echo(f"pass^{k} = {score:.3f}")

    (run_dir / "trial_results.json").write_text(json.dumps(all_trial_results, indent=2, default=str))
    (run_dir / "task_results.json").write_text(json.dumps(task_results, indent=2))

    # Every agent and simulator response already reports its token usage, and
    # until now the run threw the totals away at exit — leaving `--dry-run`
    # extrapolations with nothing to be checked against. Record what the run
    # actually consumed so the next estimate can be calibrated against it.
    usage = costlib.snapshot()
    priced = costlib.price_tokens(
        model,
        simulator_model,
        usage,
        simulator_enabled=simulator_interaction_mode != "symbolic",
    )
    provider_priced = costlib.provider_cost_snapshot()
    budget_status = costlib.provider_budget_status()
    # A resumed pass records its own spend beside the original's, not over it.
    usage_name = f"usage_resume_{timestamp}.json" if resume_dir else "usage.json"
    # The counters belong to this process only. In a resume, all_trial_results
    # also contains traces from earlier processes, so using its length would
    # understate the per-trial token cost of the resumed pass.
    trials_done = completed_count
    (run_dir / usage_name).write_text(json.dumps({
        "agent_model": model,
        "simulator_model": simulator_model,
        "trials_attempted": total_trials,
        "trials_completed": trials_done,
        "tokens": usage,
        "per_trial": {k: (v / trials_done if trials_done else 0) for k, v in usage.items()},
        # The v1.3.0 field name, kept for compatibility. This is a local
        # token-price estimate, not necessarily the amount charged.
        "cost_usd": priced,
        "estimated_cost_usd": priced,
        "provider_reported_cost_usd": provider_priced,
        "provider_budget": budget_status,
    }, indent=2))
    if budget_stopped:
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["budget_stop"] = {
            "reason": budget_message,
            "status": budget_status,
            "stopped_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if (
        provider_priced["successful_response_calls"]
        and provider_priced["coverage"] == 1.0
    ):
        click.echo(
            f"provider-reported cost = ${provider_priced['total_usd']:.2f} "
            f"over {trials_done} trials"
        )
    if priced.get("total") is not None:
        click.echo(
            f"token-price estimate = ${priced['total']:.2f} over "
            f"{trials_done} trials"
        )

    click.echo(f"\nRun artifacts saved to {run_dir}")
    import os as _os
    # litellm httpx connection pools do not close cleanly; force exit after
    # files are written, but preserve a distinct status for a capped partial run.
    _os._exit(2 if budget_stopped else 0)


@main.command()
@click.option("--results", required=True, type=click.Path(exists=True), help="Path to task_results.json")
@click.option("--trials", default=None, type=click.Path(exists=True), help="Path to trial_results.json (enables detailed breakdown)")
@click.option("--tasks", default=None, type=click.Path(exists=True), help="Path to tasks directory (enables per-dimension breakdown)")
def report(results: str, trials: str | None, tasks: str | None):
    """Generate metrics report from results."""
    from tau_rec.metrics.pass_k import pass_at_k, aggregate_pass_k
    from tau_rec.metrics.bootstrap import bootstrap_ci

    task_results = json.loads(Path(results).read_text())

    click.echo("=== Overall ===")
    for k in [1, 2, 4]:
        score = aggregate_pass_k(task_results, k=k)
        per_task = [pass_at_k(r["n"], r["c"], k) for r in task_results.values()]
        if len(set(per_task)) > 1:
            lo, hi = bootstrap_ci(per_task)
        else:
            lo, hi = per_task[0], per_task[0]
        click.echo(f"  pass^{k} = {score:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    if tasks:
        from tau_rec.data_model.task import Task
        task_dir = Path(tasks)
        all_tasks = {Task.from_json(str(tf)).id: Task.from_json(str(tf)) for tf in sorted(task_dir.glob("*.json"))}

        for dim, get_val in [("complexity", lambda t: t.complexity), ("reveal_difficulty", lambda t: t.reveal_difficulty)]:
            groups: dict[str, list[str]] = {}
            for tid, t in all_tasks.items():
                groups.setdefault(get_val(t), []).append(tid)

            click.echo(f"\n=== By {dim} ===")
            for group in sorted(groups):
                subset = {tid: task_results[tid] for tid in groups[group] if tid in task_results}
                if not subset:
                    continue
                p1 = aggregate_pass_k(subset, k=1)
                per_task_p1 = [pass_at_k(r["n"], r["c"], 1) for r in subset.values()]
                lo, hi = bootstrap_ci(per_task_p1) if len(set(per_task_p1)) > 1 else (per_task_p1[0], per_task_p1[0])
                click.echo(f"  {group:12s}  n={len(subset):2d}  pass^1={p1:.3f}  [{lo:.3f}, {hi:.3f}]")

    if trials:
        import statistics
        trial_data = json.loads(Path(trials).read_text())
        n = len(trial_data)

        mean_constraint = sum(t["constraint_score"] for t in trial_data) / n
        mean_policy = sum(t["policy_score"] for t in trial_data) / n

        # Turns: conditional on a recommend() call (None means agent never recommended).
        # Reporting an unconditional mean would silently impute 0 for non-recommending trials.
        turns_vals = [t["efficiency"]["turns_to_first_recommendation"] for t in trial_data]
        no_rec_count = sum(1 for v in turns_vals if v is None)
        turns_clean = [v for v in turns_vals if v is not None]
        cond_turns = sum(turns_clean) / len(turns_clean) if turns_clean else 0.0

        # Tools: distribution is right-skewed (one trial can call >100 tools), so report
        # median as the headline and mean alongside for transparency.
        tool_vals = [t["efficiency"]["total_tool_calls"] for t in trial_data]
        median_tools = statistics.median(tool_vals)
        mean_tools = sum(tool_vals) / n

        rej_vals = [t["efficiency"].get("rejection_count", 0) for t in trial_data]
        mean_rejections = sum(rej_vals) / n
        zero_rej = sum(1 for v in rej_vals if v == 0)

        click.echo("\n=== Score breakdown ===")
        click.echo(f"  mean constraint_score = {mean_constraint:.3f}")
        click.echo(f"  mean policy_score     = {mean_policy:.3f}")

        click.echo("\n=== Efficiency ===")
        click.echo(f"  trials without recommend()   = {no_rec_count}/{n} ({100*no_rec_count/n:.1f}%)")
        click.echo(f"  mean turns to recommendation = {cond_turns:.2f}  (conditional on recommend, n={len(turns_clean)})")
        click.echo(f"  median tool calls per trial  = {median_tools:.0f}  (mean {mean_tools:.1f})")
        click.echo(f"  mean user rejections/trial   = {mean_rejections:.2f}")
        click.echo(f"  trials with 0 rejections     = {zero_rej}/{n} ({100*zero_rej/n:.0f}%)")

        violations: dict[str, int] = {}
        for t in trial_data:
            for v in t["policy_detail"]["violations"]:
                violations[v] = violations.get(v, 0) + 1
        if violations:
            click.echo("\n=== Policy violations ===")
            for flag, count in sorted(violations.items(), key=lambda x: -x[1]):
                click.echo(f"  {flag:30s}  {count:4d} trials ({100*count/len(trial_data):.1f}%)")


@main.command()
@click.option("--track", default=None, type=click.Path(exists=True, file_okay=False),
              help="Track directory; supplies catalog, tasks, domain, and retrieval profile.")
@click.option("--catalog", default=None, type=click.Path(exists=True), help="Path to catalog JSON")
@click.option("--tasks", default=None, type=click.Path(exists=True), help="Path to tasks directory")
@click.option("--results", default=None, type=click.Path(exists=True), help="Path to task_results.json (for empirical checks)")
@click.option("--trials", default=None, type=click.Path(exists=True), help="Path to trial_results.json (for failure analysis)")
def health(track: str | None, catalog: str | None, tasks: str | None, results: str | None, trials: str | None):
    """Run structural validation + empirical health check on tasks."""
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import Task
    from tau_rec.catalog.validator import CatalogValidator
    from tau_rec.track import load_track, track_for_catalog

    if track:
        trk = load_track(track)
        catalog = catalog or str(trk.require_catalog())
        tasks = tasks or str(trk.tasks_path)
    elif catalog and tasks:
        trk = track_for_catalog(catalog)
    else:
        raise click.UsageError("pass --track, or both --catalog and --tasks")

    cat = Catalog.from_json(catalog, trk.spec)
    validator = CatalogValidator(cat, trk.spec)

    task_dir = Path(tasks)
    task_files = sorted(task_dir.glob("*.json"))
    all_tasks = {}
    for tf in task_files:
        t = Task.from_json(str(tf))
        all_tasks[t.id] = t

    # --- Layer 1: Structural checks ---
    click.echo(f"=== Structural Validation ({len(all_tasks)} tasks) ===\n")
    struct_errors = 0
    struct_warnings = 0
    near_retrieval = []

    for tid in sorted(all_tasks):
        result = validator.validate_task(all_tasks[tid])
        if result.errors:
            struct_errors += len(result.errors)
            for e in result.errors:
                click.echo(f"  ERR   {tid}: {e}")
        if result.warnings:
            struct_warnings += len(result.warnings)
            for w in result.warnings:
                click.echo(f"  WARN  {tid}: {w}")
        if not result.valid_as_designed:
            click.echo(f"  FAIL  {tid}: invalid design (solvable={result.solvable}, nvr={all_tasks[tid].no_valid_recommendation})")
            struct_errors += 1
        if result.solution_set_size <= 2 and not all_tasks[tid].no_valid_recommendation:
            near_retrieval.append(tid)

    click.echo(f"\nStructural: {struct_errors} errors, {struct_warnings} warnings, {len(near_retrieval)} near-retrieval tasks")

    # --- Layer 2: Empirical health (if results provided) ---
    if not results:
        click.echo("\nNo --results provided. Skipping empirical health check.")
        click.echo("Run a benchmark first, then pass --results and --trials.")
        return

    task_results = json.loads(Path(results).read_text())
    trial_data = json.loads(Path(trials).read_text()) if trials else []

    click.echo(f"\n=== Empirical Health ({len(task_results)} tasks with results) ===\n")

    healthy = []
    too_easy = []
    too_hard = []
    untested = []

    for tid in sorted(all_tasks):
        task = all_tasks[tid]
        r = task_results.get(tid)

        if r is None:
            untested.append(tid)
            continue

        rate = r["c"] / r["n"] if r["n"] > 0 else 0

        if rate == 1.0:
            too_easy.append(tid)
        elif rate == 0.0:
            too_hard.append(tid)
        else:
            healthy.append(tid)

    click.echo(f"  HEALTHY:        {len(healthy)} tasks (pass rate between 0% and 100%)")
    click.echo(f"  TOO_EASY:       {len(too_easy)} tasks (100% pass rate)")
    click.echo(f"  TOO_HARD:       {len(too_hard)} tasks (0% pass rate)")
    click.echo(f"  NEAR_RETRIEVAL: {len(near_retrieval)} tasks (≤2 solutions)")
    click.echo(f"  UNTESTED:       {len(untested)} tasks (no results)")

    # Detailed breakdown of too-hard tasks
    if too_hard and trial_data:
        click.echo(f"\n=== Flagged: TOO_HARD tasks ===\n")
        for tid in too_hard:
            task = all_tasks[tid]
            task_trials = [t for t in trial_data if t["task_id"] == tid]
            constraint_fails = sum(1 for t in task_trials if t["constraint_score"] == 0)
            policy_fails = sum(1 for t in task_trials if t["policy_score"] == 0)
            violations = set()
            for t in task_trials:
                for v in t.get("policy_detail", {}).get("violations", []):
                    violations.add(v)
            no_rec = sum(1 for t in task_trials if not t.get("constraint_detail", {}).get("per_constraint"))

            result = validator.validate_task(task)
            is_nr = tid in near_retrieval
            is_nvr = task.no_valid_recommendation

            reason = []
            if is_nvr:
                reason.append("NVR task")
            if is_nr:
                reason.append(f"near-retrieval ({result.solution_set_size} solutions)")
            if constraint_fails > 0:
                reason.append(f"constraint_fail={constraint_fails}")
            if policy_fails > 0:
                reason.append(f"policy_fail={policy_fails} ({','.join(violations)})")

            label = "NVR" if is_nvr else ("NEAR_RETRIEVAL" if is_nr else "HARD")
            click.echo(f"  [{label}] {tid}: {task.complexity}/{task.reveal_difficulty} — {'; '.join(reason)}")

    if untested:
        click.echo(f"\n=== Untested tasks ===")
        for tid in untested:
            task = all_tasks[tid]
            click.echo(f"  {tid}: {task.complexity}/{task.reveal_difficulty}")

    # Summary
    click.echo(f"\n=== Summary ===")
    if struct_errors > 0:
        click.echo(f"  FIX {struct_errors} structural errors before benchmarking.")
    if untested:
        click.echo(f"  RUN benchmark on {len(untested)} untested tasks.")
    if too_hard and not all(all_tasks[t].no_valid_recommendation for t in too_hard):
        non_nvr_hard = [t for t in too_hard if not all_tasks[t].no_valid_recommendation]
        click.echo(f"  REVIEW {len(non_nvr_hard)} non-NVR tasks with 0% pass rate.")
    if not struct_errors and not untested:
        click.echo(f"  All tasks structurally valid and empirically tested.")


@main.group()
def leaderboard():
    """Validate submissions and render LEADERBOARD.md."""


_ENTRIES = click.option("--entries", default="leaderboard/entries",
                        type=click.Path(exists=True), show_default=True,
                        help="Directory of submission JSON files")
_OUT = click.option("--out", default="LEADERBOARD.md", type=click.Path(),
                    show_default=True, help="Rendered board path")


@leaderboard.command("validate")
@_ENTRIES
def leaderboard_validate(entries: str):
    """Check every entry parses and satisfies the submission rules."""
    from pydantic import ValidationError
    from tau_rec.leaderboard.entry import LeaderboardEntry

    paths = sorted(Path(entries).glob("*.json"))
    if not paths:
        raise click.ClickException(f"no entries found in {entries}")

    failures = 0
    for path in paths:
        try:
            entry = LeaderboardEntry.model_validate_json(path.read_text())
        except ValidationError as exc:
            failures += 1
            click.echo(f"  FAIL {path.name}: {exc.error_count()} error(s)")
            for err in exc.errors():
                click.echo(f"        {'.'.join(str(p) for p in err['loc'])}: {err['msg']}")
            continue
        n_trials = sum(tc.n for tc in entry.per_task.values())
        click.echo(f"  OK   {path.name}: {entry.n_tasks} tasks, {n_trials} trials")

    click.echo(f"\n{len(paths)} entries checked, {failures} invalid.")
    if failures:
        raise SystemExit(1)


@leaderboard.command("render")
@_ENTRIES
@_OUT
@click.option("--check", "check_only", is_flag=True,
              help="Exit 1 if the rendered board is stale instead of writing it")
@click.option("--readme", default="README.md", type=click.Path(),
              help="README whose generated top-N region is kept in sync. "
                   "Skipped if the file has no LEADERBOARD:BEGIN marker.")
def leaderboard_render(entries: str, out: str, check_only: bool, readme: str):
    """Render LEADERBOARD.md from the committed entries."""
    from tau_rec.leaderboard import render as render_mod

    if check_only:
        if render_mod.check(entries, out, readme):
            click.echo(f"{out} is up to date.")
            return
        raise click.ClickException(f"{out} is stale — run `tau-rec leaderboard render`.")

    render_mod.write(entries, out)
    click.echo(f"Wrote {out} from {len(list(Path(entries).glob('*.json')))} entries.")
    if render_mod.sync_readme(entries, readme):
        click.echo(f"Updated the generated top-{render_mod.README_TOP_N} region of {readme}.")


@leaderboard.command("make-entry")
@click.option("--run-dir", required=True, type=click.Path(exists=True),
              help="Run output directory containing traces/")
@click.option("--submission-id", required=True, help="Slug, e.g. g1-my-model")
@click.option("--display-name", required=True, help="Name shown on the board")
@click.option("--submitted-by", required=True)
@click.option("--generation", default="g1", show_default=True)
@click.option("--reasoning-effort", default=None)
@click.option("--pinned/--not-pinned", default=False, show_default=True)
@click.option("--run-date", required=True, help="YYYY-MM-DD")
@click.option("--traces-url", default=None,
              help="Public URL of this run's traces. Becomes the board's Traces link; "
                   "pin it to a fixed path, not a 'latest' alias.")
@click.option("--track", default=None, type=click.Path(exists=True, file_okay=False),
              help="Track directory the run scored against. Defaults to the track named "
                   "in run_manifest.json, else movies-v1 (data/).")
@click.option("--mode", default=None, type=click.Choice(["tools", "memory"]),
              help="Agent configuration. Taken from run_manifest.json when recorded there.")
@click.option("--allow-uneven", is_flag=True, default=False,
              help="Accept a run whose tasks have unequal or fewer than 4 trials, adding "
                   "the uneven_trials caveat. Without it such a run is refused, because "
                   "pass^k is 0 for any task with n < k and the loss would be silent.")
@click.option("--catalog", default=None, type=click.Path(exists=True),
              help="Override the track's catalog.")
@click.option("--tasks", default=None, type=click.Path(exists=True),
              help="Override the track's tasks directory.")
@click.option("--policy", default=None, type=click.Path(exists=True),
              help="Override the track's policy.")
@click.option("--entries", default="leaderboard/entries", show_default=True,
              type=click.Path(exists=True))
def leaderboard_make_entry(run_dir: str, submission_id: str, display_name: str,
                           submitted_by: str, generation: str, reasoning_effort: str | None,
                           pinned: bool, run_date: str, traces_url: str | None,
                           track: str | None, mode: str | None, allow_uneven: bool,
                           catalog: str | None, tasks: str | None, policy: str | None,
                           entries: str):
    """Build a submission entry by re-scoring a run's traces.

    Counts come from re-scoring the traces with the current evaluator, never
    from the run's stored task_results.json — re-derived is what the board
    publishes, so an entry is built the same way it will later be checked.
    """
    import json
    from collections import Counter

    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.conversation import ConversationTrace
    from tau_rec.data_model.task import Task
    from tau_rec.evaluator.evaluator import CombinedEvaluator
    from tau_rec.leaderboard.entry import STANDARD_TRIALS, UNEVEN_TRIALS, LeaderboardEntry
    from tau_rec.leaderboard.hashing import hash_file, hash_task_dir
    from tau_rec.track import find_track, load_track

    root = Path(run_dir)
    trace_dirs = sorted(p for p in root.rglob("traces") if p.is_dir())
    if not trace_dirs:
        raise click.ClickException(f"no traces/ directory under {run_dir}")
    # A top-up pass re-runs a task into a fresh timestamp directory under the
    # same run, writing <task>_trial<N>.json a second time. Counting both
    # copies inflates n and silently corrupts pass^k, so the later pass
    # supersedes the earlier one — timestamp directory names sort in order.
    by_name = {p.name: p for d in trace_dirs for p in sorted(d.glob("*.json"))}
    trace_paths = [by_name[name] for name in sorted(by_name)]
    if not trace_paths:
        raise click.ClickException(f"no trace files under {trace_dirs[0]}")

    # The manifest is what the run actually did; the flags are what someone
    # typed afterwards. Where they disagree the manifest wins, because a
    # mistyped reasoning_effort is indistinguishable from a real one on the
    # board. Runs predating the manifest fall back to the flags.
    manifests = sorted(root.rglob("run_manifest.json"))
    manifest = json.loads(manifests[0].read_text()) if manifests else {}

    # A manifest that records no_tools without a mode is the old ablation: no
    # recommend() tool, so no recommendation could register and every
    # satisfiable task scored 0 by construction. Nothing to publish.
    if manifest.get("no_tools") and "mode" not in manifest:
        raise click.ClickException(
            "this run used --no-tools before memory mode existed; its scores are 0 by "
            "construction and cannot be entered. Re-run with --mode memory."
        )
    mode = manifest.get("mode") or mode or "tools"

    # The track the run scored against: an explicit --track, else the one the
    # manifest names, else movies-v1 (every run that predates tracks).
    if track:
        trk = load_track(track)
    else:
        trk = find_track(manifest.get("track", "movies-v1"))
    catalog = catalog or str(trk.require_catalog())
    tasks = tasks or str(trk.tasks_path)
    # The policy digest an entry records is of the policy its mode ran under.
    policy = policy or str(trk.policy_for(mode))
    spec = trk.spec

    if manifest:
        reasoning_effort = manifest.get("reasoning_effort")
        simulator = manifest.get("simulator_model", "gpt-5-mini")
        drifted = [
            name
            for name, recorded, current in (
                ("policy", manifest.get("policy_sha256"), hash_file(policy)),
                ("tasks", manifest.get("tasks_sha256"), hash_task_dir(tasks)),
                ("catalog", manifest.get("catalog_sha256"), hash_file(catalog)),
            )
            if recorded and recorded != current
        ]
        if drifted:
            raise click.ClickException(
                f"{', '.join(drifted)} changed since this run; its scores are not "
                f"comparable to entries built from the current content. Re-run, or "
                f"submit under a new harness generation."
            )
    else:
        simulator = "gpt-5-mini"
        click.echo("  no run_manifest.json: reasoning_effort/digests taken from flags.")

    cat = Catalog.from_json(catalog, spec)
    task_map = {p.stem: Task.model_validate_json(p.read_text())
                for p in sorted(Path(tasks).glob("*.json"))}
    evaluator = CombinedEvaluator(cat, spec)

    # Memory mode has one tool, so four of the seven policies are unsatisfiable
    # by construction; it scores constraints only, over the satisfiable tasks.
    # NVR tasks are reported beside the entry, never inside its counts.
    per_task: dict[str, dict[str, int]] = {}
    nvr_abstained: dict[str, dict[str, int]] = {}
    models: set[str] = set()
    for path in trace_paths:
        trace = ConversationTrace.model_validate_json(path.read_text())
        models.add(trace.model)
        task = task_map[trace.task_id]
        result = evaluator.evaluate(task=task, trace=trace)
        if mode == "memory":
            if task.no_valid_recommendation:
                row = nvr_abstained.setdefault(trace.task_id, {"n": 0, "abstained": 0})
                row["n"] += 1
                row["abstained"] += 1 if result.constraint_score == 1.0 else 0
                continue
            success = result.constraint_score == 1.0
        else:
            success = result.primary_reward == 1.0
        row = per_task.setdefault(trace.task_id, {"n": 0, "c": 0})
        row["n"] += 1
        row["c"] += 1 if success else 0

    if len(models) != 1:
        raise click.ClickException(f"traces mix model ids: {sorted(models)}")

    expected_task_ids = {
        task_id for task_id, task in task_map.items()
        if mode != "memory" or not task.no_valid_recommendation
    }
    missing_task_ids = sorted(expected_task_ids - set(per_task))
    if missing_task_ids:
        raise click.ClickException(
            f"run is incomplete: missing traces for {len(missing_task_ids)} expected "
            f"task(s): {missing_task_ids[:8]}"
            f"{'...' if len(missing_task_ids) > 8 else ''}. Resume the run before "
            "building a leaderboard entry."
        )

    # A trial that died to a rate limit wrote no trace, and pass^k is 0 for any
    # task with n < k, so a short task silently drags the entry down. Say so
    # or refuse.
    n_dist = Counter(row["n"] for row in per_task.values())
    caveats: list[str] = []
    if len(n_dist) > 1 or min(n_dist) < STANDARD_TRIALS:
        short = sorted(tid for tid, row in per_task.items() if row["n"] < STANDARD_TRIALS)
        summary = ", ".join(f"n={n}: {k} task(s)" for n, k in sorted(n_dist.items()))
        if not allow_uneven:
            raise click.ClickException(
                f"trials per task are uneven or below {STANDARD_TRIALS} ({summary}); "
                f"short tasks: {short[:8]}{'...' if len(short) > 8 else ''}. Fill them with "
                f"`tau-rec run --resume {run_dir}` or pass --allow-uneven to publish with a caveat."
            )
        caveats.append(UNEVEN_TRIALS)
        click.echo(f"  uneven trials accepted ({summary}); entry carries `{UNEVEN_TRIALS}`.")

    entry = LeaderboardEntry(
        track=trk.id,
        mode=mode,
        caveats=caveats,
        submission_id=submission_id,
        display_name=display_name,
        submitted_by=submitted_by,
        model_id=models.pop(),
        pinned=pinned,
        run_date=run_date,
        reasoning_effort=reasoning_effort,
        simulator_model=simulator,
        harness_generation=generation,
        tau_rec_version=__version__,
        trials_per_task=max(row["n"] for row in per_task.values()),
        policy_sha256=hash_file(policy),
        tasks_sha256=hash_task_dir(tasks),
        catalog_sha256=hash_file(catalog),
        per_task=per_task,
        traces_url=traces_url,
    )

    out_path = Path(entries) / f"{submission_id}.json"
    out_path.write_text(json.dumps(json.loads(entry.model_dump_json()), indent=2) + "\n")
    total = sum(row["n"] for row in per_task.values())
    solved = sum(row["c"] for row in per_task.values())
    click.echo(f"Wrote {out_path}: {len(per_task)} tasks, {total} trials, {solved} solved "
               f"(track {trk.id}, mode {mode}).")
    if nvr_abstained:
        n = sum(r["n"] for r in nvr_abstained.values())
        a = sum(r["abstained"] for r in nvr_abstained.values())
        click.echo(f"  NVR tasks (not in the entry): abstained on {a}/{n} trials.")
    if entry.caveats:
        click.echo(f"  caveats: {', '.join(entry.caveats)}")
