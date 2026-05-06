from __future__ import annotations
import asyncio
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


from tau_rec.play import play as _play_cmd
main.add_command(_play_cmd, name="play")

@main.command()
@click.option("--catalog", required=True, type=click.Path(exists=True), help="Path to catalog JSON")
@click.option("--tasks", required=True, type=click.Path(exists=True), help="Path to tasks directory")
@click.option("--baseline", type=click.Path(), default=None,
              help="Snapshot file path. If it doesn't exist, write the current "
                   "task signatures there. If it exists, diff current vs. snapshot "
                   "and report any drift. Use to guard against silent regressions "
                   "when editing tasks.")
@click.option("--strict", is_flag=True, default=False,
              help="Promote warnings to errors (e.g., near-retrieval, untestable "
                   "policy flags). Use in CI to keep task quality from drifting.")
def validate(catalog: str, tasks: str, baseline: str | None, strict: bool):
    """Validate tasks against catalog."""
    import json as _json
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import Task
    from tau_rec.catalog.validator import CatalogValidator, task_signature, grid_summary

    cat = Catalog.from_json(catalog)
    validator = CatalogValidator(cat)

    task_dir = Path(tasks)
    task_files = sorted(task_dir.glob("*.json"))
    click.echo(f"Validating {len(task_files)} tasks against catalog ({len(cat)} movies)...\n")

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
            baseline_path.write_text(_json.dumps(signatures, indent=2, sort_keys=True) + "\n")
            click.echo(f"\nBaseline snapshot written to {baseline_path}.")
        else:
            prev = _json.loads(baseline_path.read_text())
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
    all_ids = set(prev) | set(curr)
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
@click.option("--catalog", required=True, type=click.Path(exists=True), help="Path to catalog JSON")
@click.option("--tasks", required=True, type=click.Path(exists=True), help="Path to tasks directory")
@click.option("--policy", required=True, type=click.Path(exists=True), help="Path to policy.md")
@click.option("--trials", default=16, help="Number of trials per task")
@click.option("--output", required=True, type=click.Path(), help="Output directory for results")
@click.option("--simulator-model", default="gpt-5-mini", help="Simulator LLM model")
@click.option("--max-turns", default=20, help="Maximum turns per conversation")
@click.option("--concurrency", default=16, help="Max trials running in parallel")
@click.option("--no-tools", is_flag=True, default=False, help="Disable all tools (ablation mode)")
@click.option("--tasks-limit", default=None, type=int, help="Only run the first N tasks (for quick smoke tests)")
@click.option("--tasks-filter", default=None, type=str, help="Comma-separated list of task IDs to run (e.g. task_002,task_004)")
@click.option("--dry-run", is_flag=True, default=False,
              help="Estimate experiment cost: runs a small calibration sample, measures tokens, extrapolates. No artifacts written.")
@click.option("--dry-run-samples", default=3, type=int,
              help="Number of trials to run for calibration under --dry-run (default 3).")
@click.option("--reasoning-effort", default=None, type=click.Choice(["minimal", "low", "medium", "high", "xhigh"]),
              help="Reasoning effort for the agent model. Passed through to litellm. DeepSeek v4 maps low/medium→high and xhigh→max in thinking mode.")
def run(model: str, catalog: str, tasks: str, policy: str, trials: int, output: str, simulator_model: str, max_turns: int, concurrency: int, no_tools: bool, tasks_limit: int | None, tasks_filter: str | None, dry_run: bool, dry_run_samples: int, reasoning_effort: str | None):
    """Run benchmark evaluation."""
    if dry_run:
        asyncio.run(_dry_run_estimate(model, catalog, tasks, policy, simulator_model, max_turns, no_tools, tasks_limit, trials, dry_run_samples))
        return
    asyncio.run(_run_benchmark(model, catalog, tasks, policy, trials, output, simulator_model, max_turns, concurrency, no_tools, tasks_limit, tasks_filter, reasoning_effort))


async def _dry_run_estimate(
    model: str, catalog_path: str, tasks_path: str, policy_path: str,
    simulator_model: str, max_turns: int, no_tools: bool, tasks_limit: int | None,
    full_trials: int, sample_size: int,
) -> None:
    """Run `sample_size` trials (one per randomly selected task) to measure token
    usage, then extrapolate to the full experiment and print a cost estimate.
    No results or traces are written to disk."""
    import random
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import Task
    from tau_rec.catalog.search import CatalogSearch
    from tau_rec.environment.tools import ToolKit
    from tau_rec.agents.litellm_agent import LiteLLMAgent, AGENT_SYSTEM_PROMPT_TEMPLATE, AGENT_NO_TOOLS_SYSTEM_PROMPT_TEMPLATE
    from tau_rec.simulator.user_simulator import UserSimulator
    from tau_rec.orchestrator.orchestrator import Orchestrator
    from tau_rec import cost as costlib

    catalog = Catalog.from_json(catalog_path)
    search = CatalogSearch(catalog)
    policy_text = Path(policy_path).read_text()
    task_files = sorted(Path(tasks_path).glob("*.json"))
    all_tasks = [Task.from_json(str(tf)) for tf in task_files]
    if tasks_limit:
        all_tasks = all_tasks[:tasks_limit]

    total_trials_full = len(all_tasks) * full_trials
    sample_size = min(sample_size, len(all_tasks))
    rng = random.Random("dry-run")
    sampled = rng.sample(all_tasks, sample_size)

    click.echo(f"Dry run: calibrating on {sample_size} trials...")
    click.echo(f"  agent:     {model}")
    click.echo(f"  simulator: {simulator_model}")

    costlib.reset()

    completed = 0

    async def run_one(task):
        nonlocal completed
        toolkit = ToolKit(catalog=catalog, search=search, user_histories=task.user_history or {})
        user_id_instruction = f"The current user's ID is {task.user_id}. Use this when checking their watch history." if task.user_id else ""
        prompt_template = AGENT_NO_TOOLS_SYSTEM_PROMPT_TEMPLATE if no_tools else AGENT_SYSTEM_PROMPT_TEMPLATE
        system_prompt = prompt_template.format(policy=policy_text, user_id_instruction=user_id_instruction)
        tool_defs = [] if no_tools else toolkit.tool_definitions()
        agent = LiteLLMAgent(model=model, system_prompt=system_prompt, tool_definitions=tool_defs)
        simulator = UserSimulator(model=simulator_model, task=task)
        orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=max_turns)
        try:
            await orch.run(task_id=task.id, model=model, trial=0)
            completed += 1
            click.echo(f"  [{completed}/{sample_size}] {task.id}")
        except Exception as e:
            completed += 1
            click.echo(f"  [{completed}/{sample_size}] {task.id} FAILED: {type(e).__name__}: {e!s:.80s}")

    await asyncio.gather(*(run_one(t) for t in sampled))

    est = costlib.estimate_cost(model, simulator_model, sample_size, total_trials_full)

    click.echo()
    click.echo("=== Per-trial averages (from calibration) ===")
    p = est["per_trial"]
    click.echo(f"  agent     input={p['agent_input']:>8.0f}  output={p['agent_output']:>8.0f}  tokens")
    click.echo(f"  simulator input={p['sim_input']:>8.0f}  output={p['sim_output']:>8.0f}  tokens")

    click.echo()
    click.echo(f"=== Full experiment extrapolation ({len(all_tasks)} tasks × {full_trials} trials = {total_trials_full} trials) ===")
    t = est["estimated_total_tokens"]
    click.echo(f"  agent     input={t['agent_input']:>10.0f}  output={t['agent_output']:>10.0f}  tokens")
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

    click.echo()
    click.echo(
        f"Note: estimate is based on {sample_size} sampled trial(s). "
        f"Actual cost can vary ±30% depending on task difficulty and conversation length."
    )


async def _run_benchmark(
    model: str, catalog_path: str, tasks_path: str, policy_path: str,
    trials: int, output_dir: str, simulator_model: str, max_turns: int,
    concurrency: int, no_tools: bool = False, tasks_limit: int | None = None,
    tasks_filter: str | None = None, reasoning_effort: str | None = None,
):
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import Task
    from tau_rec.catalog.search import CatalogSearch
    from tau_rec.environment.tools import ToolKit
    from tau_rec.agents.litellm_agent import LiteLLMAgent, AGENT_SYSTEM_PROMPT_TEMPLATE, AGENT_NO_TOOLS_SYSTEM_PROMPT_TEMPLATE
    from tau_rec.simulator.user_simulator import UserSimulator
    from tau_rec.orchestrator.orchestrator import Orchestrator
    from tau_rec.evaluator.evaluator import CombinedEvaluator
    from tau_rec.metrics.pass_k import pass_at_k, aggregate_pass_k

    catalog = Catalog.from_json(catalog_path)
    search = CatalogSearch(catalog)
    evaluator = CombinedEvaluator(catalog)
    out = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out / timestamp
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    policy_text = Path(policy_path).read_text()

    task_dir = Path(tasks_path)
    task_files = sorted(task_dir.glob("*.json"))
    all_tasks = [Task.from_json(str(tf)) for tf in task_files]
    if tasks_filter:
        allowed = set(tasks_filter.split(","))
        all_tasks = [t for t in all_tasks if t.id in allowed]
    elif tasks_limit:
        all_tasks = all_tasks[:tasks_limit]

    mode = " [NO-TOOLS ABLATION]" if no_tools else ""
    click.echo(f"Running {model} on {len(all_tasks)} tasks x {trials} trials (concurrency={concurrency}){mode}...")

    sem = asyncio.Semaphore(concurrency)
    total_trials = len(all_tasks) * trials
    completed_count = 0

    failed_count = 0

    async def run_trial(task, trial_idx, toolkit, system_prompt):
        nonlocal completed_count, failed_count
        async with sem:
            try:
                tool_defs = [] if no_tools else toolkit.tool_definitions()
                agent = LiteLLMAgent(model=model, system_prompt=system_prompt, tool_definitions=tool_defs, reasoning_effort=reasoning_effort)
                simulator = UserSimulator(model=simulator_model, task=task)
                orch = Orchestrator(agent=agent, simulator=simulator, toolkit=toolkit, max_turns=max_turns)
                trace = await orch.run(task_id=task.id, model=model, trial=trial_idx)
                result = evaluator.evaluate(task, trace)
                trace_doc = trace.model_dump(mode="json")
                trace_doc["evaluation"] = result.model_dump(mode="json")
                (traces_dir / f"{task.id}_trial{trial_idx}.json").write_text(
                    json.dumps(trace_doc, indent=2, default=str)
                )
                completed_count += 1
                pct = completed_count / total_trials
                bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
                click.echo(f"  [{bar}] {pct:5.1%} ({completed_count}/{total_trials})  {task.id} trial {trial_idx}: primary={result.primary_reward}")
                return task.id, result
            except Exception as e:
                failed_count += 1
                click.echo(f"  FAIL  {task.id} trial {trial_idx}: {type(e).__name__}: {e!s:.120s}")
                return None

    coros = []
    for task in all_tasks:
        user_history = task.user_history or {}
        toolkit = ToolKit(catalog=catalog, search=search, user_histories=user_history)

        user_id_instruction = ""
        if task.user_id:
            user_id_instruction = f"The current user's ID is {task.user_id}. Use this when checking their watch history."

        prompt_template = AGENT_NO_TOOLS_SYSTEM_PROMPT_TEMPLATE if no_tools else AGENT_SYSTEM_PROMPT_TEMPLATE
        system_prompt = prompt_template.format(
            policy=policy_text, user_id_instruction=user_id_instruction,
        )

        for trial_idx in range(trials):
            coros.append(run_trial(task, trial_idx, toolkit, system_prompt))

    completed = await asyncio.gather(*coros)

    task_results = {t.id: {"n": 0, "c": 0} for t in all_tasks}
    all_trial_results = []
    for item in completed:
        if item is None:
            continue
        task_id, result = item
        task_results[task_id]["n"] += 1
        all_trial_results.append(result.model_dump())
        if result.primary_reward >= 1.0 - 1e-6:
            task_results[task_id]["c"] += 1

    if failed_count:
        click.echo(f"\n{failed_count} trial(s) failed. Results computed from {len(all_trial_results)}/{total_trials} successful trials.")

    for k in [1, 2, 4]:
        score = aggregate_pass_k(task_results, k=k)
        click.echo(f"pass^{k} = {score:.3f}")

    (run_dir / "trial_results.json").write_text(json.dumps(all_trial_results, indent=2, default=str))
    (run_dir / "task_results.json").write_text(json.dumps(task_results, indent=2))
    click.echo(f"\nRun artifacts saved to {run_dir}")
    import os as _os
    _os._exit(0)  # litellm httpx connection pools don't close cleanly; force exit after files are written


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
@click.option("--catalog", required=True, type=click.Path(exists=True), help="Path to catalog JSON")
@click.option("--tasks", required=True, type=click.Path(exists=True), help="Path to tasks directory")
@click.option("--results", default=None, type=click.Path(exists=True), help="Path to task_results.json (for empirical checks)")
@click.option("--trials", default=None, type=click.Path(exists=True), help="Path to trial_results.json (for failure analysis)")
def health(catalog: str, tasks: str, results: str | None, trials: str | None):
    """Run structural validation + empirical health check on tasks."""
    from tau_rec.data_model.catalog import Catalog
    from tau_rec.data_model.task import Task
    from tau_rec.catalog.validator import CatalogValidator

    cat = Catalog.from_json(catalog)
    validator = CatalogValidator(cat)

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
