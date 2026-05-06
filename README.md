# τ-Rec

A verifiable benchmark for LLM-based conversational recommender systems.

τ-Rec measures whether an agent-under-test can hold a multi-turn conversation with a simulated user, use catalog tools to gather information, respect a written policy, and ultimately recommend a movie that satisfies all of the user's constraints. Success is checked programmatically against the catalog — not by an LLM judge — so scores are reproducible and cheap to compute.

## Highlights

- **Verifiable rewards.** Each task declares constraints as structured predicates (`runtime <= 120`, `genres contains Comedy`, …) that are evaluated directly against the catalog.
- **Reveal-tagged user simulation.** Constraints carry `volunteer` / `on_ask` / `hidden` tags so the simulated user leaks information at a controlled rate — `hidden` constraints are never stated and must be inferred from rejections.
- **Policy compliance scoring.** A natural-language policy (`data/policy.md`) is shown to the agent; per-task `policy_flags` enable programmatic checks for watch-history, availability, sponsored-content disclosure, age gating, and more.
- **`pass^k` metric.** Uses the unbiased combinatoric estimator `C(c,k)/C(n,k)` across multiple trials per task — rewarding agents that succeed consistently, not just once.
- **Real catalog, stratified tasks.** 153-movie TMDB catalog with 60 tasks stratified across `complexity` × `reveal_difficulty` cells.
- **Parallel execution.** `asyncio.gather` + semaphore runs trials concurrently (default 16).
- **Ablation support.** `--no-tools` mode disables all tools so you can measure how much the agent is leaning on retrieval vs. memorization.
- **Model-agnostic.** Any LiteLLM-supported model works as the agent or the simulator.

## Install

Requires Python 3.12+. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Usage

Provide credentials for whichever model providers you use via the standard environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …) — LiteLLM routes based on the model string.

You can put these in a `.env` file at the repo root; the CLI loads it automatically on startup (shell env vars take precedence over `.env`). See `.env.example` for the expected keys.

### Validate tasks

Checks every task has at least one satisfying movie in the catalog (or zero, for `no_valid_recommendation` tasks):

```bash
uv run tau-rec validate \
  --catalog data/catalog.json \
  --tasks data/tasks
```

### Run the benchmark

```bash
uv run tau-rec run \
  --model anthropic/claude-opus-4-7 \
  --catalog data/catalog.json \
  --tasks data/tasks \
  --policy data/policy.md \
  --trials 16 \
  --output out/
```

Options:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--model` | — | Agent-under-test (LiteLLM model string) |
| `--simulator-model` | `gemini/gemini-2.5-pro` | Model powering the user simulator |
| `--trials` | `16` | Independent trials per task |
| `--max-turns` | `20` | Hard cap on agent↔user turns per trial |
| `--concurrency` | `16` | Trials to run in parallel |
| `--no-tools` | off | Ablation mode: disable all tools |
| `--tasks-limit` | — | Run only the first N tasks (smoke test) |
| `--output` | — | Directory for `trial_results.json`, `task_results.json`, and `traces/` |

Each trial's full `ConversationTrace` (messages + tool calls interleaved) is written to `<output>/traces/<task_id>_trial<N>.json` for debugging.

### Report metrics

`pass^1`, `pass^2`, `pass^4` with 95% bootstrap confidence intervals. Pass `--trials` and `--tasks` for per-dimension breakdowns (complexity, reveal_difficulty), policy-violation frequency, and efficiency stats:

```bash
uv run tau-rec report \
  --results out/task_results.json \
  --trials out/trial_results.json \
  --tasks data/tasks
```

## How a trial works

```
Orchestrator
   ├── Agent-under-test  ── tool calls ──▶  ToolKit (catalog search, metadata,
   │                                         availability, user history)
   │
   └── User Simulator    ── persona + reveal-tagged constraints
                            emits ###ACCEPTED### / ###REJECTED###
```

Each conversation is seeded with a fixed agent greeting so the simulator opens by stating the user's request. The orchestrator then alternates turns, executes tool calls inline, and records every message and tool invocation into a `ConversationTrace`. Tools available to the agent: `search_catalog`, `get_metadata`, `check_availability`, `get_user_history`, `check_content_preference`, and `recommend(item_id)`. **A recommendation is registered only when the agent calls `recommend(item_id)`** — naming a title in chat is not enough. Policy 7 makes calling this tool mandatory.

Once the simulator emits a stop token (or `--max-turns` is reached), the trace is scored:

- **Constraint score** — 1.0 if the final recommendation satisfies every task constraint, else 0.0 (inverted for `no_valid_recommendation` tasks).
- **Policy score** — 1.0 if no flag is violated, else 0.0. Individual violations are also recorded for failure-mode analysis.

The headline reward is `constraint × policy`.

## Task format

Each file under `data/tasks/` defines one scenario:

```json
{
  "id": "task_002",
  "constraints": [
    {"constraint": {"field": "runtime", "op": "<=", "value": 120}, "reveal": "volunteer"},
    {"constraint": {"field": "genres", "op": "contains", "value": "Comedy"}, "reveal": "on_ask"}
  ],
  "persona": "You are a tired parent looking for something light after the kids go to bed.",
  "soft_preferences": ["prefers feel-good endings"],
  "policy_flags": ["watch_history", "availability"],
  "no_valid_recommendation": false,
  "complexity": "simple",
  "reveal_difficulty": "mixed",
  "user_id": "user_1",
  "user_history": {"user_1": {"watched": ["tmdb_123"], "ratings": {}}}
}
```

Supported constraint operators: `<=`, `>=`, `==`, `!=`, `contains`, `contains_any`, `not_contains`, `in`.

Supported policy flags (each implemented as `_check_<flag>` in `evaluator/policy.py`): `watch_history`, `availability`, `sponsored`, `age_restricted`, `single_recommendation`, `transparency`, `recommend_tool`.

## Catalog

`data/catalog.json` ships with 153 real TMDB movies (streaming-service names are normalized; rating-0 / vote-count-0 entries are filtered). To rebuild or extend the catalog, use `tau_rec.catalog.pipeline.TMDBPipeline` with a TMDB API key.

## Answer key

`data/answers.json` is a pre-computed reference: for each task, it lists every catalog movie that satisfies all constraints, plus the subset that is actually streamable on the task's `user_services`. NVR tasks report empty lists.

```json
"task_012": {
  "no_valid_recommendation": false,
  "user_services": ["HBO Max", "Paramount+"],
  "constraint_solutions": ["tmdb_467905", ...],          // 32 movies
  "reachable_solutions": ["tmdb_991494"],                 // 1 streamable
  "reachable_solutions_with_titles": [
    {"id": "tmdb_991494",
     "title": "The SpongeBob Movie: Search for SquarePants"}
  ]
}
```

**The answer key is not consumed by the evaluator today** — it's an analysis artifact. Use it to audit which tasks are effectively unsolvable given `user_services`, to spot-check agent failures, or to gauge task difficulty. It is regenerated by running `CatalogValidator` over all tasks; re-run any time the catalog or task constraints change.

## Repository layout

```
src/tau_rec/
  agents/         # BaseAgent + LiteLLMAgent (the agent-under-test)
  simulator/      # UserSimulator (persona- and reveal-driven)
  orchestrator/   # Conversation loop, trace construction
  environment/    # ToolKit exposed to the agent
  catalog/        # BM25 search, TMDB pipeline, task validator
  evaluator/      # constraint, policy, efficiency scorers
  metrics/        # pass^k + bootstrap CI
  data_model/     # Pydantic schemas
  cli.py          # `tau-rec` entry point
data/
  catalog.json
  policy.md
  tasks/*.json
  answers.json   # pre-computed solution sets per task (analysis-only)
tests/            # pytest suite (asyncio-auto)
```

## Development

```bash
uv run pytest                                      # full suite
uv run pytest tests/test_orchestrator.py -k name   # single test
```

## Citation

If you use τ-Rec in your work, please cite this repository. A peer-reviewed
publication will be added here when available.

```bibtex
@misc{taurec2026,
  author = {Narasimhan, Bharath Sivaram and Narasimhan, Karthik R},
  title  = {{τ-Rec}: A Verifiable Benchmark for Agentic Recommender Systems},
  year   = 2026,
  url    = {https://github.com/nbharaths/tau-rec}
}
```
