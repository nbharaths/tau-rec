# τ-Rec (Tau-Rec)

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/nbharaths/tau-rec)
[![Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2606.10156)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3773078.3831847-blue)](https://doi.org/10.1145/3773078.3831847)
[![arXiv](https://img.shields.io/badge/arXiv-2606.10156-b31b1b)](https://arxiv.org/abs/2606.10156)
[![τ-Rec-Books on arXiv](https://img.shields.io/badge/%CF%84--Rec--Books-arXiv%202611.04459-b31b1b)](https://arxiv.org/abs/2611.04459)
[![τ-Rec-Books on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-%CF%84--Rec--Books-blue)](https://huggingface.co/datasets/nbharaths/tau-rec-books)

A verifiable benchmark for LLM-based conversational recommender systems.

τ-Rec measures whether an agent-under-test can hold a multi-turn conversation with a simulated user, use catalog tools to gather information, respect a written policy, and ultimately recommend a movie that satisfies all of the user's constraints. Success is checked programmatically against the catalog — not by an LLM judge — so scores are reproducible and cheap to compute.

Published at [ACM RecSys 2026](https://doi.org/10.1145/3773078.3831847) (Reproducibility and Resource Track).

**New in v2.0:** a controlled books track, [`books-wikidata-v2`](#books-track-books-wikidata-v2): 20,000 Wikidata works in nested 5k/10k/20k catalogs, 150 request families in matched volunteer/mixed/hidden variants, a deterministic rule-based user, and keyword versus structured retrieval.

## Highlights

- **Verifiable rewards.** Each task declares constraints as structured predicates (`runtime <= 120`, `genres contains Comedy`, …) that are evaluated directly against the catalog.
- **Reveal-tagged user simulation.** Constraints carry `volunteer` / `on_ask` / `hidden` tags so the simulated user leaks information at a controlled rate — `hidden` constraints are never stated and must be inferred from rejections.
- **Policy compliance scoring.** A natural-language policy (`data/policy.md`) is shown to the agent; per-task `policy_flags` enable programmatic checks for watch-history, availability, sponsored-content disclosure, age gating, and more.
- **`pass^k` metric.** Uses the unbiased combinatoric estimator `C(c,k)/C(n,k)` across multiple trials per task — rewarding agents that succeed consistently, not just once.
- **Real catalog, stratified tasks.** 153-movie TMDB catalog with 60 tasks stratified across `complexity` × `reveal_difficulty` cells.
- **Model-agnostic.** Any LiteLLM-supported model works as the agent or the simulator.

## Benchmark comparison

Comparison of CRS evaluation resources from the paper. **✅** = fully
supported; **◐** = partially supported.

| Resource | VR | MT | TU | PC | FC | `pass^k` | HI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CRSLab | — | ◐ | — | — | — | — | — |
| iEvaLM | — | ✅ | — | — | — | — | ◐ |
| UserSimCRS v2 | ◐ | ✅ | — | — | — | — | ◐ |
| AgentRecBench | ◐ | — | ✅ | — | — | — | — |
| InteRecAgent | — | ✅ | ✅ | — | — | — | ◐ |
| MATCHA | — | ✅ | ✅ | ◐ | — | — | — |
| **τ-Rec** | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** |

VR = verifiable rewards; MT = multi-turn dialogue; TU = tool use; PC = policy
checks; FC = fresh catalog; HI = hidden-intent simulation. The table compares
the movies track; the books track trades the post-cutoff catalog for a CC0
Wikidata catalog that can be redistributed and scaled.

## Leaderboard

**[LEADERBOARD.md](LEADERBOARD.md)** — current standings across two harness generations.

<!-- LEADERBOARD:BEGIN — generated, do not edit by hand -->

`g1`, the current harness (top 3 of 16):

| # | Model | pass^1 | pass^2 | pass^4 |
|---|-------|--------|--------|--------|
| 1 | GPT-5.4 (medium thinking) | 0.533 | 0.450 | **0.400** |
| 2 | DeepSeek V4 Flash (high thinking) | 0.604 | 0.494 | 0.383 |
| 3 | GPT-5.6 Sol (medium thinking) | 0.571 | 0.442 | 0.350 |

`g0`, the paper cohort (top 3 of 9):

| # | Model | pass^1 | pass^2 | pass^4 |
|---|-------|--------|--------|--------|
| 1 | DeepSeek V4 Flash (high thinking) | 0.560 | 0.461 | **0.383** |
| 2 | DeepSeek V4 Flash (max thinking) | 0.571 | 0.461 | 0.350 |
| 3 | GPT-5.4 (medium thinking) | 0.551 | 0.450 | 0.350 |

<!-- LEADERBOARD:END -->

**The two tables are not comparable:** `g0` and `g1` use different policy
prompts. [GENERATIONS.md](leaderboard/GENERATIONS.md) records the differences.

The board is generated from raw per-task `{n, c}` counts, and full traces are
attached to the [v1.3.0 release](https://github.com/nbharaths/tau-rec/releases/tag/v1.3.0).

To submit a run, see **[leaderboard/CONTRIBUTING.md](leaderboard/CONTRIBUTING.md)**.

## Dataset

The catalog, tasks, and answer key are hosted on Hugging Face:

```python
from datasets import load_dataset

catalog = load_dataset("nbharaths/tau-rec", "catalog")
tasks   = load_dataset("nbharaths/tau-rec", "tasks")
answers = load_dataset("nbharaths/tau-rec", "answers")
```

The JSON files in `data/` are the same data — use whichever is more convenient.

## Books track (`books-wikidata-v2`)

A controlled books benchmark built on τ-Rec's harness, described in
[τ-Rec-Books](https://arxiv.org/abs/2611.04459). The movie track above is unchanged.

- **Catalog.** 20,000 literary works derived from Wikidata (CC0), ordered by a
  seeded hash; the first 5,000 and 10,000 form nested smaller catalogs.
- **Tasks.** 150 request families with two to five hard requirements on genre,
  language, country of origin, and publication year. Each family has three
  variants with identical requirements: the reader states everything up front
  (`volunteer`), states one requirement and reveals the rest only when asked
  (`mixed`), or also withholds some (`hidden`). Fifteen families have no answer
  in the catalog; the agent should abstain.
- **Rule-based user.** A deterministic disclosure controller answers field
  questions and gives accept/reject verdicts; no LLM runs on the user side, so
  repeated trials measure the agent. LLM users are available as a sensitivity
  condition (`--simulator-mode llm --simulator-model <model>`).
- **Two retrieval interfaces.** `keyword` (per-field BM25, top 20) and
  `structured` (conjunctive filters, pages of 20).

Run one agent under the reportable protocol (primary panel, 20k, rule-based user,
20 turns, 600 s of agent time, four trials per task):

```bash
uv run python scripts/run_books_protocol.py \
  --model openrouter/deepseek/deepseek-v4.1-flash --provider-order deepinfra/fp8 \
  --retrieval-mode structured --reasoning-effort high --max-provider-cost-usd 20 --execute
```

Without `--execute` it prints the `tau-rec run` command instead, and
`scripts/report_books_run.py --run-dir <run>` reports a finished run. The track
[README](data/tracks/books-wikidata-v2/README.md) covers files, validation, and
the byte-for-byte rebuild from the source snapshot;
[`analysis/books-controlled/`](analysis/books-controlled/README.md) reproduces
the reported run results from the traces attached to the release.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Provide credentials for whichever model providers you use via the standard environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …) — LiteLLM routes based on the model string.

The CLI also reads a root `.env`; see `.env.example`.

### Validate tasks

Checks every task has at least one satisfying movie in the catalog (or zero, for `no_valid_recommendation` tasks):

```bash
uv run tau-rec validate --catalog data/catalog.json --tasks data/tasks
```

### Run the benchmark

```bash
uv run tau-rec run \
  --model anthropic/claude-opus-4-7 \
  --catalog data/catalog.json \
  --tasks data/tasks \
  --policy data/policy.md \
  --trials 4 \
  --output out/
```

Use `--tasks-limit N` for a smoke test, or add `--dry-run` to estimate the cost
of a full run. See `uv run tau-rec run --help` for all options.

> **`--no-tools` is deprecated and now means `--mode memory`.** Runs made with
> the old flag had no `recommend()` tool, so they cannot receive credit.

Artifacts are written under `<output>/<timestamp>/`, including traces,
per-trial results, task counts, the run manifest, and token/cost usage.

### Report metrics

`pass^1`, `pass^2`, `pass^4` with 95% bootstrap confidence intervals. Pass `--trials` and `--tasks` for per-dimension breakdowns (complexity, reveal_difficulty), policy-violation frequency, and efficiency stats:

```bash
uv run tau-rec report \
  --results out/TIMESTAMP/task_results.json \
  --trials out/TIMESTAMP/trial_results.json \
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

The orchestrator records messages and tool calls in chronological order. The
agent can search the catalog, inspect metadata, check availability and user
history, and check content preferences. Calling `recommend(item_id)` registers
the final recommendation; calling it without an item ID records abstention.
Either ends the trial. Simulator `###ACCEPTED###` / `###REJECTED###` tokens are
feedback, not stop signals.

The trace is then scored:

- **Constraint score** — 1.0 if the final recommendation satisfies every task constraint, else 0.0 (inverted for `no_valid_recommendation` tasks).
- **Policy score** — 1.0 if no flag is violated, else 0.0. Individual violations are also recorded for failure-mode analysis.

The headline reward is `constraint × policy`.

## Task format

Each file under `data/tasks/` defines one scenario:

```json
{
  "id": "task_002",
  "constraints": [
    {"constraint": {"field": "genres", "op": "contains", "value": "Horror"}, "reveal": "volunteer"},
    {"constraint": {"field": "runtime", "op": "<=", "value": 120}, "reveal": "volunteer"}
  ],
  "persona": "You are a retired literature professor. You value storytelling craft above all else. You speak in complete, thoughtful sentences and aren't in a rush.",
  "soft_preferences": ["likes supernatural horror over slashers"],
  "policy_flags": ["recommend_tool", "availability"],
  "no_valid_recommendation": false,
  "complexity": "simple",
  "reveal_difficulty": "volunteer",
  "user_id": "user_1",
  "user_history": {"user_1": {"watched": [], "ratings": {}}},
  "user_services": ["Hulu"]
}
```

Supported constraint operators: `<=`, `>=`, `==`, `!=`, `contains`, `contains_any`, `not_contains`, `in`.

Supported policy flags (each implemented as `_check_<flag>` in `evaluator/policy.py`): `watch_history`, `availability`, `sponsored`, `age_restricted`, `single_recommendation`, `transparency`, `recommend_tool`.

## Catalog

`data/catalog.json` ships with 153 real TMDB movies (streaming-service names are normalized; rating-0 / vote-count-0 entries are filtered). To rebuild or extend the catalog, use `tau_rec.catalog.pipeline.TMDBPipeline` with a TMDB API key.

## Answer key

`data/answers.json` lists every constraint-satisfying and streamable item for
each task. It is an analysis artifact; the evaluator scores the final
recommendation directly against the catalog and does not read this file.

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
  leaderboard/    # submission schema, content digests, board renderer
  data_model/     # Pydantic schemas
  cli.py          # `tau-rec` entry point
data/
  catalog.json
  policy.md
  tasks/*.json
  answers.json   # pre-computed solution sets per task (analysis-only)
  tracks/
    books-wikidata-v2/  # books track: catalog, tasks, answer key, manifests
analysis/
  books-controlled/     # books analysis code and the rule-based user's protocol
  books-wikidata-v2/    # books plans, panels, and results
scripts/                # catalog/task rebuild, validators, protocol runner
leaderboard/
  entries/*.json # one submission each; raw per-task {n, c} only
  GENERATIONS.md # what makes two entries comparable
LEADERBOARD.md   # generated — do not edit by hand
tests/            # pytest suite (asyncio-auto)
```

## Development

```bash
uv sync --extra dev                                # test dependencies (pytest)
uv run pytest                                      # full suite
uv run pytest tests/test_orchestrator.py -k name   # single test
```

## Citation

If you use τ-Rec in your work, please cite the τ-Rec paper. If you use the books
track (`books-wikidata-v2`), please also cite the τ-Rec-Books paper.

```bibtex
@inproceedings{narasimhan2026taurec,
  author    = {Narasimhan, Bharath Sivaram and Narasimhan, Karthik R},
  title     = {{$\tau$-Rec}: A Verifiable Benchmark for Agentic Recommender Systems},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems},
  series    = {RecSys '26},
  year      = {2026},
  pages     = {915--919},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3773078.3831847},
  url       = {https://doi.org/10.1145/3773078.3831847}
}

@misc{narasimhan2026taurecbooks,
  author        = {Narasimhan, Bharath Sivaram and Narasimhan, Karthik R.},
  title         = {{$\tau$-Rec-Books}: A Disclosure-Controlled Benchmark for Requirement Elicitation in Agentic Recommender Systems},
  year          = {2026},
  eprint        = {2611.04459},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR},
  url           = {https://arxiv.org/abs/2611.04459}
}
```

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)). The movie
track's task data and catalog are released under CC BY 4.0
([`data/LICENSE`](data/LICENSE)). Each track under `data/tracks/` carries its
own `LICENSE.md`, which controls for that track: for `books-wikidata-v2`, the
catalog and source snapshot are CC0 1.0, and the tasks, answers, and policies
are CC BY 4.0. The repository license does not relicense an upstream dataset or
grant rights the authors do not hold.
