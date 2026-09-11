# Changelog

## v1.3.0 — 2026-09-11

Leaderboard and cost-accounting release. The task set, catalog, policy prompt,
and evaluator are unchanged, so `g1` remains one generation and no published
number moved. The agent-side changes here — prompt caching and reasoning-effort
steering — alter what a run costs and how it is configured, not what the
evaluator reads, so they do not open a new generation.

### Added

- **Four more `g1` entries**, taking the current-harness table to sixteen:
  GPT-5.6 Sol (medium thinking), GPT-5.4 at medium thinking and at no
  thinking, and DeepSeek V4 Flash (max thinking). GPT-5.4 (medium thinking)
  takes the top of `g1` at `pass^4` 0.400.

  The useful result is an inversion rather than the ranking. DeepSeek V4 Flash
  (max thinking) is the better model on a single attempt — `pass^1` 0.592
  against GPT-5.4's 0.533 — and the worse one across four, `pass^4` 0.333
  against 0.400. A model can be more capable per try and less reliable in
  aggregate, which is the distinction `pass^k` exists to surface and a plain
  success rate hides.
- **Prompt caching on Anthropic models.** OpenAI and OpenRouter cache a repeated
  prefix automatically; Anthropic caches only what the request marks, so the
  agent's static prefix went uncached and was re-billed at the full input rate
  on every step of every tool loop. `LiteLLMAgent` now sets a `cache_control`
  breakpoint on the system block, which covers the tool schemas too because
  Anthropic orders the prompt tools → system → messages. Measured against the
  live API: 88% of input served from cache, 2.6× cheaper across a three-step
  loop, and more on the 11–25-call loops a real trial runs.
- **`--reasoning-effort` steering** for model families that reject the
  parameter outright, so a thinking mode can be selected rather than inferred.
- **`usage.json` per run**, recording measured token counts and dollar cost.
  It ships inside the trace archives, so a cost figure in a paper or a
  changelog can be checked rather than taken on trust. The three runs new to
  this release cost $34.97, $11.34, and $5.56 at 74%, 61%, and 91% of agent
  input served from cache.
- **`scripts/verify_thinking_labels.py`.** Three `g0` rows share a model string
  and differ only by thinking mode, which no trace records. The labels rest on
  per-step agent latency separating the three in the order the labels predict,
  and that evidence previously had no script behind it. This recomputes every
  figure from the traces and fails if the medians stop separating.
- A **generated top-3 region in `README.md`**, owned by `leaderboard render`
  and checked by `render --check`. It was maintained by hand and had drifted
  twice, still advertising "top 3 of 12" against a board of sixteen and naming
  a leader that had been displaced.

### Fixed

- **Cached input was billed at the full input rate.** Anthropic and OpenAI
  serve cache reads at a tenth of the input price, and cache writes at 1.25×,
  but the estimator priced every prompt token the same. Runs with a high cache
  rate were overstated by several times. Cost estimates are now split across
  fresh, cached, and written input at their respective rates.
- **Token usage was estimated where the provider had already reported it.**
  Runs now record the counts the API returns. `--dry-run` also claimed a ±30%
  accuracy it had never demonstrated; it now reports what was measured.
- Cost estimates for Claude 4.7+ and for thinking-model pricing tiers.
- GPT-5.6 Sol shipped in v1.2.0 with no **Traces** link, because the published
  archives are pinned to v1.2.0 and its run postdated that tag. Its traces are
  in this release's `g1` archive and the row links to them.

## v1.2.0 — 2026-09-11

Leaderboard release. The task set, catalog, policy prompt, and evaluator are
unchanged, so `g1` remains one generation and no published number moved.

### Added

- **Seven more `g1` entries**, taking the current-harness table to twelve:
  MiniMax M3, Kimi K2.5, and five paper-cohort configurations re-run under
  `g1` — Qwen3-32B, Gemini 2.5 Flash, GPT-5 mini, DeepSeek V4 Flash, and
  DeepSeek V4 Flash (high thinking). Those five appear in both tables; across
  the shared pairs `pass^1` moves by at most 0.08. The tables are still ranked
  separately — `leaderboard/GENERATIONS.md` explains why a shared row is an
  observation rather than a controlled ablation.
- A **Traces** column linking each row to the archive behind it.

### Fixed

- `tau-rec leaderboard make-entry` counted a re-run trial twice. A top-up pass
  writes `<task>_trial<N>.json` into a fresh timestamp directory under the same
  run, and both copies were collected, inflating `n` and corrupting `pass^k`.
  The later pass now supersedes the earlier. No published entry was affected;
  all were rebuilt to confirm.

## v1.1.0 — 2026-09-10

Additive release. The task set, catalog, and policy prompt are byte-identical
to v1.0, and no number published in the paper moved.

### Errata: v1.0 could not reproduce the paper

The v1.0 artifact shipped a 7-policy `data/policy.md`. The nine runs behind the
paper's main results table (Table 2) were executed against an 8-policy version
that also included Preference Respect. Cloning v1.0 and re-running therefore
never reproduced the conditions those numbers were produced under, because the
agent read a different policy prompt.

Several scoring changes also landed between the last paper run and the v1.0
tag, so they are present in the shipped artifact but were not active when the
published numbers were produced. Re-scoring every archived run with the current
evaluator yields zero task-level deltas, so the **published numbers stand** —
but the harness that produced them and the harness that shipped were not the
same thing, and v1.0 gave no way to tell.

This release labels harness generations, so a run can be traced to the content
it scored against. `g0` is the paper cohort; `g1` is the harness in this tag.
See `leaderboard/GENERATIONS.md`.

### Fixed

- **Rec-then-abstain scoring exploit.** The orchestrator executes every tool
  call in an assistant response and lets the last win, so a batched
  `[recommend(item_id="X"), recommend()]` left `stop_reason == ABSTAINED` beside
  a non-empty recommendation list. On the five `no_valid_recommendation` tasks
  scoring read only `stop_reason` and paid full credit. Constraint credit and
  the `transparency` check now both require abstention *and* an empty
  recommendation list.
- `"null"` and `"none"` item_id strings are treated as abstention sentinels.
- `bootstrap_ci` is seeded, so confidence intervals are reproducible.
- `README.md` documented the wrong `--simulator-model` default. It is
  `gpt-5-mini`, which is what the leaderboard expects.

### Added

- **Leaderboard.** `LEADERBOARD.md` is generated from JSON entries in
  `leaderboard/entries/`, which store only raw per-task `{n, c}`. Every
  published figure, including confidence intervals, is derived at render time,
  so the board can be recomputed when the evaluator changes. Submission process
  in `leaderboard/CONTRIBUTING.md`.
- **Harness generations.** Entries record a generation label, and the renderer
  ranks generations in separate tables rather than implying comparability
  between them.
- **`run_manifest.json`**, recording each run's model, configuration, and
  content digests. `tau-rec leaderboard make-entry` reads it in preference to
  its own flags, and refuses a run whose policy, tasks, or catalog have changed
  since.
- Nine `g0` entries from the paper cohort, and five `g1` entries: GLM-5.3
  Flash, DeepSeek Flash, GPT-5.6 Luna (medium thinking), Mistral Small 3,
  Grok 4.3.
- Full conversation traces for both cohorts, attached to this release.
- CI running the test suite, entry validation, and a board re-render check.

## v1.0-recsys2026 — 2026-05-05

Initial release accompanying the RecSys 2026 Reproducibility and Resource Track
paper. 153-film TMDB catalog, 60 tasks, nine evaluated configurations.
