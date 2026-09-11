# Changelog

## v1.1.0 — 2026-09-10

Additive release. The task set, catalog, and policy prompt are byte-identical
to v1.0, and no number published in the paper moved. What changes is that runs
are now labelled well enough to say which harness produced them.

### Errata: v1.0 could not reproduce the paper

The v1.0 artifact shipped a 7-policy `data/policy.md`. The nine runs behind the
paper's main results table (Table 2) were executed against an 8-policy version
that also included Preference Respect. Cloning v1.0 and re-running therefore
never reproduced the conditions those numbers were produced under, because the
agent read a different policy prompt.

Four changes landed between the last paper run and the v1.0 tag, so they are
present in the shipped artifact but were not active when the published numbers
were produced:

- Preference Respect (then Policy 6) and its `_check_preference_respect`
  evaluator method were removed, and the remaining policies renumbered 1–7.
- Tasks 055–057 stopped carrying a `preference_respect` flag.
- `no_valid_recommendation` trials that hit the turn limit stopped scoring as
  deliberate abstentions.
- `single_recommendation` switched to counting `all_recommendations` rather
  than `recommend()` tool calls, which had made the check vacuous.
- The orchestrator stopped issuing a wasted agent call after `recommend()` had
  already fired.

None of them moves a published number. Re-scoring every archived run directory
with the current evaluator yields zero task-level deltas, tasks 055–057
included. So the **published numbers stand** — but the harness that produced
them and the harness that shipped were not the same thing, and v1.0 gave no way
to tell.

This release fixes that by labelling harness generations. `g0` is the paper
cohort; `g1` is the harness in this tag, and is the first published state whose
content matches its own numbers. See `leaderboard/GENERATIONS.md`.

### Fixed

- **Rec-then-abstain scoring exploit.** The orchestrator executes every tool
  call in an assistant response and lets the last win, so a batched
  `[recommend(item_id="X"), recommend()]` left `stop_reason == ABSTAINED` beside
  a non-empty recommendation list. On the five `no_valid_recommendation` tasks
  scoring read only `stop_reason` and paid full credit — strictly
  score-increasing, and worth up to 8.3 points (5 of 60 tasks) to a deliberate
  submitter. It also occurred unprompted in the archive:
  `task_052_trial3` of `ds-v4-flash-4t-gpt5mini` calls
  `recommend(item_id="null")` followed by `recommend()`. Constraint credit and
  the `transparency` check now both require abstention *and* an empty
  recommendation list.
- `"null"` and `"none"` item_id strings are treated as abstention sentinels,
  so an agent cannot register a recommendation and an abstention at once.
- `bootstrap_ci` is seeded, so confidence intervals are reproducible.

### Added

- **Leaderboard.** `LEADERBOARD.md` is generated from JSON entries in
  `leaderboard/entries/`, which store only raw per-task `{n, c}`. Every
  published figure, including confidence intervals, is derived at render time,
  so the whole board can be recomputed when the evaluator changes. Submission
  process in `leaderboard/CONTRIBUTING.md`.
- **Harness generations.** Entries record a generation label, and the renderer
  ranks generations in separate tables rather than implying comparability
  between them. `leaderboard/GENERATIONS.md` records what differs between them.
- **`run_manifest.json`**, written before a run's first trial so a killed run
  still records what it was. Nothing else on disk captured `reasoning_effort`,
  which made a completed run impossible to label afterwards.
  `tau-rec leaderboard make-entry` prefers the manifest over its own flags and
  refuses a run whose policy, tasks, or catalog have changed since.
- Content digests (`policy_sha256`, `tasks_sha256`, `catalog_sha256`) recorded
  per run and per entry, for comparability checking.
- Nine `g0` entries reconstructed from the paper cohort, and five `g1` entries:
  GLM-5.3 Flash, DeepSeek Flash, GPT-5.6 Luna (medium thinking), Mistral
  Small 3, Grok 4.3.
- Full conversation traces for both cohorts, attached to this release, so any
  published number can be re-derived rather than taken on trust.

### Fixed in documentation

- `README.md` documented `--simulator-model` as defaulting to
  `gemini/gemini-2.5-pro`; the actual default is `gpt-5-mini`. Since the
  simulator is worth roughly 13.7 `pass^1` points, the stale line could have
  produced submissions that were not comparable to the board.

## v1.0-recsys2026 — 2026-05-05

Initial release accompanying the RecSys 2026 Reproducibility and Resource Track
paper. 153-film TMDB catalog, 60 tasks, nine evaluated configurations.
