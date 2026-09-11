# Changelog

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
