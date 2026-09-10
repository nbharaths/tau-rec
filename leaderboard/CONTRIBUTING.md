# Submitting a run

A submission is one JSON file in `leaderboard/entries/`. It records **raw
per-task counts and nothing else** — no `pass^k`, no averages, no rank. Every
number on the board is derived from those counts at render time, so the board
can be recomputed whenever the evaluator changes and nobody has to trust a
figure you calculated yourself.

## 1. Run the benchmark

```bash
uv run tau-rec run \
  --model <litellm-model-string> \
  --catalog data/catalog.json \
  --tasks data/tasks \
  --policy data/policy.md \
  --output out/ \
  --trials 4
```

Four trials per task over all 60 tasks is the standard configuration. More
trials are welcome — `pass^4` at `n=4` can only take five distinct values per
task, so higher `n` buys real resolution. Fewer than 4 will not be accepted:
`pass^k` needs `n >= k`, and a task with too few trials counts as zero.

Keep the simulator at the default `gpt-5-mini`. Changing it is allowed, but
those entries are ranked in a separate section — the simulator is worth roughly
13.7 `pass^1` points against a board span of about 28, so it would be the
largest uncontrolled variable on the page.

## 2. Write the entry

```json
{
  "submission_id": "g1-my-model",
  "display_name": "My Model (medium thinking)",
  "submitted_by": "your name or org",
  "model_id": "provider/my-model-2026-08-01",
  "pinned": true,
  "run_date": "2026-09-01",
  "reasoning_effort": "medium",
  "no_tools": false,
  "simulator_model": "gpt-5-mini",
  "harness_generation": "g1",
  "tau_rec_version": "0.1.0",
  "trials_per_task": 4,
  "policy_sha256": "...",
  "tasks_sha256": "...",
  "catalog_sha256": "...",
  "per_task": {
    "task_001": { "n": 4, "c": 3 },
    "task_002": { "n": 4, "c": 0 }
  },
  "traces_url": "https://github.com/.../releases/download/...",
  "caveats": []
}
```

`n` is trials attempted, `c` is trials where `primary_reward == 1.0`.

The three digests come from your run's `run_manifest.json`. They pin the exact
policy, task set, and catalog you scored against; without them an entry cannot
be checked for comparability and is marked `content_digests_unrecorded` on the
board.

**`model_id` must not end in a floating tag** (`-latest`, `:latest`,
`-preview`, `@latest`). Those silently re-point at a different model and make
the entry unreproducible. Pin a dated snapshot. If your provider only offers an
unpinned alias, set `"pinned": false` and give an accurate `run_date`.

Two caveats are added automatically and cannot be omitted by leaving them out:
`same_model_as_simulator` when the agent and simulator are the same model, and
`content_digests_unrecorded` when any digest is missing.

## 3. Validate and render

```bash
uv run tau-rec leaderboard validate
uv run tau-rec leaderboard render
uv run pytest
```

`render` rewrites `LEADERBOARD.md`. Commit it along with your entry — CI runs
`render --check` and fails if the committed board does not match a fresh render
of the entry files.

## 4. Open a pull request

Include the model, the configuration, and where the traces live. Traces are not
required, but an entry with a `traces_url` can be re-derived by anyone, and
re-derived numbers are the ones that count. An entry without traces is taken on
trust and labelled as such.

We re-derive submitted entries from traces where they are available. If
re-derivation disagrees with your counts, the re-derived numbers win and we
will say so on the PR.
