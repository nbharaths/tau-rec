# Harness generations

A **generation** labels one set of scored content: the policy prompt, the task
files, the catalog, and the evaluator that reads them. Two entries in the same
generation are directly comparable. Two entries in different generations are
not, even if their `pass^k` columns sit next to each other.

A raw content hash is a poor comparability signal on its own — it changes when
a typo is fixed and stays put when an evaluator method is rewritten. The
generation label is a judgement about whether a change could move a score.

## Rules

- A new generation opens when a change **could** move a published score: policy
  text, task constraints or reveal tags, catalog contents, or evaluator logic.
  Not for docs, tests, renderers, or CLI plumbing.
- Entries record `harness_generation`. The renderer shows it as the **Gen**
  column and never mixes generations into a single ranking.
- Scores are **re-derived from traces at submission time**, not copied from a
  run's `task_results.json`. Re-derived is canonical; a stored number that
  disagrees with the current evaluator is treated as a bug in the stored number.

## g0 — the RecSys paper cohort

Nine configurations, run 2026-05-02 to 2026-05-03, 60 tasks × 4 trials,
simulator `gpt-5-mini`. This is the cohort behind the paper's main results
table (Table 2).

`g0` entries carry the `content_digests_unrecorded` caveat: they predate
`run_manifest.json`, so no digest was recorded at run time and none can be
reconstructed. Recording today's digests on them would be a false claim, so the
fields are null.

The content they ran against differs from `g1` in three ways:

| Area | g0 | g1 |
|------|----|----|
| `data/policy.md` | 8 policies, incl. #6 Preference Respect | 7 policies |
| `data/tasks` | tasks 055–057 carry a `preference_respect` flag | flag removed |
| Evaluator | `_check_preference_respect` present | removed |

**The g0 numbers stand, and that is a measurement rather than an argument.**
`scripts/rederive_seed_cohort.py` re-scores every archived trace with the
current evaluator and diffs per-task `{n, c}` against each run's stored
results. The result is zero task-level deltas, tasks 055–057 included.

Three `g0` rows share a model string and differ only by thinking mode, which no
trace records. The labels are supported by per-step agent latency, which
separates the three in the order the labels predict (medians 14.9s / 19.7s /
24.4s; Mann-Whitney `p = 2.4e-9` and `p = 1.2e-3`).

## g1 — the 7-policy harness

Content digests:

| File | sha256 |
|------|--------|
| `data/policy.md` | `9199737dfa34f4975a66ed5188d6e774191f4c00a2f5c6a1970a9a2a70553064` |
| `data/tasks` | `aa671830eb79e2896dd1b0388fe19fd126b4c44d70fad26feefceaded6700736` |
| `data/catalog.json` | `5f0facd351f3e67e558534814492b5120d324f1921a67f0bdd2952a8373fff52` |

The task set and catalog are byte-identical to g0. The policy prompt and the
evaluator moved, and the two differ in what can be shown about them.

**Evaluator changes, measured inert on the g0 archive.** Re-scoring every
archived run with the g1 evaluator produces zero task-level deltas:

- `_check_preference_respect` removed.
- An NVR trial that hits the turn limit no longer scores as a deliberate
  abstention.
- `single_recommendation` counts `all_recommendations` rather than `recommend()`
  tool calls, which had made the check vacuous.
- NVR constraint credit and the `transparency` check both require abstention
  **and** an empty recommendation list. Previously a batched
  `[recommend(item_id="X"), recommend()]` left `stop_reason == ABSTAINED` beside
  a recommendation and was paid full credit on the five NVR tasks. This is the
  one change that moves a score, and it moves it back toward the published
  value.

**Prompt and orchestration changes, not testable against the archive.** These
change what the agent sees or does, so a g0 trace cannot be replayed through
them:

- `data/policy.md` goes from 8 policies to 7, so every g1 agent reads a
  materially different prompt.
- Tasks 055–057 no longer carry a `preference_respect` flag.
- The orchestrator stops issuing an agent call after `recommend()` has fired.
  That call's output was already discarded, so this should be score-neutral,
  but it is an argument rather than a measurement.

**g0 and g1 are therefore not comparable**, and the renderer ranks them in
separate tables. The policy prompt alone rules out a single ordering,
independent of anything the evaluator does.

The v1.0 artifact shipped the 7-policy `data/policy.md` while the g0 cohort had
been run against 8, so cloning v1.0 never reproduced g0 conditions. `g1` is the
first generation whose published content is the content its numbers were
produced under.

## Opening the next generation

1. Bump the label and add a section here: what changed, and whether
   re-derivation moves any existing entry.
2. Re-derive the existing cohort. If numbers move, publish the old and new
   values rather than quietly re-rendering.
3. New submissions record the new `harness_generation` plus all three digests.
