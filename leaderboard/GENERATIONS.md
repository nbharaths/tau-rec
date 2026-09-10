# Harness generations

A **generation** is a label for one set of scored content: the policy prompt, the
task files, the catalog, and the evaluator that reads them. Two entries in the
same generation are directly comparable. Two entries in different generations
are not, even if their `pass^k` columns sit next to each other.

Generations exist because a raw content hash is a bad comparability signal on its
own. It changes when a typo is fixed and stays put when an evaluator method is
rewritten, so it flags noise and misses the thing you care about. The generation
label is a human judgement about whether a scoring change could move a number.

## Rules

- A new generation is opened when a change **could** move a published score:
  policy text, task constraints or reveal tags, catalog contents, or evaluator
  logic. Not for docs, tests, renderers, or CLI plumbing.
- Entries record `harness_generation`. The renderer shows it as the **Gen**
  column and never mixes generations into a single ranking claim.
- Scores are **re-derived at submission time from traces**, not copied from a
  run's `task_results.json`. Re-derived is canonical. A stored number that
  disagrees with the current evaluator is treated as a bug in the stored number.

## g0 — the RecSys paper cohort

Nine configurations, run 2026-05-02 to 2026-05-03, 60 tasks × 4 trials,
simulator `gpt-5-mini`. This is the cohort behind the paper's Table 3.

`g0` entries carry the `content_digests_unrecorded` caveat: they predate
`run_manifest.json`, so no policy/task/catalog digest was recorded at run time
and none can be reconstructed. Recording today's digests on them would be a
false claim, so the fields are null.

The content they actually ran against differs from HEAD in three known ways:

| Area | At g0 | At HEAD |
|------|-------|---------|
| `data/policy.md` | 8 policies, incl. #6 Preference Respect | 7 policies (`a99840f`) |
| `data/tasks` | tasks 055–057 carry a `preference_respect` flag | flag removed |
| Evaluator | `_check_preference_respect` live from `54eba85` (2026-04-16) to `a99840f` (2026-05-03 22:30) | removed |

Every g0 run started before `a99840f` — the latest began 2026-05-03 21:20 — so
all nine were scored by a harness in which that check was live and three tasks
carried its flag.

**This is measured, not assumed.** `scripts/rederive_seed_cohort.py` re-scores
every archived trace with the HEAD evaluator and diffs per-task `{n, c}` against
each run's stored `task_results.json`. Across all ten archived run directories
the result is **zero task-level deltas**, tasks 055–057 included. The removed
check never changed an outcome on any archived trial, and
`ds-v4-flash-4t-gpt5mini` re-derives to `pass^1/2/4 = 0.546 / 0.433 / 0.333`,
matching the paper exactly.

So `g0` numbers stand as published, and the reason is a reproduction result
rather than an argument about what the code used to do.

### The rec-then-abstain fix

`g0` re-derivation only agrees with the paper **after** the scoring fix in
`70c2f17` / `d8789d4`. The orchestrator executes every tool call in an assistant
response and lets the last one win, so a batched
`[recommend(item_id="X"), recommend()]` left `stop_reason == ABSTAINED` next to a
non-empty recommendation list. On the five `no_valid_recommendation` tasks
(050–054), scoring read only `stop_reason` and paid full credit — strictly
score-increasing, worth about 8.3 points. `task_052_trial3` of
`ds-v4-flash-4t-gpt5mini` is a live instance: the pre-fix HEAD evaluator scored
it 1.0 where the stored result said 0.0.

Constraint credit and the `transparency` check now both require abstention
**and** an empty recommendation list. That restores agreement with the stored
g0 numbers, which is why the exploit is a fix rather than a new generation: it
moves scores toward what was published, not away from it.

## Opening the next generation

When a score-affecting change lands:

1. Bump the label (`g1`) and add a section here: what changed, and whether
   re-derivation moves any existing entry.
2. Re-derive the existing cohort at the new HEAD. If numbers move, say so with
   the old and new values — don't quietly re-render.
3. New submissions record `harness_generation: g1` plus all three digests.
