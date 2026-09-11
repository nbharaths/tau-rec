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

### Are the DeepSeek thinking labels real?

The three DeepSeek rows differ only by `reasoning_effort`, and nothing in a g0
trace records that field, so the labels rest on how the runs were launched.
`ds-v4-flash-thinking-4t` started at 19:45 on 2026-05-02 while `a8131ac`
("enable DeepSeek thinking mode when `--reasoning-effort` is set") is stamped
20:40 the same evening — which reads as if the run predates its own feature and
is really a second sample of the plain config.

It isn't. Per-step agent latency separates the three cleanly and in the order
the labels predict:

| Config | median step | mean | p90 |
|--------|-------------|------|-----|
| plain | 14.86s | 22.24s | 50.03s |
| high | 19.70s | 29.29s | 68.89s |
| max | 24.36s | 34.21s | 76.07s |

Mann-Whitney over per-step latencies gives plain < high at `p = 2.4e-9` and
high < max at `p = 1.2e-3`. The plain run started 19:35, ten minutes before the
"high" run, so the two are matched on provider conditions; had thinking been
off in both they would not differ.

The commit timestamp records when the change reached git, not when it existed
on disk. The run was launched from a tree that already carried it. Behaviour
observed in the traces outranks a commit clock, so the labels stand.

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

## g1 — the 7-policy harness

Everything run against HEAD from 2026-09-10 onward. Content digests:

| File | sha256 |
|------|--------|
| `data/policy.md` | `9199737dfa34f4975a66ed5188d6e774191f4c00a2f5c6a1970a9a2a70553064` |
| `data/tasks` | `aa671830eb79e2896dd1b0388fe19fd126b4c44d70fad26feefceaded6700736` |
| `data/catalog.json` | `5f0facd351f3e67e558534814492b5120d324f1921a67f0bdd2952a8373fff52` |

The task set and catalog are byte-identical to g0. What moved is the policy
prompt and the evaluator, and the two have very different evidentiary status.

**Scoring-side — measured inert on the g0 archive.** Re-scoring all ten
archived run directories with the g1 evaluator produces **zero task-level
deltas**, so none of these moved a published number:

- `a99840f` — `_check_preference_respect` removed from `PolicyEvaluator`.
- `48ec2fc` — an NVR trial that hit the turn limit no longer scores as a
  deliberate abstention.
- `4056859` — `single_recommendation` counts `all_recommendations` rather than
  `recommend()` tool calls, which had made the check vacuous.
- `c0804be` / `84165b3` — NVR constraint credit and the `transparency` check
  both require abstention **and** an empty recommendation list. This is the
  rec-then-abstain fix described above; it is the one change that does move a
  score, and it moves it back toward the published g0 value.

**Run-side — not testable against the archive.** These change what the agent
sees or does, so a g0 trace cannot be replayed through them:

- `a99840f` + `e93f4d7` — `data/policy.md` goes from 8 policies to 7. Preference
  Respect (#6) is gone and the rest are renumbered, so every g1 agent reads a
  materially different prompt.
- Tasks 055–057 no longer carry a `preference_respect` flag.
- `8783244` — the orchestrator stops issuing an agent call after `recommend()`
  has fired. The output of that call was already discarded, so this should be
  score-neutral, but "should be" is an argument, not a measurement.

**g0 and g1 are therefore not comparable**, and the renderer ranks them in
separate tables. The policy prompt alone is enough to rule out putting the two
in one ordering, independent of anything the evaluator does.

One consequence worth recording: the public v1.0 artifact shipped the 7-policy
`data/policy.md` while the g0 cohort had been run against 8. Cloning v1.0 and
re-running never reproduced g0 conditions. `g1` is the first generation whose
published content is the content its numbers were produced under.

## Opening the next generation

When a score-affecting change lands:

1. Bump the label (`g1`) and add a section here: what changed, and whether
   re-derivation moves any existing entry.
2. Re-derive the existing cohort at the new HEAD. If numbers move, say so with
   the old and new values — don't quietly re-render.
3. New submissions record `harness_generation: g1` plus all three digests.
