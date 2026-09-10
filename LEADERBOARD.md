<!-- GENERATED FILE — DO NOT EDIT BY HAND.
     Regenerate with: uv run tau-rec leaderboard render
     Source of truth: leaderboard/entries/*.json -->

# τ-Rec Leaderboard

`primary_reward = constraint_score × policy_score`, scored programmatically against the catalog — no LLM judge. `pass^k` is the unbiased estimator `C(c,k)/C(n,k)`, averaged over tasks; a task with fewer than `k` trials counts as zero. Brackets are 95% BCa bootstrap intervals over per-task scores.

Entries store only raw per-task `{n, c}`. Every number below is derived at render time, so the whole board moves when the evaluator does.

## Standard configuration

Simulator: `gpt-5-mini`, 60 tasks.

| # | Model | pass^1 | pass^2 | pass^4 | Trials | Gen | Notes |
|---|-------|--------|--------|--------|--------|-----|-------|
| 1 | DeepSeek V4 Flash (high thinking) | 0.560 <sub>[0.45, 0.66]</sub> | 0.461 <sub>[0.35, 0.58]</sub> | 0.383 <sub>[0.27, 0.52]</sub> | 4 | g0 | — |
| 2 | DeepSeek V4 Flash (max thinking) | 0.571 <sub>[0.46, 0.67]</sub> | 0.461 <sub>[0.35, 0.57]</sub> | 0.350 <sub>[0.23, 0.47]</sub> | 4 | g0 | — |
| 3 | GPT-5.4 (medium thinking) | 0.551 <sub>[0.44, 0.65]</sub> | 0.450 <sub>[0.34, 0.56]</sub> | 0.350 <sub>[0.23, 0.48]</sub> | 4 | g0 | — |
| 4 | Claude Sonnet 4.6 | 0.537 <sub>[0.43, 0.64]</sub> | 0.433 <sub>[0.32, 0.55]</sub> | 0.350 <sub>[0.23, 0.48]</sub> | 4 | g0 | — |
| 5 | DeepSeek V4 Flash | 0.546 <sub>[0.44, 0.65]</sub> | 0.433 <sub>[0.33, 0.55]</sub> | 0.333 <sub>[0.22, 0.47]</sub> | 4 | g0 | — |
| 6 | GPT-5.4 | 0.471 <sub>[0.37, 0.57]</sub> | 0.358 <sub>[0.25, 0.47]</sub> | 0.283 <sub>[0.18, 0.40]</sub> | 4 | g0 | — |
| 7 | GPT-5 mini | 0.417 <sub>[0.31, 0.53]</sub> | 0.333 <sub>[0.23, 0.45]</sub> | 0.250 <sub>[0.15, 0.37]</sub> | 4 | g0 | `same_model_as_simulator` |
| 8 | Gemini 2.5 Flash | 0.275 <sub>[0.19, 0.38]</sub> | 0.189 <sub>[0.11, 0.29]</sub> | 0.133 <sub>[0.07, 0.23]</sub> | 4 | g0 | — |
| 9 | Qwen3-32B | 0.271 <sub>[0.19, 0.37]</sub> | 0.181 <sub>[0.11, 0.28]</sub> | 0.117 <sub>[0.05, 0.22]</sub> | 4 | g0 | — |

## Columns

- **Trials** — trials per task (`n`). The standard is 4; more is welcome, not required.
- **Gen** — harness generation. Entries from different generations were scored against different policy/task/evaluator content; see `leaderboard/GENERATIONS.md`.
- **Notes** — `same_model_as_simulator` means the agent was graded by a simulator running the same model, which is a confound.

Some entries carry `content_digests_unrecorded`: they predate content-digest recording, so their policy/task/catalog hashes cannot be verified against the repository. See `leaderboard/GENERATIONS.md`.
