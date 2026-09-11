<!-- GENERATED FILE — DO NOT EDIT BY HAND.
     Regenerate with: uv run tau-rec leaderboard render
     Source of truth: leaderboard/entries/*.json -->

# τ-Rec Leaderboard

`primary_reward = constraint_score × policy_score`, scored programmatically against the catalog — no LLM judge. `pass^k` is the unbiased estimator `C(c,k)/C(n,k)`, averaged over tasks; a task with fewer than `k` trials counts as zero. Brackets are 95% BCa bootstrap intervals over per-task scores.

Entries store only raw per-task `{n, c}`. Every number below is derived at render time, so the whole board moves when the evaluator does.

**Generations are ranked separately and cannot be compared across tables.** Each was scored against different policy, task, or evaluator content; `leaderboard/GENERATIONS.md` records exactly what differs.

## Generation `g1` — most recent

Simulator: `gpt-5-mini`, 60 tasks.

| # | Model | pass^1 | pass^2 | pass^4 | Trials | Gen | Notes | Traces |
|---|-------|--------|--------|--------|--------|-----|-------|--------|
| 1 | DeepSeek V4 Flash (high thinking) | 0.604 <sub>[0.50, 0.70]</sub> | 0.494 <sub>[0.38, 0.60]</sub> | 0.383 <sub>[0.27, 0.52]</sub> | 4 | g1 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |
| 2 | GLM-5.3 Flash | 0.525 <sub>[0.42, 0.63]</sub> | 0.419 <sub>[0.31, 0.54]</sub> | 0.333 <sub>[0.23, 0.47]</sub> | 4 | g1 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |
| 3 | DeepSeek Flash | 0.479 <sub>[0.38, 0.59]</sub> | 0.389 <sub>[0.29, 0.51]</sub> | 0.300 <sub>[0.20, 0.43]</sub> | 4 | g1 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |
| 4 | DeepSeek V4 Flash | 0.567 <sub>[0.47, 0.65]</sub> | 0.419 <sub>[0.32, 0.52]</sub> | 0.283 <sub>[0.18, 0.40]</sub> | 4 | g1 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |
| 5 | Kimi K2.5 | 0.487 <sub>[0.39, 0.59]</sub> | 0.367 <sub>[0.26, 0.48]</sub> | 0.283 <sub>[0.18, 0.40]</sub> | 4 | g1 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |
| 6 | GPT-5.6 Luna (medium thinking) | 0.471 <sub>[0.37, 0.57]</sub> | 0.358 <sub>[0.26, 0.47]</sub> | 0.250 <sub>[0.15, 0.37]</sub> | 4 | g1 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |
| 7 | MiniMax M3 | 0.467 <sub>[0.38, 0.57]</sub> | 0.331 <sub>[0.24, 0.44]</sub> | 0.250 <sub>[0.15, 0.37]</sub> | 4 | g1 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |
| 8 | GPT-5 mini | 0.396 <sub>[0.30, 0.50]</sub> | 0.289 <sub>[0.20, 0.39]</sub> | 0.183 <sub>[0.10, 0.30]</sub> | 4 | g1 | `same_model_as_simulator` | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |
| 9 | Gemini 2.5 Flash | 0.308 <sub>[0.22, 0.41]</sub> | 0.219 <sub>[0.14, 0.32]</sub> | 0.150 <sub>[0.08, 0.27]</sub> | 4 | g1 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |
| 10 | Qwen3-32B | 0.350 <sub>[0.26, 0.45]</sub> | 0.242 <sub>[0.17, 0.34]</sub> | 0.133 <sub>[0.07, 0.23]</sub> | 4 | g1 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |
| 11 | Mistral Small 3 | 0.362 <sub>[0.27, 0.46]</sub> | 0.233 <sub>[0.16, 0.33]</sub> | 0.117 <sub>[0.05, 0.22]</sub> | 4 | g1 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |
| 12 | Grok 4.3 | 0.221 <sub>[0.14, 0.32]</sub> | 0.156 <sub>[0.08, 0.26]</sub> | 0.117 <sub>[0.05, 0.22]</sub> | 4 | g1 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g1-traces.tar.gz) |

## Generation `g0`

Simulator: `gpt-5-mini`, 60 tasks.

| # | Model | pass^1 | pass^2 | pass^4 | Trials | Gen | Notes | Traces |
|---|-------|--------|--------|--------|--------|-----|-------|--------|
| 1 | DeepSeek V4 Flash (high thinking) | 0.560 <sub>[0.45, 0.66]</sub> | 0.461 <sub>[0.35, 0.58]</sub> | 0.383 <sub>[0.27, 0.52]</sub> | 4 | g0 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g0-traces.tar.gz) |
| 2 | DeepSeek V4 Flash (max thinking) | 0.571 <sub>[0.46, 0.67]</sub> | 0.461 <sub>[0.35, 0.57]</sub> | 0.350 <sub>[0.23, 0.47]</sub> | 4 | g0 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g0-traces.tar.gz) |
| 3 | GPT-5.4 (medium thinking) | 0.551 <sub>[0.44, 0.65]</sub> | 0.450 <sub>[0.34, 0.56]</sub> | 0.350 <sub>[0.23, 0.48]</sub> | 4 | g0 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g0-traces.tar.gz) |
| 4 | Claude Sonnet 4.6 | 0.537 <sub>[0.43, 0.64]</sub> | 0.433 <sub>[0.32, 0.55]</sub> | 0.350 <sub>[0.23, 0.48]</sub> | 4 | g0 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g0-traces.tar.gz) |
| 5 | DeepSeek V4 Flash | 0.546 <sub>[0.44, 0.65]</sub> | 0.433 <sub>[0.33, 0.55]</sub> | 0.333 <sub>[0.22, 0.47]</sub> | 4 | g0 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g0-traces.tar.gz) |
| 6 | GPT-5.4 | 0.471 <sub>[0.37, 0.57]</sub> | 0.358 <sub>[0.25, 0.47]</sub> | 0.283 <sub>[0.18, 0.40]</sub> | 4 | g0 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g0-traces.tar.gz) |
| 7 | GPT-5 mini | 0.417 <sub>[0.31, 0.53]</sub> | 0.333 <sub>[0.23, 0.45]</sub> | 0.250 <sub>[0.15, 0.37]</sub> | 4 | g0 | `same_model_as_simulator` | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g0-traces.tar.gz) |
| 8 | Gemini 2.5 Flash | 0.275 <sub>[0.19, 0.38]</sub> | 0.189 <sub>[0.11, 0.29]</sub> | 0.133 <sub>[0.07, 0.23]</sub> | 4 | g0 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g0-traces.tar.gz) |
| 9 | Qwen3-32B | 0.271 <sub>[0.19, 0.37]</sub> | 0.181 <sub>[0.11, 0.28]</sub> | 0.117 <sub>[0.05, 0.22]</sub> | 4 | g0 | — | [archive](https://github.com/nbharaths/tau-rec/releases/download/v1.2.0/tau-rec-g0-traces.tar.gz) |


## Columns

- **Trials** — trials per task (`n`). The standard is 4; more is welcome, not required.
- **Gen** — harness generation. Entries from different generations were scored against different policy/task/evaluator content; see `leaderboard/GENERATIONS.md`.
- **Notes** — `same_model_as_simulator` means the agent was graded by a simulator running the same model, which is a confound.
- **Traces** — every conversation behind the row, so the numbers can be re-derived rather than taken on trust. `—` means the entry ships no traces.

Some entries carry `content_digests_unrecorded`: they predate content-digest recording, so their policy/task/catalog hashes cannot be verified against the repository. See `leaderboard/GENERATIONS.md`.
