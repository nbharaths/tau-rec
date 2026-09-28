# Benchmark comparison

Last verified: 2026-09-27.

This table compares evaluation protocols, not overall paper quality or scope.
**✓** means the feature is part of the benchmark, **◐** means a limited or
proxy form, and **—** means it is not part of the evaluation. In the final
column, the values are **public tasks / public leaderboard**.

| Benchmark | Primary evaluation target | Live multi-turn dialogue | Benchmark-exposed tools | Automated user simulation | Programmatic task verification | Public tasks / leaderboard |
|---|---|:---:|:---:|:---:|---|:---:|
| **[τ-Rec](https://doi.org/10.1145/3773078.3831847)** | Constraint- and policy-compliant conversational recommendation | ✓ | ✓ Catalog and commitment tools | ✓ LLM | ✓ Catalog predicates and trace-policy checks | ✓ / ✓ |
| [AgentRecBench](https://arxiv.org/abs/2505.19623) | Personalized top-*N* ranking | — No user dialogue | ✓ Query interface | — | ◐ Ground-truth HR@*N* | ✓ / ✓ |
| [CRS Arena](https://doi.org/10.1145/3701551.3704120) | Human preference among CRSs | ✓ Human–CRS | — | — Human users | — Human vote and Elo | — / ✓ |
| [RecToM](https://arxiv.org/abs/2511.22275) | Theory-of-mind QA over recommendation dialogues | — Fixed transcripts | — | — Human ReDial dialogues | ◐ Answer-key accuracy | ✓ / — |
| [UserSimCRS v2](https://arxiv.org/abs/2512.04588) | CRS and user-simulator evaluation toolkit | ✓ CRS–simulator | — Text interface only | ✓ Agenda and LLM | ◐ Intent-derived metrics plus LLM judge | — / — |
| [iEvaLM](https://doi.org/10.18653/v1/2023.emnlp-main.621) | Interactive CRS evaluation | ✓ CRS–simulator | — | ✓ LLM | ◐ Target-item metrics | ✓ / — |
| [RecBench+](https://arxiv.org/abs/2503.09382) | Single-turn recommendation-assistant queries | — Single query | — | — | ✓ Recall and condition matching | ✓ / — |
| [τ-bench](https://arxiv.org/abs/2406.12045) | Customer-service task completion | ✓ Agent–simulator | ✓ Agent APIs | ✓ LLM | ✓ Final database goal state | ✓ / ✓ |
| [τ²-bench](https://arxiv.org/abs/2506.07982) | Dual-control support task completion | ✓ Agent–simulator | ✓ Agent and user tools | ✓ Environment-coupled LLM | ✓ State and action assertions | ✓ / ✓ |

τ-Rec is the only recommendation-specific benchmark in this comparison that
combines live multi-turn evaluation, benchmark-exposed tools, automated user
simulation, programmatic end-to-end verification, public tasks, and a public
leaderboard. The τ-bench family has the same broad evaluation shape in customer
service rather than recommendation.

## Definitions

- **Live multi-turn dialogue** requires the evaluated system to exchange turns
  with a human or simulated user during evaluation. Supplying a fixed
  multi-turn transcript as model context does not qualify.
- **Benchmark-exposed tools** are callable environment or action APIs available
  to the evaluated system. A model's private retriever or a dataset-generation
  tool does not qualify.
- **Programmatic task verification** receives a full check only when the main
  success outcome is computed from structured catalog/database state,
  constraints, or assertions without a human or model judge. Deterministic
  proxies such as HR@*N*, target-item recall, or answer-key accuracy are marked
  partial because they do not verify completion of the interactive task.
- **Public tasks** means fixed evaluation instances or an executable evaluation
  set, not only source dialogue corpora or example configurations. A **public
  leaderboard** is a publicly viewable benchmark ranking.

## Primary sources

- **τ-Rec:** [paper](https://doi.org/10.1145/3773078.3831847),
  [tasks](https://github.com/nbharaths/tau-rec/tree/main/data/tasks), and
  [leaderboard](https://github.com/nbharaths/tau-rec/blob/main/LEADERBOARD.md).
- **AgentRecBench:** [paper](https://arxiv.org/abs/2505.19623),
  [public benchmark data](https://huggingface.co/datasets/SGJQovo/AgentRecBench),
  and [challenge leaderboard](https://tsinghua-fib-lab.github.io/AgentSocietyChallenge/pages/overview.html).
- **CRS Arena:** [paper](https://doi.org/10.1145/3701551.3704120),
  [public arena and Elo ranking](https://huggingface.co/spaces/iai-group/CRSArena), and
  [CRSArena-Dial](https://github.com/iai-group/crsarena-dial).
- **RecToM:** [paper](https://arxiv.org/abs/2511.22275) and
  [dataset/evaluation scripts](https://github.com/CGCL-codes/RecToM).
- **UserSimCRS v2:** [paper](https://arxiv.org/abs/2512.04588) and
  [toolkit](https://github.com/iai-group/UserSimCRS).
- **iEvaLM:** [paper](https://doi.org/10.18653/v1/2023.emnlp-main.621) and
  [code/evaluation data](https://github.com/RUCAIBox/iEvaLM-CRS).
- **RecBench+:** [paper](https://arxiv.org/abs/2503.09382) and
  [dataset/evaluation code](https://github.com/jiani-huang/RecBench).
- **τ-bench:** [paper](https://arxiv.org/abs/2406.12045),
  [tasks/code, with its current supersession notice](https://github.com/sierra-research/tau-bench),
  and the repository leaderboard.
- **τ²-bench:** [paper](https://arxiv.org/abs/2506.07982),
  [tasks/code](https://github.com/sierra-research/tau2-bench), and
  [live leaderboard](https://taubench.com/).

If a public capability changes, please open an issue with a primary-source
link so this table can be corrected.
