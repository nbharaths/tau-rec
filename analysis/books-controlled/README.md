# Analysis code for the books track

Scripts behind the reported `books-wikidata-v2` results. Their inputs and
outputs live in [`analysis/books-wikidata-v2/`](../books-wikidata-v2/).

| Script | Produces |
|---|---|
| `make_panels.py` | `panels.json`: outcome-blind family panels (seed in the file) |
| `make_v4_experiment_plan.py` | `experiment_plan_v4.json`: the frozen plan (stages, models, pins, harness, analysis) |
| `make_v4_addendum_plan.py` | `experiment_plan_v4_addendum.json`: the exploratory follow-up plan |
| `make_v4_ablation_plan.py` | `experiment_plan_v4_ablation.json`: the exploratory field-list follow-up plan |
| `make_v4_followup_plans.py` | `experiment_plan_v4_prompt_open.json`, `experiment_plan_v4_verify.json` and `experiment_plan_v4_protocol.json`: the exploratory GLM/Qwen prompt-variant, four-agent ask-and-verify and four-agent protocol follow-ups (model configuration `model_config_v4_followup.json`) |
| `run_v4_matrix.py` (with `matrix_stages.py`) | The run itself: plans, launches, resumes, and validates every cell |
| `analyze_v4.py` | `v4_results.json`: the pre-specified primary and secondary contrasts, and the exploratory measures |
| `analyze_v4_addendum.py` | `v4_addendum_results.json`: the exploratory follow-up contrasts |
| `analyze_v4_ablation.py` | `v4_ablation_results.json`: the field-list follow-up contrasts |
| `analyze_v4_followups.py` | `v4_followup_results.json`: the GLM/Qwen prompt-variant follow-up contrasts |
| `posthoc_v4.py` | `v4_posthoc_results.json`: post hoc analyses (interface by tier, sign-flip tests, minimum detectable effect, stall sensitivity, non-Anglo split, rejections per hidden success, hidden success within k rejected proposals, mixed success by elicitation) |
| `extract_disclosure_turns.py`, `score_disclosure_misreads.py` | `disclosure_misread_bound.json`: the agent turns that made the rule-based user disclose a fact on mixed tasks, scored against blind LLM labels (`disclosure_misread_labels.json`), as a lower bound on mixed success |
| `prompt_headroom.py` | `prompt_headroom.json`: each agent's reliability-block levels under the base, ask-requirements, ask-and-verify (`--verify-root`) and protocol (`--protocol-root`) prompts, beside the popularity and eliminating agents (`--references`), including the share of hidden failures whose final book was never proposed (post hoc) |
| `recognizer_bounds.py` | `recognizer_bounds.json`: how far recogniser errors in either direction can move mixed success, with missed questions labelled blind by an LLM (`missed_question_labels.json`) (post hoc) |
| `hidden_failures.py` | `hidden_failures.json`: how satisfiable hidden trials on the primary panel end: success, no recommendation, an untested book, or a rejected or accepted one (post hoc) |
| `score_label_agreement.py` | `controller_eval_second_labeller.json`: agreement between two blind LLM labellers of the recogniser's evaluation set (second labels in `controller_eval_labels_v4_second.json`), and the recogniser re-scored against the second labels (post hoc) |
| `redial_disclosure.py` | `redial_disclosure.json`: when human seekers in ReDial first state genre, era and origin requirements (lexicon-based; the corpus is not redistributed, pass its path with `--redial`) |
| `run_scripted_reference.py` | `scripted_reference_v4.json`: the popularity agent, a deterministic scripted agent (no LLM); with `--order eliminate --hypotheses broad`, `scripted_eliminating_broad_v4.json`: the eliminating agent (`scripted_eliminating_v4.json` assumes the generator's template shapes) |
| `no_ask_floor.py` | `no_ask_floor.json`: the no-ask reference |
| `analyze_retrieval.py` | `retrieval_diagnostics.json` / `.md`: fixed-query retrieval checks |
| `score_controller_eval_v4.py` | `controller_eval_v4_results.json`: recogniser precision and recall against the labels in `controller_eval_labels_v4.json` |

`CONTROLLER_PROTOCOL.md` specifies the rule-based user.

## Reproducing the reported numbers

Every trace behind the reported results is attached to the GitHub release.
Extract the archives at the repository root, then re-run the analysis and
compare with the committed results:

```bash
tar -xzf books-wikidata-v4.tar.gz            # main run -> out/books-wikidata-v4/
tar -xzf books-wikidata-v4-addendum.tar.gz   # follow-up -> out/books-wikidata-v4-addendum/
tar -xzf books-wikidata-v4-ablation.tar.gz   # field-list follow-up -> out/books-wikidata-v4-ablation/
tar -xzf books-wikidata-v4-prompt-open.tar.gz  # GLM/Qwen prompt-variant follow-up -> out/books-wikidata-v4-prompt-open/
tar -xzf books-wikidata-v4-verify.tar.gz     # ask-and-verify follow-up -> out/books-wikidata-v4-verify/
tar -xzf books-wikidata-v4-protocol.tar.gz   # protocol follow-up -> out/books-wikidata-v4-protocol/

uv run python analysis/books-controlled/analyze_v4.py \
  --artifact data/tracks/books-wikidata-v2 \
  --plan analysis/books-wikidata-v2/experiment_plan_v4.json \
  --root out/books-wikidata-v4 \
  --scripted analysis/books-wikidata-v2/scripted_reference_v4.json \
  --output out/v4_results.json
uv run python analysis/books-controlled/analyze_v4_addendum.py \
  --artifact data/tracks/books-wikidata-v2 \
  --main-root out/books-wikidata-v4 --addendum-root out/books-wikidata-v4-addendum \
  --output out/v4_addendum_results.json
uv run python analysis/books-controlled/analyze_v4_ablation.py \
  --artifact data/tracks/books-wikidata-v2 \
  --main-root out/books-wikidata-v4 --ablation-root out/books-wikidata-v4-ablation \
  --output out/v4_ablation_results.json
uv run python analysis/books-controlled/posthoc_v4.py \
  --artifact data/tracks/books-wikidata-v2 \
  --plan analysis/books-wikidata-v2/experiment_plan_v4.json \
  --root out/books-wikidata-v4 \
  --scripted analysis/books-wikidata-v2/scripted_reference_v4.json \
  --eliminating analysis/books-wikidata-v2/scripted_eliminating_broad_v4.json \
  --eliminating-template analysis/books-wikidata-v2/scripted_eliminating_v4.json \
  --output out/v4_posthoc_results.json
uv run python analysis/books-controlled/analyze_v4_followups.py \
  --artifact data/tracks/books-wikidata-v2 --main-root out/books-wikidata-v4 \
  --prompt-root out/books-wikidata-v4-prompt-open --output out/v4_followup_results.json
uv run python analysis/books-controlled/prompt_headroom.py \
  --artifact data/tracks/books-wikidata-v2 --main-root out/books-wikidata-v4 \
  --addendum-root out/books-wikidata-v4-addendum --prompt-root out/books-wikidata-v4-prompt-open \
  --verify-root out/books-wikidata-v4-verify --protocol-root out/books-wikidata-v4-protocol \
  --references analysis/books-wikidata-v2/scripted_reference_v4.json \
               analysis/books-wikidata-v2/scripted_eliminating_broad_v4.json \
  --output out/prompt_headroom.json
uv run python analysis/books-controlled/hidden_failures.py \
  --artifact data/tracks/books-wikidata-v2 \
  --plan analysis/books-wikidata-v2/experiment_plan_v4.json \
  --root out/books-wikidata-v4 --scripted analysis/books-wikidata-v2/scripted_reference_v4.json \
  --output out/hidden_failures.json
uv run python analysis/books-controlled/recognizer_bounds.py \
  --artifact data/tracks/books-wikidata-v2 \
  --plan analysis/books-wikidata-v2/experiment_plan_v4.json \
  --root out/books-wikidata-v4 --labels analysis/books-wikidata-v2/missed_question_labels.json \
  --output out/recognizer_bounds.json
uv run python analysis/books-controlled/run_scripted_reference.py \
  --track data/tracks/books-wikidata-v2 --order eliminate --hypotheses broad \
  --output out/scripted_eliminating_broad_v4.json
uv run python analysis/books-controlled/score_label_agreement.py \
  --output out/controller_eval_second_labeller.json
uv run python analysis/books-controlled/extract_disclosure_turns.py   # blind packets and key
uv run python analysis/books-controlled/score_disclosure_misreads.py \
  --output out/disclosure_misread_bound.json
```

The outputs then equal the committed files of the same names in
`analysis/books-wikidata-v2/`. No LLM is called. `redial_disclosure.json` needs
the ReDial corpus (Li et al., 2018; CC BY 4.0), which is not redistributed here:
`uv run python analysis/books-controlled/redial_disclosure.py --redial <dir with
train_data.jsonl and test_data.jsonl> --output out/redial_disclosure.json`.
Each archive also holds the stage state files, every run manifest, the stalled
first attempts that were re-run (under `quarantine/`), and, except for the
ask-and-verify follow-up, the launcher logs. Account identifiers and local
network addresses that provider error messages echo back are replaced with
`<redacted>`.

The main run's manifests record the harness commit its trials ran on
(`53ff968`). The follow-ups' record `4a1bfc0`, `ddcd2ee`, `125f9dd`, `102c3f9`
and `0f2c8fd`, which add the opt-in agent prompt variants and provider support
outside the scored path; the protocol follow-up's manifests also flag modified
files, which lay outside `src/` and the track (the launcher refuses to run
otherwise). The plans record the commits they were frozen on (`8ca4808`,
`b8d1c65`, `c170007`, `5e2e114`, `a48cfc4` and `432d745` for the ask-and-verify
follow-up, and `e1d01c4`). These commits, and the launcher commits the state files
record, are not in this repository's history. The released `src/` differs from them only in ways that leave the
scored prompts, tools, rule-based user, and evaluator unchanged, which
`scripts/verify_books_archive.py` checks directly: on every extracted archive
it re-executes each recorded tool call, re-scores each trace, and replays the
rule-based user, and every check reproduces.

```bash
uv run python scripts/verify_books_archive.py --root out/books-wikidata-v4 \
  --root out/books-wikidata-v4-addendum --root out/books-wikidata-v4-ablation \
  --root out/books-wikidata-v4-prompt-open --root out/books-wikidata-v4-verify \
  --root out/books-wikidata-v4-protocol
```

`experiment_plan_v4.json`, `experiment_plan_v4_addendum.json`,
`experiment_plan_v4_ablation.json`, `experiment_plan_v4_prompt_open.json`,
`experiment_plan_v4_verify.json`, `experiment_plan_v4_protocol.json`,
`model_config_v4.json`, and
`model_config_v4_followup.json` are redacted copies: internal notes were removed,
and the artifact hashes the plans pin were updated to match. The original plans' SHA-256 values are recorded in the stage
state files (main run `abb29802…`, also in `v4_results.json`; addendum
`2f4c0354…`; field-list follow-up `845742df…`; GLM/Qwen prompt-variant
follow-up `011cd906…`; ask-and-verify follow-up `671d9461…`, then
`197ed485…`, both in its state file; protocol follow-up `f0b6de98…`); the original model
configurations' were `675f3a71…` and `b88bf28b…`. Stages, the agents in every stage,
pins, runtime limits, and the analysis specifications are unchanged.

## Running new experiments

`scripts/run_books_protocol.py` runs one agent under the reportable protocol,
and `scripts/report_books_run.py --run-dir <run>` reports it: pass^1 by
condition (hidden satisfiable and no-valid-answer tasks separately) with
family-bootstrap intervals, hidden success within three rejected proposals,
pass^4, elicitation, exposure, and stop classes.
`run_v4_matrix.py` records how the reported matrix was executed; its
`--execute` refuses unless the harness matches the commit the plan was frozen
on, so a new study should freeze its own plan with `make_v4_experiment_plan.py`.
