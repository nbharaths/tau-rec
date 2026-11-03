# τ-Rec-Books (`books-wikidata-v2`)

This folder is the resource described in *τ-Rec-Books: A Disclosure-Controlled
Benchmark for Requirement Elicitation in Agentic Recommender Systems* (under
review at ECIR 2027). An agent talks to a simulated reader and must recommend
one catalog book that satisfies every hard requirement, or abstain when the
catalog has none. Each request is fixed; only *when* the reader reveals its
requirements varies.

- **Catalog:** 20,000 literary works derived from Wikidata (CC0), ordered by a
  seeded hash so that the first 5,000 and 10,000 works form nested smaller
  catalogs. Every 5k answer is still present at 10k and 20k.
- **Tasks:** 150 request families, each with two to five hard requirements on
  genre, language, country of origin, and publication year. Each family comes
  in three disclosure variants with identical requirements and answer set
  (450 tasks):
  - `volunteer`: the reader states every requirement up front;
  - `mixed`: the reader states one and reveals the rest only when asked;
  - `hidden`: the reader also withholds at least one requirement and only
    accepts or rejects proposed books.
- **No-valid-recommendation families:** 15 families have no answer in the
  catalog, though dropping any single requirement leaves at least three
  matches in the full 20,000-work catalog. The agent should abstain on them.
- **Rule-based user:** a deterministic disclosure controller
  (`book-disclosure-v4`) answers field questions, declines hidden ones, and
  gives accept/reject verdicts for candidates the agent marks with
  `[[candidate_id:ID]]`. No language model runs on the user side. See
  [`analysis/books-controlled/CONTROLLER_PROTOCOL.md`](../../../analysis/books-controlled/CONTROLLER_PROTOCOL.md).
- **Two retrieval interfaces:** `keyword` (per-field BM25, top 20) and
  `structured` (conjunctive filters on genre, language, country, year range,
  and formats, pages of 20).

## Results

The paper's main results (its Table 3), on the primary panel at 20k with the
rule-based user, the base prompt, high reasoning effort, and four trials per
task. Volunteer, mixed, and hidden are `pass^1` on the 45 satisfiable families;
hidden NVR is `pass^1` on the 6 hidden no-valid-recommendation tasks; the "All"
columns cover all 153 tasks; *Elic.* is the share of mixed trials that elicit
every askable fact. Bold marks the best LLM agent per column.

| Agent | Interface | Volunteer | Mixed | Hidden | Hidden NVR | All, pass^1 | All, pass^4 | Elic. |
|---|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | keyword | 0.989 | 0.872 | 0.297 | 0.000 | 0.691 | 0.609 | 0.576 |
|  | structured | **1.000** | **0.967** | 0.358 | 0.000 | **0.760** | **0.666** | 0.859 |
| GLM 5.3 Flash | keyword | 0.978 | 0.869 | 0.325 | **0.167** | 0.714 | 0.574 | 0.569 |
|  | structured | **1.000** | 0.947 | **0.367** | 0.042 | 0.756 | 0.627 | 0.641 |
| Qwen 3.8 27B | keyword | 0.842 | 0.569 | 0.294 | **0.167** | 0.550 | 0.381 | 0.377 |
|  | structured | **1.000** | 0.942 | 0.325 | 0.000 | 0.737 | 0.625 | 0.656 |
| GPT-6.1 Sol | keyword | 0.994 | 0.458 | 0.139 | 0.000 | 0.523 | 0.456 | 0.207 |
|  | structured | **1.000** | 0.419 | 0.156 | 0.000 | 0.511 | 0.424 | 0.159 |
| LLM average | both | 0.975 | 0.756 | 0.283 | 0.047 | – | – | – |
| Scripted popularity agent | structured | 1.000 | 1.000 | 0.622 | 0.000 | 0.850 | 0.850 | – |
| Scripted eliminating agent | structured | 1.000 | 1.000 | 0.956 | 0.833 | 0.980 | 0.980 | – |
| No-ask floor | – | 1.000 | 0.067 | 0.067 | 0.000 | 0.373 | 0.373 | – |

Under the paper's prompt variants, on the 18-family reliability block with
structured search: told to ask (`ask-requirements-v1`), every agent reaches at
least 0.931 on mixed tasks; told to propose a book before recommending it
(`ask-and-verify-v1`), agents are not detectably different from the popularity
agent on hidden tasks; told also what a declined question and a rejection mean
(`ask-verify-protocol-v1`), they gain a further 22.5 points on hidden tasks but
all stay below the eliminating agent.

## Where the paper's pieces live

| Paper | In this repository |
|---|---|
| Catalog, tasks, answer key, and policy (Sect. 3) | this folder ([Files](#files)) |
| Rule-based user (Sect. 3) | [`CONTROLLER_PROTOCOL.md`](../../../analysis/books-controlled/CONTROLLER_PROTOCOL.md), `src/tau_rec/simulator/user_simulator.py` |
| Agents, interfaces, prompts, and caps (Sect. 4) | `scripts/run_books_protocol.py` ([Running the reportable protocol](#running-the-reportable-protocol)) |
| Validators and reference agents (Sect. 5) | [Validating](#validating), [Reference agents](#reference-agents) |
| Every reported number (Sect. 6) | `analysis/books-wikidata-v2/` and the traces on the [v2.0.0 release](https://github.com/nbharaths/tau-rec/releases/tag/v2.0.0) ([Reproducing the paper's numbers](#reproducing-the-papers-numbers)) |

## Files

| File | Contents |
|---|---|
| `catalog.json` | The 20,000-work catalog |
| `source_selection.jsonl`, `catalog_manifest.json`, `source_provenance.json` | Source records, build parameters, and provenance |
| `label_overrides.json` | Documented corrections of source-label errors |
| `source_snapshot_asset.json` | Identity of the source snapshot release asset |
| `tasks/` | The 450 task files |
| `answers.json` | Answer key: valid and policy-eligible answers per task |
| `family_manifest.json`, `task_manifest.json`, `task_baseline.json`, `lineage.json` | Family structure, generation rules, and content hashes |
| `policy.md` | The policy shown to the agent |
| `audit_sample.json`, `catalog_audit_sample.json`, `task_audit_sample.json` | The frozen samples used for the automated and human audits (results in `analysis/books-wikidata-v2/`) |
| `track.json` | Track metadata (domain, harness profile, retrieval profiles) |

## Running the reportable protocol

The paper's results use the 51-family primary panel (153 tasks) at 20k with the
rule-based user, 20 agent turns, 600 s of agent time per trial, and four trials
per task:

```bash
uv run python scripts/run_books_protocol.py \
  --model openrouter/deepseek/deepseek-v4.1-flash --provider-order deepinfra/fp8 \
  --retrieval-mode structured --reasoning-effort high --max-provider-cost-usd 20 --execute
```

- Without `--execute`, the script prints the `tau-rec run` command instead.
- `--retrieval-mode keyword` runs the keyword interface. `--block reliability`
  runs the nested 18-family reliability block with eight trials per task.
- Run two prompts: the base prompt and `--agent-prompt-variant ask-and-verify-v1`,
  which asks agents to elicit requirements and to propose a book for a verdict
  before recommending it. Agents rank differently under the two. The paper's
  other variants are `ask-requirements-v1` and `ask-verify-protocol-v1`.
- `scripts/report_books_run.py --run-dir <run>` reports `pass^1` by disclosure
  condition (hidden satisfiable and no-valid-answer tasks separately), `pass^4`,
  hidden success within three rejected proposals, elicitation, exposure, and how
  trials end. Compare with `primary_table` in
  `analysis/books-wikidata-v2/v4_results.json`.
- Structured volunteer cells are saturated; mixed, hidden, and keyword cells
  discriminate between agents.
- Tune on the 99 families outside the primary panel
  (`analysis/books-wikidata-v2/panels.json`), and report non-US/UK and
  non-English requirements separately.

## Reference agents

Two scripted agents run through the same user, tools, scoring, and caps, with
no language model. The popularity agent proposes the most popular match until
one is accepted; the eliminating agent also uses each decline and rejection to
rule out possible requirements:

```bash
uv run python analysis/books-controlled/run_scripted_reference.py \
  --track data/tracks/books-wikidata-v2 --output out/scripted_reference_v4.json
uv run python analysis/books-controlled/run_scripted_reference.py \
  --track data/tracks/books-wikidata-v2 --order eliminate --hypotheses broad \
  --output out/scripted_eliminating_broad_v4.json
```

## Reproducing the paper's numbers

Every trace behind the paper is attached to the
[v2.0.0 release](https://github.com/nbharaths/tau-rec/releases/tag/v2.0.0).
Download and check the six run archives, then extract them at the repository
root:

```bash
gh release download v2.0.0 -R nbharaths/tau-rec -p 'books-wikidata-v4*.tar.gz' -p SHA256SUMS
shasum -a 256 -c SHA256SUMS --ignore-missing
for f in books-wikidata-v4*.tar.gz; do tar -xzf "$f"; done
```

`scripts/verify_books_archive.py` re-executes every recorded tool call,
re-scores every trace, and replays the rule-based user. The scripts in
[`analysis/books-controlled/`](../../../analysis/books-controlled/README.md#reproducing-the-reported-numbers)
then regenerate every reported number from the traces and the committed labels,
with no model calls.

## Validating

```bash
uv run tau-rec validate --track data/tracks/books-wikidata-v2 \
  --baseline data/tracks/books-wikidata-v2/task_baseline.json
uv run python scripts/validate_books_controlled.py \
  --catalog data/tracks/books-wikidata-v2/catalog.json --artifact data/tracks/books-wikidata-v2
```

## Rebuilding from the source snapshot

The catalog, tasks, and answer key rebuild byte for byte, offline, from the 21
hash-pinned Wikidata query responses published as
`books-wikidata-v2-source-snapshot.tar.gz` (sha256 in
`source_snapshot_asset.json`):

```bash
tar -xzf books-wikidata-v2-source-snapshot.tar.gz
uv run python scripts/export_wikidata_books_catalog.py --offline \
  --audit-cache books-wikidata-v2-source-snapshot/cache --output-track rebuilt \
  --catalog-profile v2 --label-overrides data/tracks/books-wikidata-v2/label_overrides.json
uv run python scripts/generate_books_controlled.py --catalog rebuilt/catalog.json \
  --output rebuilt --task-profile wikidata-v2 --track-id books-wikidata-v2 \
  --source-provenance data/tracks/books-wikidata-v2/source_provenance.json
```

`rebuilt/catalog.json`, `source_selection.jsonl`, `tasks/`, `answers.json`,
`family_manifest.json`, `task_manifest.json`, and `task_baseline.json` then
match the committed files; `catalog_manifest.json` differs only in its build
timestamp and `lineage.json` only in the catalog path it records.

## Known issues

- 60 records (18 of them task answers) have a title, description, author, or
  publisher name cut off at a double quote inside the Wikidata text, from a
  parsing defect in the export. Scoring reads identifiers and the genre,
  language, country, year, and format fields, so it is unaffected. The records
  stay as they are because the reported results used this catalog.

## Licensing

The catalog, source selection, and source snapshot are CC0 1.0 (Wikidata
structured data). Tasks, answers, policies, manifests, and documentation are
CC BY 4.0. Code is MIT. See [`LICENSE.md`](LICENSE.md).
