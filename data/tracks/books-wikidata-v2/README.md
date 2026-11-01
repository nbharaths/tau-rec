# `books-wikidata-v2`: a controlled books track

A τ-Rec track for conversational book recommendation with verifiable outcomes.
An agent talks to a simulated reader and must recommend one catalog book that
satisfies every hard requirement, or abstain when the catalog has none.

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

The reportable protocol, used for the reported books results, is the
51-family primary panel (153 tasks) at 20k with the rule-based user, 20 agent
turns, 600 s of agent time per trial, and four trials per task:

```bash
uv run python scripts/run_books_protocol.py \
  --model openrouter/deepseek/deepseek-v4.1-flash --provider-order deepinfra/fp8 \
  --retrieval-mode structured --reasoning-effort high --max-provider-cost-usd 20 --execute
```

Without `--execute` the script prints the `tau-rec run` command. `--block
reliability` runs the nested 18-family block with eight trials per task.
Report `pass^1` by disclosure condition (hidden satisfiable and no-valid-answer
tasks separately), `pass^4`, hidden success within three rejected proposals,
elicitation, exposure, and stop classes (`scripts/report_books_run.py --run-dir
<run>` computes them), and compare with `primary_table` in
`analysis/books-wikidata-v2/v4_results.json`. Because the agents' order changes with the prompt, also run with
`--agent-prompt-variant ask-and-verify-v1`, which asks agents to elicit
requirements and to propose a book for a verdict before recommending it. Structured volunteer cells are
saturated; mixed, hidden, and keyword cells discriminate between agents. We
suggest tuning on the 99 families outside the primary panel
(`analysis/books-wikidata-v2/panels.json`) and reporting non-US/UK and
non-English requirements separately.

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
