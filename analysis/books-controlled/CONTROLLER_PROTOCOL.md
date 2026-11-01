# Rule-based books user protocol

Version: `book-disclosure-v4`; candidate verdicts: `candidate-id-v5`

The controller owns task facts, disclosure state, and candidate verdicts. No
language model is called with the rule-based user (`--simulator-mode symbolic`). Every transition is stored
in each raw trace under `simulator_decisions`.

| Input state/event | Transition | User response |
|---|---|---|
| First turn | Mark every `volunteer` predicate disclosed; in the volunteer variant also the accepted formats | State the volunteer predicates with fixed templates; the volunteer variant also states its accepted formats; every reading-history variant asks for an unread book |
| Question about a field with stated or askable predicates | Mark the field's undisclosed `on_ask` predicates disclosed | "To answer your question:" and every non-hidden predicate of the field, new or already stated |
| Question about a field holding a `hidden` predicate | No change | "I'd rather not say." ("I'd rather not say more about the genre." when part of the field was stated) |
| Question about a field with no predicate | No change | "No preference on that." |
| Question about formats | Mark the accepted formats disclosed | State the accepted formats, again if asked again |
| Generic request for more requirements | Mark the next undisclosed `on_ask` predicate disclosed | State exactly that predicate; "That's everything I can share." when none is left |
| Declarative or unrecognized utterance | No change | Ask the agent to ask a narrowing question |
| One `[[candidate_id:ID]]` marker | No disclosure change | ACCEPT only if every predicate holds, the book is not in the reading history (when that policy applies), and, once the user has named formats, it has one of them; otherwise the generic rejection. Questions in the same turn are answered after the verdict |
| Several markers, or a truncated one | No change | Ask for one marker, or for a valid item ID |
| `[[candidate_id:NONE]]` | No change | State that the user cannot verify catalog exhaustiveness; never reveal task satisfiability |

`hidden` predicates are never disclosed. They affect only candidate
acceptance. A rejected candidate receives a generic message that never names
the failed check; each verdict decision records the candidate and the failed
checks for analysis. Verdicts come only from candidate markers: a title named
in prose never draws one. The same verdict rules and text apply to the rule-based
and LLM users, so simulator identity is not confounded with feedback.

Policy antecedents remain observable: the unread-book requirement is stated on
the opening turn, and accepted formats are stated in the volunteer opening or
on a direct format question. Catalog history and availability themselves
still require tools.

Question interpretation is a frozen lexical map in
`src/tau_rec/simulator/user_simulator.py`. It recognizes seven book fields
(`genres`, `languages`, `countries`, `pages`, `rating`, `publication_year`,
and `formats`) plus a generic-question class. This track's tasks constrain
genre, language of work, country of origin, and publication year; no task
constrains pages or rating, and format remains observable policy state. Unknown paraphrases
deterministically disclose nothing. The paraphrase suite is pinned in
`tests/test_simulator.py` and `tests/test_books_v4_contract.py`.

## Recogniser evaluation

Agent turns from development runs with the rule-based user (mixed and hidden
variants) were labelled blind by an LLM, not by people. The
recogniser was tuned on earlier labelled samples; a held-out set of 200 turns
(50 per agent model), labelled after it was frozen, gives the reported estimate.
On those 200 turns, field precision is 0.898 and recall 0.944 (203 of 226
predicted fields correct, 203 of 215 labelled fields found), micro-averaged over
the five fields; per agent model, precision is 0.82-0.94 and recall 0.90-0.98.
Open requests: precision 1.00, recall 0.77 (23 of 30). Known gaps: descriptive
pitches without a genre word ("would you enjoy an epic quest?"), alternative
titles offered with genre labels, and open requests phrased outside the list
("is there something specific I should adjust?").
`analysis/books-controlled/score_controller_eval_v4.py` reproduces the numbers
from `analysis/books-wikidata-v2/controller_eval_labels_v4.json`.

## Retrieval arms

- `keyword`: the weighted per-field BM25 `search_catalog(query)` interface.
- `structured`: a typed `search_catalog` interface over genres, languages,
  countries, publication-year bounds, format, ordering, and pagination.

The structured arm scans the complete active catalog prefix and returns the
same list-shaped, at-most-20 result envelope as keyword search; `offset`
supports pagination. It does not construct a BM25 index. Run manifests record
`retrieval_mode`, `catalog_size`, `books-wikidata-core-v3`, the parent-label
taxonomy version, and the hash of the exact catalog-prefix IDs.
