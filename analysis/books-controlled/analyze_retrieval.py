#!/usr/bin/env python3
"""Measure retrieval diagnostics on independently sampled matched families."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tau_rec.catalog.search import BM25F_PROFILE, CatalogSearch
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import RevealTag, Task, TaskConstraint
from tau_rec.domain import BOOKS


def _available(task: Task, stage: str) -> list[TaskConstraint]:
    if stage == "opening":
        return [tagged for tagged in task.constraints if tagged.reveal == RevealTag.VOLUNTEER]
    if stage == "perfect_elicitation":
        return [tagged for tagged in task.constraints if tagged.reveal != RevealTag.HIDDEN]
    if stage == "full_information":
        return list(task.constraints)
    raise ValueError(f"unknown disclosure stage {stage!r}")


def _satisfies(item, constraints: list[TaskConstraint]) -> bool:
    return all(tagged.constraint.evaluate(item) for tagged in constraints)


def _success(task: Task, item) -> bool:
    if task.no_valid_recommendation:
        return item is None
    return item is not None and _satisfies(item, list(task.constraints))


def _constraint_key(tagged: TaskConstraint) -> tuple:
    constraint = tagged.constraint
    value = tuple(constraint.value) if isinstance(constraint.value, list) else constraint.value
    return constraint.field, constraint.op, value


class PredicateIndex:
    """Bitset index shared by every size/reveal/stage analysis cell."""

    def __init__(self, catalog: Catalog, tasks: dict[str, Task]) -> None:
        representatives: dict[tuple, TaskConstraint] = {}
        for task in tasks.values():
            for tagged in task.constraints:
                representatives.setdefault(_constraint_key(tagged), tagged)
        self.masks: dict[tuple, int] = {}
        for key, tagged in representatives.items():
            mask = 0
            for index, item in enumerate(catalog.items):
                if tagged.constraint.evaluate(item):
                    mask |= 1 << index
            self.masks[key] = mask

    def select(
        self,
        catalog: Catalog,
        constraints: list[TaskConstraint],
        ranked_indices: list[int],
    ):
        mask = (1 << len(catalog)) - 1
        for tagged in constraints:
            mask &= self.masks[_constraint_key(tagged)]
            if not mask:
                return None
        return next(
            (catalog.items[index] for index in ranked_indices if mask & (1 << index)),
            None,
        )


def _keyword_filter(
    catalog: Catalog,
    search: CatalogSearch,
    constraints: list[TaskConstraint],
    cache: dict[str, list[str]],
):
    values = {field: [] for field in ("genres", "languages", "countries")}
    for tagged in constraints:
        constraint = tagged.constraint
        if constraint.field in values and constraint.op == "contains":
            values[constraint.field].append(str(constraint.value))
    queries: list[str] = []
    for terms in (
        values["genres"] + values["languages"] + values["countries"],
        values["genres"] + values["languages"],
        values["genres"] + values["countries"],
        values["genres"],
    ):
        query = " ".join(terms)
        if query and query not in queries:
            queries.append(query)
    if not queries:
        return None
    for query in queries[:3]:
        if query not in cache:
            cache[query] = [row["id"] for row in search.search(query, top_k=20)]
        match = next(
            (
                item for item_id in cache[query]
                if (item := catalog.get(item_id)) is not None
                and _satisfies(item, constraints)
            ),
            None,
        )
        if match is not None:
            return match
    return None


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[int]] = defaultdict(list)
    for row in rows:
        for complexity in (row["complexity"], "all"):
            key = (
                row["catalog_size"], row["stage"], row["retriever"],
                row["reveal"], complexity,
            )
            grouped[key].append(int(row["success"]))
    return [
        {
            "catalog_size": key[0],
            "stage": key[1],
            "retriever": key[2],
            "reveal": key[3],
            "complexity": key[4],
            "n": len(values),
            "correct": sum(values),
            "agreement": sum(values) / len(values),
        }
        for key, values in sorted(grouped.items())
    ]


def _markdown(summary: list[dict], metadata: dict) -> str:
    lines = [
        "# Controlled retrieval diagnostics",
        "",
        "These are deterministic retrieval diagnostics, not conversational-agent scores.",
        "Tasks were selected without consulting either retriever.",
        "",
        f"- Catalog SHA-256: `{metadata['catalog_sha256']}`",
        f"- Families: {metadata['family_count']}",
        f"- Catalog sizes: {', '.join(f'{value:,}' for value in metadata['catalog_sizes'])}",
        "",
        "`opening` uses only volunteer facts. `perfect_elicitation` adds every on-ask fact",
        "but never hidden facts. `full_information` is an oracle manipulation check.",
        "",
        "| Catalog | Stage | Retriever | Reveal | Complexity | Correct | Rate |",
        "|---:|---|---|---|---|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['catalog_size']:,} | {row['stage']} | {row['retriever']} | "
            f"{row['reveal']} | {row['complexity']} | {row['correct']}/{row['n']} | "
            f"{row['agreement']:.3f} |"
        )
    lines.extend([
        "",
        "The typed-filter result is a deterministic retrieval ceiling, not an agent baseline.",
        "The keyword result uses up to three fixed BM25 top-20 queries and then applies",
        "every currently available predicate. Policy compliance is intentionally excluded.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.artifact.parent / "retrieval_diagnostics.json"
    task_manifest = json.loads((args.artifact / "task_manifest.json").read_text())
    family_manifest = json.loads((args.artifact / "family_manifest.json").read_text())
    full_catalog = Catalog.from_json(str(args.catalog), BOOKS)
    tasks = {
        path.stem: Task.from_json(str(path))
        for path in sorted((args.artifact / "tasks").glob("*.json"))
    }
    predicate_index = PredicateIndex(full_catalog, tasks)

    rows: list[dict] = []
    for size in task_manifest["catalog_sizes"]:
        catalog = Catalog(items=full_catalog.items[:size])
        search = CatalogSearch(catalog, BOOKS, profile=BM25F_PROFILE)
        def popularity(item) -> int:
            return item.vote_count if item.vote_count is not None else item.sitelink_count

        def rating(item) -> float:
            return item.rating if item.rating is not None else float("-inf")

        popularity_order = sorted(
            range(len(catalog)),
            key=lambda index: (
                -popularity(catalog.items[index]),
                -rating(catalog.items[index]),
                catalog.items[index].id,
            ),
        )
        query_cache: dict[str, list[str]] = {}
        typed_cache: dict[tuple, object] = {}
        for family in family_manifest["families"]:
            for reveal, task_id in family["task_ids"].items():
                task = tasks[task_id]
                for stage in ("opening", "perfect_elicitation", "full_information"):
                    constraints = _available(task, stage)
                    constraint_key = tuple(_constraint_key(tagged) for tagged in constraints)
                    if constraint_key not in typed_cache:
                        typed_cache[constraint_key] = predicate_index.select(
                            catalog, constraints, popularity_order
                        )
                    selected = {
                        "keyword_top20x3+visible_filter": _keyword_filter(
                            catalog, search, constraints, query_cache
                        ),
                        "typed_filter": typed_cache[constraint_key],
                    }
                    for retriever, item in selected.items():
                        rows.append({
                            "catalog_size": size,
                            "family_id": family["family_id"],
                            "task_id": task_id,
                            "complexity": task.complexity,
                            "reveal": reveal,
                            "stage": stage,
                            "retriever": retriever,
                            "selected_id": item.id if item is not None else None,
                            "success": _success(task, item),
                        })

    summary = _aggregate(rows)
    result = {
        "protocol": {
            "task_sampling": task_manifest["sampling_mode"],
            "retrieval_role": task_manifest["retrieval_role"],
            "catalog_sha256": task_manifest["catalog_sha256"],
            "catalog_sizes": task_manifest["catalog_sizes"],
            "family_count": family_manifest["family_count"],
            "keyword_top_k": 20,
            "keyword_query_budget": 3,
            "keyword_query_strategy": (
                "up to three fixed queries over visible genre, language, and country values"
            ),
            "typed_sort": (
                "vote_count desc when present, otherwise sitelink_count desc; "
                "rating desc when present; id asc"
            ),
            "policy_scoring": False,
        },
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(summary, result["protocol"]))
    print(f"Wrote {len(rows):,} deterministic rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
