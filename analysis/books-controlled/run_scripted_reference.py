#!/usr/bin/env python3
"""Run a protocol-aware scripted agent (no LLM) through the real books harness.

The agent asks once about every field, parses the symbolic user's templated
disclosures, filters with the structured tool, removes previously read works,
proposes ``[[candidate_id:...]]`` candidates until one is accepted, checks the
accepted item's metadata, and commits.  It is an end-to-end solvability check
and an attainable reference score.  Because it relies on the controller's field
vocabulary and reply templates, it is not evidence of natural-language
robustness.

With ``--order eliminate`` it reads which fields the user refuses to discuss and,
after each reasonless rejection, rules out every hypothesis about those fields
that the rejected book satisfies, proposing next the book that fits the most
remaining hypotheses (see ``Hypotheses``).  It abstains once no remaining
hypothesis fits any candidate.  ``--hypotheses broad`` drops the generator's
template shapes: a refused field may hold any one or two filter values.

The agent is deterministic, so one trial per task is sufficient:

    uv run python analysis/books-controlled/run_scripted_reference.py \\
        --track data/tracks/books-wikidata-v2 \\
        --output analysis/books-wikidata-v2/scripted_reference_v4.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


import tau_rec
from tau_rec.agents.base import AgentResponse, BaseAgent
from tau_rec.catalog.taxonomy import expand_parent_labels
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Constraint, Task

from tau_rec.domain import BOOKS
from tau_rec.environment.tools import ToolKit
from tau_rec.evaluator.evaluator import CombinedEvaluator
from tau_rec.orchestrator.orchestrator import Orchestrator
from tau_rec.simulator.user_simulator import UserSimulator
from tau_rec.track import load_track

QUESTION = (
    "Do you have any preferences about genre, language, country of origin, "
    "publication year, or format?"
)
CLAUSE = re.compile(
    r"(its language should include|its country of origin should include|it should include"
    r"|it should be available as|it should have been published no earlier than"
    r"|it should have been published no later than) (.+?)"
    r"(?=; and |\.\s|\.$| I can use| I want)",
    re.IGNORECASE,
)
FORMATS = re.compile(r"I can use any of these formats: ([^.]+)\.")
REFUSAL = re.compile(
    r"I'd rather not say(?: more)? about "
    r"(the genre|the language|the country of origin|the publication year)\."
)
REFUSED_FIELDS = {
    "the genre": "genres", "the language": "languages",
    "the country of origin": "countries", "the publication year": "publication_year",
}
YEAR = "publication_year"
PAGE_SIZE = 20
# 140 result pages, within the books-v4 harness's 150 tool calls per step.
MAX_CANDIDATES = 2800
LAST_PROBE_STEP = 19  # the next step must be the terminal recommendation


def refused_fields(text: str) -> list[str]:
    """Fields the user declines to discuss, so each holds a hidden requirement."""
    return [REFUSED_FIELDS[name] for name in REFUSAL.findall(text)]


class Hypotheses:
    """What the refused fields may require, ruled out by rejected proposals.

    Hypotheses follow the generator's templates.  A refused language or
    country requires one label; a refused genre one label, or two when no genre
    was stated; a refused year one bound, or a range when no bound was stated,
    and only the open bound when one was.  With two refused fields, each holds
    one predicate (families hide at most two).  Every proposal meets the
    stated facts, so a rejection rules out every hypothesis the rejected book
    satisfies.  One- and two-predicate hypotheses carry equal total weight,
    uniform within each; hypotheses no candidate satisfies are dropped.

    With ``broad`` the hypotheses assume only what the structured filters can
    express: each refused field holds one or two filter values (two labels, or
    a year range), whatever was stated, and two refused fields combine freely.
    Each shape (values per field) carries equal total weight.
    """

    def __init__(
        self, rows: list[dict], fields: list[str], stated_genres=(),
        stated_min: int | None = None, stated_max: int | None = None, broad: bool = False,
    ) -> None:
        self.ids = [row["id"] for row in rows]
        items = [
            SimpleNamespace(**{field: row.get(field) for field in ("genres", "languages", "countries", YEAR)})
            for row in rows
        ]
        # Each block is (left, right, live): a candidate c satisfies hypothesis
        # (i, j) when left[c, i] and right[c, j]; single predicates pair with a
        # constant column.
        self.blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if broad:
            shapes = [self._shapes(field, items, stated_genres) for field in fields[:2]]
            if len(shapes) == 1:
                (singles, pairs), = shapes
                self._single(singles)
                self._pairs(*pairs)
            else:
                (first, second) = ([singles, self._materialize(*pairs)] for singles, pairs in shapes)
                for left in first:
                    for right in second:
                        self._pairs(left, right, np.ones((left.shape[1], right.shape[1]), dtype=bool))
        elif len(fields) == 1:
            (field,) = fields
            if field == YEAR:
                ge, le = self._bounds(items)
                self._single(self._predicates(YEAR, items, stated_genres, stated_min, stated_max))
                if stated_min is None and stated_max is None:
                    self._pairs(ge, le, np.triu(np.ones((ge.shape[1], le.shape[1]), dtype=bool)))
            else:
                labels = self._labels(field, items, stated_genres)
                self._single(labels)
                if field == "genres" and not stated_genres:
                    self._pairs(labels, labels, np.triu(np.ones((labels.shape[1],) * 2, dtype=bool), 1))
        else:
            left, right = (self._predicates(field, items, stated_genres, stated_min, stated_max) for field in fields[:2])
            self._pairs(left, right, np.ones((left.shape[1], right.shape[1]), dtype=bool))
        sizes = [int(live.sum()) for _, _, live in self.blocks]
        self.weights = [1.0 / (len(sizes) * size) if size else 0.0 for size in sizes]

    @staticmethod
    def _labels(field: str, items: list, stated_genres) -> np.ndarray:
        values = sorted({
            value for item in items
            for value in (expand_parent_labels(field, getattr(item, field) or []) if field != "genres"
                          else getattr(item, field) or [])
        } - (set(stated_genres) if field == "genres" else set()))
        return np.array([
            [Constraint(field=field, op="contains", value=value).evaluate(item) for value in values]
            for item in items
        ], dtype=bool).reshape(len(items), len(values))

    @staticmethod
    def _bounds(items: list) -> tuple[np.ndarray, np.ndarray]:
        years = sorted({item.publication_year for item in items if item.publication_year is not None})
        ge = np.array([
            [Constraint(field=YEAR, op=">=", value=year).evaluate(item) for year in years] for item in items
        ], dtype=bool).reshape(len(items), len(years))
        le = np.array([
            [Constraint(field=YEAR, op="<=", value=year).evaluate(item) for year in years] for item in items
        ], dtype=bool).reshape(len(items), len(years))
        return ge, le

    def _predicates(self, field, items, stated_genres, stated_min, stated_max) -> np.ndarray:
        if field != YEAR:
            return self._labels(field, items, stated_genres)
        ge, le = self._bounds(items)
        return np.hstack([part for part, stated in ((ge, stated_min), (le, stated_max)) if stated is None])

    def _shapes(self, field, items, stated_genres) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """One filter value (a label, or either year bound) and two (a label pair, or a year range)."""
        if field == YEAR:
            ge, le = self._bounds(items)
            return np.hstack([ge, le]), (ge, le, np.triu(np.ones((ge.shape[1], le.shape[1]), dtype=bool)))
        labels = self._labels(field, items, stated_genres)
        return labels, (labels, labels, np.triu(np.ones((labels.shape[1],) * 2, dtype=bool), 1))

    @staticmethod
    def _materialize(left: np.ndarray, right: np.ndarray, live: np.ndarray) -> np.ndarray:
        rows, cols = np.nonzero(live)
        return left[:, rows] & right[:, cols]

    def _single(self, matrix: np.ndarray) -> None:
        self._pairs(matrix, np.ones((len(matrix), 1), dtype=bool), np.ones((matrix.shape[1], 1), dtype=bool))

    def _pairs(self, left: np.ndarray, right: np.ndarray, live: np.ndarray) -> None:
        # float32 products are exact here (counts stay far below 2**24) and use BLAS.
        live = live & ((left.T.astype(np.float32) @ right.astype(np.float32)) > 0)
        self.blocks.append((left, right, live))

    def counts(self) -> list[np.ndarray]:
        """Per block, how many live hypotheses each candidate satisfies."""
        out = []
        for left, right, live in self.blocks:
            if right.shape[1] > left.shape[1]:  # contract the wider side first
                inner = right.astype(np.float32) @ live.T.astype(np.float32)
                out.append((inner.astype(np.float64) * left).sum(axis=1))
            else:
                inner = left.astype(np.float32) @ live.astype(np.float32)
                out.append((inner.astype(np.float64) * right).sum(axis=1))
        return out

    def _scores(self) -> np.ndarray:
        total = np.zeros(len(self.ids))
        for weight, count in zip(self.weights, self.counts()):
            total += weight * count
        return total

    def score(self, item_id: str) -> float:
        return float(self._scores()[self.ids.index(item_id)])

    def best(self) -> str | None:
        """The candidate satisfying the most remaining hypothesis weight; ``None`` if none remains."""
        scores = self._scores()
        if not len(scores) or scores.max() <= 0:
            return None
        return self.ids[int(np.argmax(scores))]

    def reject(self, item_id: str) -> None:
        index = self.ids.index(item_id)
        for left, right, live in self.blocks:
            live &= ~np.outer(left[index], right[index])


class ScriptedAgent(BaseAgent):
    def __init__(self, task: Task, order: str, seed: int, broad: bool = False) -> None:
        self.task = task
        self.order = order
        self.broad = broad
        self.rng = random.Random(seed)
        self.req: dict = {
            "genres": [], "languages": [], "countries": [], "formats": [],
            "min": None, "max": None,
        }
        self.usable_formats: list[str] = []
        self.read: set[str] = set()
        self.candidates: list[str] = []
        self.phase = "ask"
        self.steps = 0
        self.pages = 0
        self.current: str | None = None
        self.accepted: str | None = None
        self.probed = 0
        self._call_id = 0
        self.refused: list[str] = []
        self.rows: dict[str, dict] = {}
        self.hypotheses: Hypotheses | None = None


    def reset(self) -> None:
        pass

    def add_assistant_message(self, text: str) -> None:
        pass

    def add_user_message(self, text: str) -> None:
        for kind, value in CLAUSE.findall(text):
            value = value.strip()
            kind = kind.lower()
            if kind == "it should include":
                self.req["genres"].append(value)
            elif kind == "its language should include":
                self.req["languages"].append(value)
            elif kind == "its country of origin should include":
                self.req["countries"].append(value)
            elif kind == "it should be available as":
                self.req["formats"].append(value)
            elif kind.endswith("no earlier than"):
                self.req["min"] = int(value)
            elif kind.endswith("no later than"):
                self.req["max"] = int(value)
        match = FORMATS.search(text)
        if match:
            self.usable_formats = [part.strip() for part in match.group(1).split(",")]
        for field in refused_fields(text):
            if field not in self.refused:
                self.refused.append(field)
        if self.phase == "probe" and "###ACCEPTED###" in text:
            self.accepted = self.current
        elif self.phase == "probe" and "###REJECTED###" in text and self.hypotheses is not None:
            self.hypotheses.reject(self.current)


    def _call(self, name: str, **arguments) -> dict:
        self._call_id += 1
        return {"id": f"c{self._call_id}", "name": name, "arguments": arguments}

    def _message(self, text: str) -> AgentResponse:
        self.steps += 1
        return AgentResponse(message=text)

    def _filters(self) -> dict:
        filters: dict = {}
        for field in ("genres", "languages", "countries"):
            if self.req[field]:
                filters[field] = sorted(set(self.req[field]))
        if self.req["min"] is not None:
            filters["min_publication_year"] = self.req["min"]
        if self.req["max"] is not None:
            filters["max_publication_year"] = self.req["max"]
        formats = sorted(set(self.usable_formats) | set(self.req["formats"]))
        if formats:
            filters["formats"] = formats
        return filters

    async def respond(self, tool_results=None) -> AgentResponse:
        if self.phase == "ask":
            self.phase = "history"
            return self._message(QUESTION)
        if self.phase == "history":
            self.phase = "search"
            return AgentResponse(None, [self._call("get_user_history", user_id=self.task.user_id)])
        if self.phase == "search":
            return await self._search(tool_results)
        if self.phase == "probe":
            return self._probe()
        if self.phase == "commit":
            self.phase = "done"
            return AgentResponse(None, [self._call("recommend", item_id=self.accepted)])
        raise RuntimeError("respond called after the terminal recommendation")

    async def _search(self, tool_results) -> AgentResponse:
        last = tool_results[-1]["content"] if tool_results else "[]"
        try:
            payload = json.loads(last)
        except Exception:
            payload = {}
        if self.pages == 0 and isinstance(payload, dict) and "read" in payload:
            self.read = {x["id"] if isinstance(x, dict) else x for x in payload.get("read", [])}
        elif self.pages > 0:
            if isinstance(payload, list):
                rows = payload
            elif isinstance(payload, dict):
                rows = payload.get("results", [])
            else:
                rows = []
            rows = [r for r in rows if isinstance(r, dict) and "id" in r]
            self.candidates += [r["id"] for r in rows]
            self.rows.update((r["id"], r) for r in rows)
            if len(rows) < PAGE_SIZE or len(self.candidates) >= MAX_CANDIDATES:
                self.phase = "probe"
                self.candidates = [c for c in dict.fromkeys(self.candidates) if c not in self.read]
                if self.order == "random":
                    self.rng.shuffle(self.candidates)
                if self.order == "eliminate" and self.refused:
                    self.hypotheses = Hypotheses(
                        [self.rows[c] for c in self.candidates], self.refused,
                        stated_genres=self.req["genres"], stated_min=self.req["min"], stated_max=self.req["max"],
                        broad=self.broad,
                    )
                return await self.respond(None)
        self.pages += 1
        offset = (self.pages - 1) * PAGE_SIZE
        return AgentResponse(None, [self._call("search_catalog", **self._filters(), offset=offset)])

    def _probe(self) -> AgentResponse:
        if self.accepted:
            self.phase = "commit"
            return AgentResponse(None, [self._call("get_metadata", item_id=self.accepted)])
        if self.hypotheses is not None:
            return self._probe_eliminating()
        if not self.candidates:
            return AgentResponse(
                "No catalog book satisfies everything you asked for, so I will not "
                "force a partial match.",
                [self._call("recommend", item_id=None)],
            )
        if self.steps >= LAST_PROBE_STEP:
            self.accepted = self.candidates[0]
            self.phase = "commit"
            return AgentResponse(None, [self._call("get_metadata", item_id=self.accepted)])
        self.current = self.candidates.pop(0)
        self.probed += 1
        return self._message(f"How about this one: [[candidate_id:{self.current}]]?")

    def _probe_eliminating(self) -> AgentResponse:
        best = self.hypotheses.best()
        if best is None:
            return AgentResponse(
                "No catalog book satisfies everything you asked for, so I will not "
                "force a partial match.",
                [self._call("recommend", item_id=None)],
            )
        if self.steps >= LAST_PROBE_STEP:
            self.accepted = best
            self.phase = "commit"
            return AgentResponse(None, [self._call("get_metadata", item_id=self.accepted)])
        self.current = best
        self.candidates.remove(best)
        self.probed += 1
        return self._message(f"How about this one: [[candidate_id:{self.current}]]?")



def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


async def _run(track: Path, profile: str, order: str, broad: bool = False) -> list[dict]:
    catalog = Catalog.from_json(str(track / "catalog.json"), spec=BOOKS)
    evaluator = CombinedEvaluator(catalog, BOOKS)
    harness = load_track(track).harness
    rows = []
    for path in sorted(track.glob("tasks/*.json")):
        task = Task.model_validate(json.loads(path.read_text()))
        toolkit = ToolKit(
            catalog=catalog, search=None, user_histories=task.user_history or {},
            spec=BOOKS, mode="tools", retrieval_mode="structured", structured_profile=profile,
        )
        simulator = UserSimulator(
            model="none", task=task, spec=BOOKS, catalog=catalog, interaction_mode="symbolic",
        )
        agent = ScriptedAgent(task, order, seed=sum(map(ord, task.id)), broad=broad)
        trace = await Orchestrator(
            agent=agent, simulator=simulator, toolkit=toolkit, max_turns=20, spec=BOOKS,
            harness=harness,
        ).run(task.id, "scripted", 0)
        result = evaluator.evaluate(task, trace)
        rows.append({
            "task_id": task.id,
            "reveal": task.reveal_difficulty,
            "nvr": task.no_valid_recommendation,
            "cx": task.complexity,
            "success": result.primary_reward >= 1 - 1e-9,
            "constraint": result.constraint_score,
            "policy": result.policy_score,
            "violations": result.policy_detail.violations,
            "probes": agent.probed,
            "stop": str(trace.stop_reason),
            "n_cands": len(agent.candidates) + agent.probed,
        })
    return rows


def _summary(rows: list[dict]) -> dict:
    cells: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        status = "nvr" if row["nvr"] else "satisfiable"
        cells[f"{row['reveal']}/{status}"].append(row["success"])
    return {
        key: {"success": sum(values), "tasks": len(values), "rate": sum(values) / len(values)}
        for key, values in sorted(cells.items())
    }



def _repo_relative(path: Path) -> str:
    """A path relative to the repository root, or its name when outside it."""
    try:
        return path.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
    except ValueError:
        return path.name

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="books-wikidata-core-v3")
    parser.add_argument("--order", choices=("tool", "random", "eliminate"), default="tool")
    parser.add_argument("--hypotheses", choices=("template", "broad"), default="template",
                        help="with --order eliminate: the generator's template shapes, or any one or two filter values")
    args = parser.parse_args()

    rows = asyncio.run(_run(args.track, args.profile, args.order, broad=args.hypotheses == "broad"))
    task_files = sorted(args.track.glob("tasks/*.json"))
    report = {
        "meta": {
            "tau_rec": _repo_relative(Path(tau_rec.__file__).parent),
            "track": _repo_relative(Path(args.track)),
            "catalog_sha256": _sha256([args.track / "catalog.json"]),
            "tasks_sha256": _sha256(task_files),
            "structured_profile": args.profile,
            "candidate_order": args.order,
            **({"hypotheses": args.hypotheses} if args.order == "eliminate" else {}),
            "trials_per_task": 1,
        },
        "summary": _summary(rows),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for key, cell in report["summary"].items():
        print(f"{key:24} {cell['success']:3}/{cell['tasks']:3}  {cell['rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
