"""Deterministic native-task authoring for the books track."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tau_rec.catalog.search import BM25F_PROFILE, CatalogSearch
from tau_rec.catalog.taxonomy import COUNTRY_PARENTS_V2, LANGUAGE_PARENTS_V2
from tau_rec.catalog.validator import CatalogValidator, task_signature
from tau_rec.catalog.wikidata_books import PUBLIC_BOOK_GENRES
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Constraint, RevealTag, Task, TaskConstraint
from tau_rec.domain import BOOKS
from tau_rec.leaderboard.hashing import hash_file

WIKIDATA_V2_TASK_PROFILE = "wikidata-v2"
WIKIDATA_V2_GENERATOR_VERSION = "books-wikidata-native-v3"
TASK_PROFILES = (WIKIDATA_V2_TASK_PROFILE,)
# Genre pairs a reader treats as overlapping but the source rarely co-labels;
# requiring both makes failures label artifacts rather than retrieval errors.
OVERLAPPING_GENRE_PAIRS = frozenset({
    frozenset({"Nonfiction", "Memoir"}),
    frozenset({"Nonfiction", "Biography"}),
    frozenset({"Nonfiction", "History"}),
    frozenset({"Nonfiction", "Self-Help"}),
    frozenset({"Biography", "Memoir"}),
})
# Genres close enough that an item labelled with a neighbor plausibly satisfies
# the other; an NVR must not hinge on that distinction.
NEIGHBOR_GENRES = {
    "History": {"Nonfiction"},
    "Nonfiction": {"History", "Memoir", "Biography", "Self-Help"},
    "Memoir": {"Biography", "Nonfiction"},
    "Biography": {"Memoir", "Nonfiction"},
    "Self-Help": {"Nonfiction"},
    "Crime": {"Mystery", "Thriller"},
    "Mystery": {"Crime", "Thriller"},
    "Thriller": {"Crime", "Mystery"},
    "Children": {"Young Adult"},
    "Young Adult": {"Children"},
}
GENRE_KEYWORDS = {
    "Adventure": ("adventure",),
    "Biography": ("biography",),
    "Children": ("children",),
    "Crime": ("crime",),
    "Fantasy": ("fantasy",),
    "Graphic Novel": ("graphic novel", "comic"),
    "Historical Fiction": ("historical novel", "historical fiction"),
    "History": ("history",),
    "Horror": ("horror",),
    "Humor": ("humor", "humour", "comic", "comedy", "satir"),
    "Memoir": ("memoir", "autobiograph"),
    "Mystery": ("mystery", "detective"),
    "Nonfiction": ("non-fiction", "nonfiction"),
    "Poetry": ("poem", "poetry", "verse"),
    "Romance": ("romance", "love story"),
    "Science Fiction": ("science fiction", "sci-fi"),
    "Self-Help": ("self-help",),
    "Thriller": ("thriller",),
    "Young Adult": ("young adult",),
}
# Source country labels that are not a single modern state; never task values.
NON_MODERN_STATES = frozenset({
    "Soviet Union", "Ottoman Empire", "Czechoslovakia", "Yugoslavia",
    "Socialist Federal Republic of Yugoslavia", "Democratic Federal Yugoslavia",
    "Kingdom of Serbs, Croats and Slovenes", "Austria–Hungary", "British Raj",
    "Holy Roman Empire", "Mughal Empire", "Dutch East Indies", "Republic of China",
})
HIDDEN_MIN_SELECTIVITY = 0.5
WIKIDATA_CATEGORICAL_MIN_COUNT = 20
WIKIDATA_MIN_PREDICATE_GROWTH = 0.05
WIKIDATA_TASK_FORMATS = ("audiobook", "ebook", "hardcover", "paperback")
DEFAULT_TASK_SEED = 20260926
FORMAT_POLICY_PER_CELL = 12
MATCHED_FAMILIES_PER_COMPLEXITY = 50
MATCHED_NVR_FAMILIES_PER_COMPLEXITY = 5
MATCHED_HISTORY_FAMILIES_PER_COMPLEXITY = 16
MATCHED_DESIGN_VERSION = "matched-reveal-v1"
MATCHED_REVEAL_ASSIGNMENT_VERSION = "blocked-hash-v1"
# A hidden arm must not withhold a fact that a revealed fact implies, such as
# a hidden United Kingdom behind a revealed British English.
REVEAL_GUARD_VERSION = "implied-fact-guard-v1"
IMPLIED_FACT_FIELDS = frozenset({"genres", "languages", "countries"})
IMPLIED_FACT_MIN_PROBABILITY = 0.8
IMPLIED_FACT_MIN_LIFT = 1.5
IMPLIED_FACT_MIN_SUPPORT = 5
MAX_FAMILY_SOLUTION_JACCARD = 0.95

# Candidates are derived only from catalog predicates; retrieval is measured
# later as an outcome.
RETRIEVER_INDEPENDENT_SAMPLING = "catalog-predicate-v1"
SAMPLING_MODES = (RETRIEVER_INDEPENDENT_SAMPLING,)

COMPLEXITIES = ("simple", "medium", "complex")
REVEALS = ("volunteer", "mixed", "hidden")
SOLUTION_BANDS = {
    "simple": (100, 500),
    "medium": (20, 100),
    "complex": (3, 20),
}

PERSONAS = (
    "You are an avid reader choosing your next book and care more about fit than hype.",
    "You are browsing for a book for an upcoming quiet weekend and want one confident recommendation.",
    "You are a regular library user looking for something specific enough to narrow a large catalog.",
    "You are returning to reading after a busy stretch and want a book that matches your requirements closely.",
    "You belong to a small book club and are selecting the next title to read before the group meets.",
    "You are planning a trip and want one book that suits both your taste and practical constraints.",
    "You read widely but are particular about this choice and will answer useful clarifying questions.",
    "You are looking for a gift for another enthusiastic reader and have a concrete set of requirements.",
    "You are exploring a part of the catalog you do not know well and want the assistant to search carefully.",
    "You have a long reading list and want help finding the single best match for your current mood.",
    "You are choosing a book for a personal reading challenge and need every stated condition to hold.",
    "You are an occasional reader who wants a straightforward recommendation without a long list of options.",
)

# A history-policy task must make the policy's antecedent visible to the
# agent: the user explicitly wants an unread book. Gift tasks are excluded
# because the buyer's reading history says nothing about the recipient.
HISTORY_PERSONAS = tuple(persona for persona in PERSONAS if "gift" not in persona.lower())
HISTORY_REQUIREMENT = "You want a book you have not already read."

# The templates use only well-covered factual fields. Format availability remains available to the
# policy layer as explicitly synthetic environment state, never as a task
# constraint. Each template expands to the validator's 2 / 3-4 / 5-6 bands.
WIKIDATA_TEMPLATES = {
    "simple": (
        ("G1", "Y"), ("G1", "L"), ("G1", "C"), ("G2",),
    ),
    "medium": (
        ("G2", "Y"), ("G1", "L", "Y"), ("G1", "C", "Y"),
        ("G1", "L", "C"), ("G1", "Y-", "Y+"),
        ("G2", "L", "Y"), ("G2", "C", "Y"),
    ),
    "complex": (
        ("G2", "L", "Y-", "Y+"),
        ("G2", "C", "Y-", "Y+"), ("G1", "L", "C", "Y-", "Y+"),
    ),
}
WIKIDATA_YEAR_GRID = (
    tuple(range(1450, 1900, 25))
    + tuple(range(1900, 1980, 10))
    + tuple(range(1980, 2027))
)

SLOT_KINDS = {
    "Y": ("year_lower", "year_upper"),
    "Y-": ("year_lower",),
    "Y+": ("year_upper",),
    "L": ("language",),
    "C": ("country",),
}


def has_overlapping_genre_pair(predicates) -> bool:
    genres = [str(p.value) for p in predicates if p.field == "genres" and p.op == "contains"]
    return any(frozenset(pair) in OVERLAPPING_GENRE_PAIRS for pair in itertools.combinations(genres, 2))


def _solution_jaccard_masks(left: int, right: int) -> float:
    union = left | right
    return (left & right).bit_count() / union.bit_count() if union else 0.0


@dataclass(frozen=True)
class Predicate:
    field: str
    op: str
    value: str | int | float
    kind: str
    mask: int

    @property
    def key(self) -> str:
        return f"{self.field}:{self.op}:{self.value}"

    def as_constraint(self) -> Constraint:
        return Constraint(field=self.field, op=self.op, value=self.value)


@dataclass(frozen=True)
class Candidate:
    predicates: tuple[Predicate, ...]
    solution_mask: int
    query: str
    retrieval_rank: int | None
    anchor_id: str | None
    popularity_quartile: int
    minimal_unsat_counts: tuple[int, ...] = ()

    @property
    def signature(self) -> str:
        return "|".join(predicate.key for predicate in self.predicates)


@dataclass
class TaskDraft:
    complexity: str
    reveal: str
    candidate: Candidate
    is_nvr: bool
    family_id: str | None = None
    task: Task | None = None
    solution_ids: list[str] | None = None
    policy_eligible_ids: list[str] | None = None
    query_results: list[str] | None = None


class BookTaskGenerator:
    def __init__(
        self,
        catalog: Catalog,
        *,
        seed: int = DEFAULT_TASK_SEED,
        core_size: int = 10_000,
        sampling_mode: str = RETRIEVER_INDEPENDENT_SAMPLING,
        task_profile: str = WIKIDATA_V2_TASK_PROFILE,
    ) -> None:
        if sampling_mode not in SAMPLING_MODES:
            raise ValueError(f"unknown books task sampling mode {sampling_mode!r}")
        if task_profile not in TASK_PROFILES:
            raise ValueError(f"unknown books task profile {task_profile!r}")
        self.catalog = catalog
        self.items = [item.model_dump() for item in catalog.items]
        self.seed = seed
        self.rng = random.Random(seed)
        self.sampling_mode = sampling_mode
        self.task_profile = task_profile
        self.templates = WIKIDATA_TEMPLATES
        self.generator_version = WIKIDATA_V2_GENERATOR_VERSION
        self.constraint_fields = ["genres", "languages", "countries", "publication_year"]
        self.core_size = min(core_size, len(self.items))
        self.core_mask = (1 << self.core_size) - 1
        self.id_to_index = {item["id"]: index for index, item in enumerate(self.items)}
        self.all_mask = (1 << len(self.items)) - 1
        self.predicates = self._build_predicates()
        self.by_kind: dict[str, list[Predicate]] = defaultdict(list)
        for predicate in self.predicates:
            self.by_kind[predicate.kind].append(predicate)
        # No retrieval index is built during task construction; one is built
        # lazily only for optional post-hoc diagnostics.
        self.search = None
        self._query_cache: dict[tuple[str, ...], tuple[str, list[str], int]] = {}
        self.popularity = self._popularity_quartiles()
        self.anchor_pools = self._build_catalog_anchor_pools()

    def _mask(self, field: str, op: str, value: str | int | float) -> int:
        constraint = Constraint(field=field, op=op, value=value)
        mask = 0
        for index, item in enumerate(self.catalog.items):
            if constraint.evaluate(item):
                mask |= 1 << index
        return mask

    def _predicate(self, field: str, op: str, value: str | int | float, kind: str) -> Predicate | None:
        mask = self._mask(field, op, value)
        count = mask.bit_count()
        if count < 3 or count == len(self.items):
            return None
        return Predicate(field, op, value, kind, mask)

    def _build_predicates(self) -> list[Predicate]:
        specs: list[tuple[str, str, str | int | float, str]] = []
        specs.extend(("genres", "contains", genre, "genre") for genre in PUBLIC_BOOK_GENRES)
        specs.extend(("formats", "contains", fmt, "format") for fmt in WIKIDATA_TASK_FORMATS)
        language_counts = Counter(
            value for item in self.items for value in item.get("languages", [])
        )
        country_counts = Counter(
            value for item in self.items for value in item.get("countries", [])
        )
        specs.extend(
            ("languages", "contains", value, "language")
            for value, count in sorted(language_counts.items())
            if count >= WIKIDATA_CATEGORICAL_MIN_COUNT
            and value not in LANGUAGE_PARENTS_V2
        )
        specs.extend(
            ("countries", "contains", value, "country")
            for value, count in sorted(country_counts.items())
            if count >= WIKIDATA_CATEGORICAL_MIN_COUNT
            and value not in COUNTRY_PARENTS_V2 and value not in NON_MODERN_STATES
        )
        year_grid = WIKIDATA_YEAR_GRID
        specs.extend(("publication_year", ">=", value, "year_lower") for value in year_grid)
        specs.extend(("publication_year", "<=", value, "year_upper") for value in year_grid)
        return [predicate for spec in specs if (predicate := self._predicate(*spec)) is not None]

    def _popularity_quartiles(self) -> dict[str, int]:
        popularity_field = "sitelink_count"
        ordered = sorted(
            self.items,
            key=lambda item: (item.get(popularity_field) or 0, item["id"]),
        )
        total = len(ordered)
        return {item["id"]: min(4, index * 4 // total + 1) for index, item in enumerate(ordered)}

    def _every_predicate_is_material(
        self, predicates: tuple[Predicate, ...], solution_mask: int,
    ) -> bool:
        """Require every public-task predicate to narrow the answer set.

        A five-field conjunction is not meaningfully complex when, for
        example, removing "English" leaves exactly the same answers. The
        five-percent margin is intentionally modest but rules out both exact
        redundancy and nearly inert bounds.
        """
        solution_count = solution_mask.bit_count()
        minimum_growth = max(
            1, math.ceil(solution_count * WIKIDATA_MIN_PREDICATE_GROWTH)
        )
        for omitted in range(len(predicates)):
            leave_one_mask = self.all_mask
            for index, predicate in enumerate(predicates):
                if index != omitted:
                    leave_one_mask &= predicate.mask
            if leave_one_mask.bit_count() < solution_count + minimum_growth:
                return False
        return True

    def _search_genres(self, genres: tuple[str, ...]) -> tuple[str, list[str], int]:
        if self.search is None:
            # Retrieval diagnostics are permitted only after candidate
            # selection.  Laziness keeps construction/assembly independent.
            self.search = CatalogSearch(self.catalog, BOOKS, profile=BM25F_PROFILE)
        if genres not in self._query_cache:
            query = " ".join(genres)
            ids = [row["id"] for row in self.search.search(query, top_k=20)]
            mask = 0
            for item_id in ids:
                mask |= 1 << self.id_to_index[item_id]
            self._query_cache[genres] = (query, ids, mask)
        return self._query_cache[genres]

    def _build_catalog_anchor_pools(self) -> dict[int, list[tuple[tuple[str, ...], str]]]:
        """Build witness pools without consulting a retrieval implementation.

        Witnesses come from the smallest nested catalog prefix so every
        satisfiable family has at least one valid item at every evaluated
        catalog size.  The witness is only a construction aid; all answers are
        still recomputed from the complete catalog predicates.
        """
        allowed = {str(predicate.value) for predicate in self.by_kind["genre"]}
        pools: dict[int, list[tuple[tuple[str, ...], str]]] = {1: [], 2: []}
        for item in self.items[:self.core_size]:
            genres = tuple(sorted(set(item["genres"]) & allowed))
            for width in (1, 2):
                pools[width].extend(
                    (combination, item["id"])
                    for combination in itertools.combinations(genres, width)
                    if not (width == 2 and frozenset(combination) in OVERLAPPING_GENRE_PAIRS)
                )
        if not pools[1] or not pools[2]:
            raise RuntimeError(
                "catalog produced no core-prefix witnesses for one- or two-genre tasks"
            )
        return pools

    @staticmethod
    def _genre_width(template: tuple[str, ...]) -> int:
        return 2 if "G2" in template else 1

    def _satisfied(self, kind: str, index: int) -> list[Predicate]:
        bit = 1 << index
        return [predicate for predicate in self.by_kind[kind] if predicate.mask & bit]

    def _instantiate_for_anchor(
        self, template: tuple[str, ...], genres: tuple[str, ...], anchor_id: str,
    ) -> tuple[Predicate, ...] | None:
        anchor_index = self.id_to_index[anchor_id]
        genre_lookup = {str(p.value): p for p in self.by_kind["genre"]}
        chosen = [genre_lookup[genre] for genre in genres]
        for slot in template:
            if slot in ("G1", "G2"):
                continue
            kinds = SLOT_KINDS[slot]
            options = [p for kind in kinds for p in self._satisfied(kind, anchor_index)]
            if not options:
                return None
            chosen.append(self.rng.choice(options))
        if len({predicate.key for predicate in chosen}) != len(chosen):
            return None
        lower_years = [int(p.value) for p in chosen if p.kind == "year_lower"]
        upper_years = [int(p.value) for p in chosen if p.kind == "year_upper"]
        if lower_years and upper_years and max(lower_years) > min(upper_years):
            return None
        return tuple(sorted(chosen, key=_predicate_order))

    def generate_satisfiable(self, complexity: str, *, target_per_quartile: int) -> list[Candidate]:
        lower, upper = SOLUTION_BANDS[complexity]
        found: dict[str, Candidate] = {}
        by_quartile: Counter[int] = Counter()
        attempts = 0
        templates = self.templates[complexity]
        while min((by_quartile[q] for q in range(1, 5)), default=0) < target_per_quartile and attempts < 2_000_000:
            attempts += 1
            template = self.rng.choice(templates)
            width = self._genre_width(template)
            genres, anchor_id = self.rng.choice(self.anchor_pools[width])
            predicates = self._instantiate_for_anchor(template, genres, anchor_id)
            if predicates is None:
                continue
            signature = "|".join(predicate.key for predicate in predicates)
            if signature in found:
                # The same conjunction can have reachable top-20 anchors in
                # several popularity quartiles. Keep one task candidate, but
                # deterministically reassign its audit anchor when that fills
                # an under-target quartile from an over-target one.
                existing = found[signature]
                new_quartile = self.popularity[anchor_id]
                if (new_quartile != existing.popularity_quartile
                        and by_quartile[new_quartile] < target_per_quartile
                        and by_quartile[existing.popularity_quartile] > target_per_quartile):
                    query, rank = " ".join(genres), None
                    if existing.solution_mask & (1 << self.id_to_index[anchor_id]):
                        by_quartile[existing.popularity_quartile] -= 1
                        by_quartile[new_quartile] += 1
                        found[signature] = Candidate(
                            predicates=predicates, solution_mask=existing.solution_mask,
                            query=query, retrieval_rank=rank, anchor_id=anchor_id,
                            popularity_quartile=new_quartile,
                        )
                continue
            mask = self.all_mask
            for predicate in predicates:
                mask &= predicate.mask
            count = mask.bit_count()
            if not lower <= count <= upper:
                continue
            if not self._every_predicate_is_material(predicates, mask):
                continue
            # The item used to instantiate the predicates is necessarily a
            # solution.  Crucially, no BM25 result is consulted here.
            if not mask & (1 << self.id_to_index[anchor_id]):
                raise AssertionError("catalog witness does not satisfy its predicates")
            query, rank = " ".join(genres), None
            quartile = self.popularity[anchor_id]
            found[signature] = Candidate(
                predicates=predicates, solution_mask=mask, query=query,
                retrieval_rank=rank, anchor_id=anchor_id, popularity_quartile=quartile,
            )
            by_quartile[quartile] += 1
        if min((by_quartile[q] for q in range(1, 5)), default=0) < target_per_quartile:
            raise RuntimeError(
                f"only generated satisfiable {complexity} candidates by quartile {dict(by_quartile)} "
                f"after {attempts:,} attempts"
            )
        return list(found.values())

    def _instantiate_unanchored(self, template: tuple[str, ...]) -> tuple[Predicate, ...] | None:
        chosen: list[Predicate] = []
        width = self._genre_width(template)
        chosen.extend(self.rng.sample(self.by_kind["genre"], width))
        for slot in template:
            if slot in ("G1", "G2"):
                continue
            kinds = SLOT_KINDS[slot]
            options = [predicate for kind in kinds for predicate in self.by_kind[kind]]
            chosen.append(self.rng.choice(options))
        if len({predicate.key for predicate in chosen}) != len(chosen):
            return None
        lower_years = [int(p.value) for p in chosen if p.kind == "year_lower"]
        upper_years = [int(p.value) for p in chosen if p.kind == "year_upper"]
        if lower_years and upper_years and max(lower_years) > min(upper_years):
            return None
        return tuple(sorted(chosen, key=_predicate_order))

    @staticmethod
    def _expanded_template_kinds(template: tuple[str, ...]) -> list[tuple[str, ...]]:
        """Expand G2 into two predicate positions for constructive search."""
        positions: list[tuple[str, ...]] = []
        for slot in template:
            if slot == "G1":
                positions.append(("genre",))
            elif slot == "G2":
                positions.extend((("genre",), ("genre",)))
            else:
                positions.append(SLOT_KINDS[slot])
        return positions

    def _generate_wikidata_complex_nvr(self, target: int) -> list[Candidate]:
        """Construct strict minimal-unsat conjunctions for the sparse public profile.

        Blindly sampling all five or six predicates almost never lands exactly
        on the minimal-unsat boundary. Instead, sample all but one predicate,
        require that base to retain at least three matches, then search the
        held-out predicate family for a disjoint value and verify every
        leave-one-out conjunction independently.
        """
        found: dict[str, Candidate] = {}
        attempts = 0
        while len(found) < target and attempts < 1_000_000:
            attempts += 1
            template = self.rng.choice(self.templates["complex"])
            positions = self._expanded_template_kinds(template)
            held_out = self.rng.randrange(len(positions))
            chosen: list[Predicate | None] = []
            for index, kinds in enumerate(positions):
                if index == held_out:
                    chosen.append(None)
                    continue
                options = [predicate for kind in kinds for predicate in self.by_kind[kind]]
                predicate = self.rng.choice(options)
                if any(
                    existing is not None and existing.key == predicate.key
                    for existing in chosen
                ):
                    break
                chosen.append(predicate)
            if len(chosen) != len(positions):
                continue

            base_mask = self.all_mask
            for predicate in chosen:
                if predicate is not None:
                    base_mask &= predicate.mask
            if base_mask.bit_count() < 3:
                continue

            held_out_options = [
                predicate
                for kind in positions[held_out]
                for predicate in self.by_kind[kind]
            ]
            self.rng.shuffle(held_out_options)
            for held_out_predicate in held_out_options:
                if any(
                    existing is not None and existing.key == held_out_predicate.key
                    for existing in chosen
                ):
                    continue
                if base_mask & held_out_predicate.mask:
                    continue
                materialized = [
                    held_out_predicate if index == held_out else predicate
                    for index, predicate in enumerate(chosen)
                ]
                if any(predicate is None for predicate in materialized):
                    raise AssertionError("held-out predicate was not materialized")
                concrete = tuple(sorted(
                    (predicate for predicate in materialized if predicate is not None),
                    key=_predicate_order,
                ))
                signature = "|".join(predicate.key for predicate in concrete)
                if signature in found:
                    continue
                leave_one_masks: list[int] = []
                for omitted in range(len(concrete)):
                    mask = self.all_mask
                    for index, predicate in enumerate(concrete):
                        if index != omitted:
                            mask &= predicate.mask
                    leave_one_masks.append(mask)
                counts = tuple(mask.bit_count() for mask in leave_one_masks)
                if min(counts, default=0) < 3:
                    continue
                if has_overlapping_genre_pair(concrete) or self.nvr_fragility(concrete):
                    continue
                near_mask = 0
                for mask in leave_one_masks:
                    near_mask |= mask
                witness_mask = near_mask & self.core_mask or near_mask
                witness_ids = self.ids_from_mask(witness_mask)
                if not witness_ids:
                    continue
                audit_witness_id = _stable_choice(
                    witness_ids, f"{self.seed}:complex:{signature}:near"
                )
                genres = tuple(
                    str(predicate.value)
                    for predicate in concrete if predicate.kind == "genre"
                )
                found[signature] = Candidate(
                    predicates=concrete,
                    solution_mask=0,
                    query=" ".join(genres),
                    retrieval_rank=None,
                    anchor_id=None,
                    popularity_quartile=self.popularity[audit_witness_id],
                    minimal_unsat_counts=counts,
                )
                break
        if len(found) < target:
            raise RuntimeError(
                f"only generated {len(found)} constructive minimal-unsat complex "
                f"candidates after {attempts:,} attempts"
            )
        return list(found.values())

    def generate_nvr(self, complexity: str, *, target: int = 30) -> list[Candidate]:
        if complexity == "complex":
            return self._generate_wikidata_complex_nvr(target)
        found: dict[str, Candidate] = {}
        attempts = 0
        while len(found) < target and attempts < 1_000_000:
            predicates = self._instantiate_unanchored(
                self.rng.choice(self.templates[complexity])
            )
            attempts += 1
            if predicates is None:
                continue
            signature = "|".join(predicate.key for predicate in predicates)
            if signature in found:
                continue
            full = self.all_mask
            for predicate in predicates:
                full &= predicate.mask
            if full:
                continue
            leave_one_masks = []
            for omitted in range(len(predicates)):
                mask = self.all_mask
                for index, predicate in enumerate(predicates):
                    if index != omitted:
                        mask &= predicate.mask
                leave_one_masks.append(mask)
            counts = tuple(mask.bit_count() for mask in leave_one_masks)
            if min(counts, default=0) < 3:
                continue
            if has_overlapping_genre_pair(predicates) or self.nvr_fragility(predicates):
                continue
            genres = tuple(str(p.value) for p in predicates if p.kind == "genre")
            near_mask = 0
            for mask in leave_one_masks:
                near_mask |= mask
            # Minimal-unsat construction is also retrieval-independent.
            # Prefer a witness in the smallest nested prefix, but never
            # require a search result to expose it.
            witness_mask = near_mask & self.core_mask or near_mask
            witness_ids = self.ids_from_mask(witness_mask)
            if not witness_ids:
                continue
            audit_witness_id = _stable_choice(
                witness_ids, f"{self.seed}:{complexity}:{signature}:near"
            )
            query = " ".join(genres)
            found[signature] = Candidate(
                predicates=predicates, solution_mask=0, query=query,
                retrieval_rank=None, anchor_id=None,
                popularity_quartile=self.popularity[audit_witness_id],
                minimal_unsat_counts=counts,
            )
        if len(found) < target:
            raise RuntimeError(
                f"only generated {len(found)} minimal-unsat {complexity} candidates after {attempts:,} attempts"
            )
        return list(found.values())

    def ids_from_mask(self, mask: int) -> list[str]:
        ids: list[str] = []
        while mask:
            least = mask & -mask
            index = least.bit_length() - 1
            ids.append(self.items[index]["id"])
            mask ^= least
        return ids

    def _ranked(self, candidates: Iterable[Candidate], label: str) -> list[Candidate]:
        return sorted(
            candidates,
            key=lambda candidate: hashlib.sha256(
                f"{self.generator_version}:{self.seed}:{label}:{candidate.signature}".encode()
            ).hexdigest(),
        )

    def _select_cell(
        self, candidates: list[Candidate], count: int, *, label: str,
        used: set[str], rotate: int, genre_usage: Counter[str],
    ) -> list[Candidate]:
        ranked = self._ranked((candidate for candidate in candidates if candidate.signature not in used), label)
        by_quartile = {q: [candidate for candidate in ranked if candidate.popularity_quartile == q] for q in range(1, 5)}
        base, remainder = divmod(count, 4)
        quotas = {q: base for q in range(1, 5)}
        for offset in range(remainder):
            quotas[(rotate + offset) % 4 + 1] += 1
        selected: list[Candidate] = []
        for quartile, quota in quotas.items():
            if len(by_quartile[quartile]) < quota:
                raise RuntimeError(f"{label}: need {quota} candidates in Q{quartile}, have {len(by_quartile[quartile])}")
            available = list(by_quartile[quartile])
            for _ in range(quota):
                best = min(available, key=lambda candidate: _genre_balance_key(candidate, genre_usage))
                available.remove(best)
                selected.append(best)
                genre_usage.update(_candidate_genres(best))
        selected = self._ranked(selected, label + ":final")
        used.update(candidate.signature for candidate in selected)
        return selected

    def _repair_solution_diversity(
        self,
        selected: list[Candidate],
        candidate_pool: list[Candidate],
        *,
        label: str,
        prior_solution_masks: list[int],
        genre_usage: Counter[str],
    ) -> tuple[list[Candidate], list[dict]]:
        """Replace only selected families that breach the frozen overlap cap.

        Repair happens after the original seeded selection and preserves each
        family's slot.  This keeps an objective diversity fix from needlessly
        renumbering every later family.
        """
        repaired = list(selected)
        repairs: list[dict] = []
        for index, candidate in enumerate(repaired):
            comparison_masks = [
                *prior_solution_masks,
                *(
                    other.solution_mask
                    for other_index, other in enumerate(repaired)
                    if other_index != index
                ),
            ]
            maximum = max(
                (
                    _solution_jaccard_masks(candidate.solution_mask, mask)
                    for mask in comparison_masks
                ),
                default=0.0,
            )
            if maximum < MAX_FAMILY_SOLUTION_JACCARD:
                continue

            used_signatures = {current.signature for current in repaired}
            eligible = [
                replacement
                for replacement in candidate_pool
                if replacement.signature not in used_signatures
                and replacement.popularity_quartile == candidate.popularity_quartile
                and all(
                    _solution_jaccard_masks(replacement.solution_mask, mask)
                    < MAX_FAMILY_SOLUTION_JACCARD
                    for mask in comparison_masks
                )
            ]
            if not eligible:
                raise RuntimeError(
                    f"{label}: no Q{candidate.popularity_quartile} replacement "
                    f"keeps solution-set Jaccard below {MAX_FAMILY_SOLUTION_JACCARD}"
                )
            for genre in _candidate_genres(candidate):
                genre_usage[genre] -= 1
            replacement = min(
                eligible,
                key=lambda current: _genre_balance_key(current, genre_usage),
            )
            genre_usage.update(_candidate_genres(replacement))
            repaired[index] = replacement
            repairs.append({
                "slot": index,
                "replaced_signature": candidate.signature,
                "replacement_signature": replacement.signature,
                "replaced_maximum_jaccard": maximum,
                "popularity_quartile": candidate.popularity_quartile,
            })
        return repaired, repairs

    def _select_unstratified(
        self, candidates: list[Candidate], count: int, *, label: str, used: set[str],
        genre_usage: Counter[str],
    ) -> list[Candidate]:
        ranked = self._ranked((candidate for candidate in candidates if candidate.signature not in used), label)
        if len(ranked) < count:
            raise RuntimeError(f"{label}: need {count} candidates, have {len(ranked)}")
        selected: list[Candidate] = []
        available = list(ranked)
        for _ in range(count):
            best = min(available, key=lambda candidate: _genre_balance_key(candidate, genre_usage))
            available.remove(best)
            selected.append(best)
            genre_usage.update(_candidate_genres(best))
        used.update(candidate.signature for candidate in selected)
        return selected

    def assemble_matched(
        self,
        *,
        families_per_complexity: int = MATCHED_FAMILIES_PER_COMPLEXITY,
        nvr_families_per_complexity: int = MATCHED_NVR_FAMILIES_PER_COMPLEXITY,
        candidate_target_per_quartile: int = 36,
    ) -> tuple[list[TaskDraft], dict]:
        """Select one task family, then cross it with every reveal condition.

        This is the controlled matched design.  Constraints, answers, policies,
        persona, and history are held fixed within each family; only reveal
        tags differ.
        """
        if not 0 <= nvr_families_per_complexity < families_per_complexity:
            raise ValueError("NVR family count must be non-negative and smaller than all families")

        satisfiable_count = families_per_complexity - nvr_families_per_complexity
        sat = {
            complexity: self.generate_satisfiable(
                complexity, target_per_quartile=candidate_target_per_quartile,
            )
            for complexity in COMPLEXITIES
        }
        # The hidden arm must withhold something that matters.
        sat = {
            complexity: [c for c in candidates if self.selective_hidden_possible(c.predicates)]
            for complexity, candidates in sat.items()
        }
        nvr_target = max(nvr_families_per_complexity * 3, nvr_families_per_complexity)
        nvr = {
            complexity: self.generate_nvr(complexity, target=nvr_target)
            for complexity in COMPLEXITIES
        }

        drafts: list[TaskDraft] = []
        genre_usage: Counter[str] = Counter()
        family_rows: list[dict] = []
        selected_by_complexity: dict[str, list[tuple[Candidate, bool]]] = {}
        for complexity_index, complexity in enumerate(COMPLEXITIES):
            selected_sat = self._select_cell(
                sat[complexity], satisfiable_count,
                label=f"{MATCHED_DESIGN_VERSION}:{complexity}:sat",
                used=set(), rotate=complexity_index, genre_usage=genre_usage,
            )
            selected_nvr = self._select_unstratified(
                nvr[complexity], nvr_families_per_complexity,
                label=f"{MATCHED_DESIGN_VERSION}:{complexity}:nvr",
                used=set(), genre_usage=genre_usage,
            )
            selected = [
                *((candidate, False) for candidate in selected_sat),
                *((candidate, True) for candidate in selected_nvr),
            ]
            selected.sort(key=lambda row: hashlib.sha256(
                f"{MATCHED_DESIGN_VERSION}:{self.seed}:{complexity}:{row[0].signature}".encode()
            ).hexdigest())
            selected_by_complexity[complexity] = selected

        # Lock the original seeded selection across all three complexities
        # before repairing overlap.  Otherwise a simple-family replacement
        # changes genre counts and cascades into unrelated medium/complex
        # choices, so one repair cannot cascade into unrelated choices.
        selected_solution_masks: list[int] = []
        diversity_repairs: list[dict] = []
        for complexity in COMPLEXITIES:
            selected = selected_by_complexity[complexity]
            satisfiable_slots = [
                index for index, (_, is_nvr) in enumerate(selected) if not is_nvr
            ]
            repaired_sat, repairs = self._repair_solution_diversity(
                [selected[index][0] for index in satisfiable_slots],
                sat[complexity],
                label=f"{MATCHED_DESIGN_VERSION}:{complexity}:diversity",
                prior_solution_masks=selected_solution_masks,
                genre_usage=genre_usage,
            )
            for slot, candidate in zip(satisfiable_slots, repaired_sat):
                selected[slot] = (candidate, False)
            selected_solution_masks.extend(
                candidate.solution_mask for candidate in repaired_sat
            )
            diversity_repairs.extend({"complexity": complexity, **row} for row in repairs)
            for family_number, (candidate, is_nvr) in enumerate(selected, 1):
                family_id = f"books_{complexity}_{family_number:03d}"
                for reveal in REVEALS:
                    drafts.append(TaskDraft(
                        complexity=complexity,
                        reveal=reveal,
                        candidate=candidate,
                        is_nvr=is_nvr,
                        family_id=family_id,
                    ))
                family_rows.append({
                    "family_id": family_id,
                    "complexity": complexity,
                    "no_valid_recommendation": is_nvr,
                    "constraint_signature": candidate.signature,
                })

        return drafts, {
            "design_version": MATCHED_DESIGN_VERSION,
            "reveal_assignment_version": MATCHED_REVEAL_ASSIGNMENT_VERSION,
            "sampling_mode": self.sampling_mode,
            "families_per_complexity": families_per_complexity,
            "nvr_families_per_complexity": nvr_families_per_complexity,
            "reveal_variants": list(REVEALS),
            "solution_set_diversity": {
                "maximum_pairwise_jaccard_exclusive": MAX_FAMILY_SOLUTION_JACCARD,
                "observed_maximum_pairwise_jaccard": max(
                    (
                        _solution_jaccard_masks(left, right)
                        for index, left in enumerate(selected_solution_masks)
                        for right in selected_solution_masks[index + 1:]
                    ),
                    default=0.0,
                ),
                "repairs": diversity_repairs,
            },
            "candidate_pool": {
                complexity: {
                    "satisfiable": len(sat[complexity]),
                    "minimal_unsat": len(nvr[complexity]),
                }
                for complexity in COMPLEXITIES
            },
            "families": family_rows,
        }

    def materialize_matched(
        self,
        drafts: list[TaskDraft],
        *,
        format_families_per_complexity: int = FORMAT_POLICY_PER_CELL,
        history_families_per_complexity: int = MATCHED_HISTORY_FAMILIES_PER_COMPLEXITY,
        include_retrieval_diagnostics: bool = False,
    ) -> list[TaskDraft]:
        """Materialize matched variants without changing task selection.

        Optional retrieval diagnostics run only after all families and task
        state are frozen.  Their outputs can therefore never determine which
        examples enter the benchmark.
        """
        groups = _matched_groups(drafts)
        representatives = {family_id: variants[0] for family_id, variants in groups.items()}
        # Hidden facts are fixed per family before policy state, because the
        # reading-history decoys must cover exactly the hidden predicates.
        hidden_by_family: dict[str, list[int]] = {}
        self.hidden_selectivity_by_family: dict[str, float] = {}
        for family_id, draft in representatives.items():
            hidden, selectivity = self.v2_hidden_indices(family_id, draft.candidate.predicates)
            hidden_by_family[family_id] = hidden
            self.hidden_selectivity_by_family[family_id] = round(selectivity, 4)

        format_families: set[str] = set()
        format_by_family: dict[str, str] = {}
        history_families: set[str] = set()
        for complexity in COMPLEXITIES:
            eligible_format = [
                draft for draft in representatives.values()
                if draft.complexity == complexity
                and not draft.is_nvr
                and not any(predicate.kind == "format" for predicate in draft.candidate.predicates)
                and self._format_options(draft.candidate.solution_mask)
            ]
            base, remainder = divmod(
                format_families_per_complexity, len(WIKIDATA_TASK_FORMATS)
            )
            quotas = {
                value: base + int(index < remainder)
                for index, value in enumerate(WIKIDATA_TASK_FORMATS)
            }
            available_counts = {
                value: sum(
                    any(option == value for option, _ in self._format_options(
                        draft.candidate.solution_mask
                    ))
                    for draft in eligible_format
                )
                for value in WIKIDATA_TASK_FORMATS
            }
            chosen_format: set[str] = set()
            for value in sorted(
                WIKIDATA_TASK_FORMATS,
                key=lambda current: (available_counts[current], current),
            ):
                eligible_for_value = [
                    draft.candidate for draft in eligible_format
                    if draft.candidate.signature not in chosen_format
                    and any(
                        option == value
                        for option, _ in self._format_options(
                            draft.candidate.solution_mask
                        )
                    )
                ]
                ranked_for_value = self._ranked(
                    eligible_for_value,
                    f"{MATCHED_DESIGN_VERSION}:{complexity}:format:{value}",
                )
                selected_for_value = ranked_for_value[:quotas[value]]
                if len(selected_for_value) != quotas[value]:
                    raise RuntimeError(
                        f"{complexity}: only {len(selected_for_value)} unused families "
                        f"support {value}; need {quotas[value]}"
                    )
                for candidate in selected_for_value:
                    chosen_format.add(candidate.signature)
                    family_id = next(
                        current_family_id
                        for current_family_id, draft in representatives.items()
                        if draft.candidate.signature == candidate.signature
                    )
                    format_by_family[family_id] = value
            if len(chosen_format) != format_families_per_complexity:
                raise RuntimeError(
                    f"{complexity}: only {len(chosen_format)} matched families have a testable "
                    f"format policy; need {format_families_per_complexity}"
                )
            format_families.update(
                family_id for family_id, draft in representatives.items()
                if draft.complexity == complexity and draft.candidate.signature in chosen_format
            )

            eligible_history = [
                draft for family_id, draft in representatives.items()
                if draft.complexity == complexity
                and not draft.is_nvr
                and (
                    self._policy_mask_before_history(
                        draft, has_format=family_id in format_families,
                        selected_format=format_by_family.get(family_id),
                    ) & self.core_mask
                ).bit_count() >= 2
                and self._history_decoys_exist(
                    family_id, draft, hidden_by_family[family_id],
                    has_format=family_id in format_families,
                    selected_format=format_by_family.get(family_id),
                )
            ]
            ranked_history = self._ranked(
                (draft.candidate for draft in eligible_history),
                f"{MATCHED_DESIGN_VERSION}:{complexity}:history",
            )
            chosen_history = {
                candidate.signature for candidate in ranked_history[:history_families_per_complexity]
            }
            if len(chosen_history) != history_families_per_complexity:
                raise RuntimeError(
                    f"{complexity}: only {len(chosen_history)} matched families can carry a "
                    f"history trap; need {history_families_per_complexity}"
                )
            history_families.update(
                family_id for family_id, draft in representatives.items()
                if draft.complexity == complexity and draft.candidate.signature in chosen_history
            )

        family_order = {
            family_id: index
            for index, family_id in enumerate(sorted(groups), 1)
        }
        reveal_code = {"volunteer": "v", "mixed": "m", "hidden": "h"}
        for family_id, variants in groups.items():
            representative = representatives[family_id]
            has_format = family_id in format_families
            has_history = family_id in history_families
            solution_ids = self.ids_from_mask(representative.candidate.solution_mask)
            eligible_mask = representative.candidate.solution_mask
            user_formats: list[str] = []
            if has_format:
                fmt, compatible_mask = self._chosen_format_option(
                    representative.candidate.solution_mask,
                    required=format_by_family.get(family_id),
                )
                user_formats = [fmt]
                eligible_mask &= compatible_mask

            read_ids: list[str] = []
            if has_history:
                read_ids = self.history_read_ids(
                    family_id, representative.candidate.predicates,
                    hidden_indices=hidden_by_family[family_id], eligible_mask=eligible_mask,
                )
                eligible_mask &= ~(1 << self.id_to_index[read_ids[0]])

            flags = []
            if has_history:
                flags.append("history")
            if has_format:
                flags.append("format_verification_v2")
            flags.append("single_recommendation")
            if representative.is_nvr:
                flags.append("transparency")
            flags.append("recommend_tool")

            persona_pool = HISTORY_PERSONAS if has_history else PERSONAS
            persona = persona_pool[
                int(hashlib.sha256(f"{self.seed}:{family_id}".encode()).hexdigest(), 16)
                % len(persona_pool)
            ]
            if has_history:
                persona = f"{persona} {HISTORY_REQUIREMENT}"

            query_results = None
            if include_retrieval_diagnostics:
                query_results = self._search_genres(_candidate_genres(
                    representative.candidate
                ))[1]

            for draft in variants:
                task_id = f"task_{family_order[family_id]:03d}_{reveal_code[draft.reveal]}"
                draft.task = Task(
                    id=task_id,
                    constraints=_tag_constraints_matched(
                        draft.candidate.predicates,
                        draft.reveal,
                        family_id=family_id,
                        seed=self.seed,
                        hidden_override=(
                            hidden_by_family[family_id]
                            if draft.reveal == "hidden" else None
                        ),
                    ),
                    persona=persona,
                    soft_preferences=[],
                    policy_flags=list(flags),
                    no_valid_recommendation=draft.is_nvr,
                    complexity=draft.complexity,
                    reveal_difficulty=draft.reveal,
                    user_history={"user_1": {"read": list(read_ids), "ratings": {}}},
                    user_id="user_1",
                    user_services=[],
                    user_formats=list(user_formats),
                )
                draft.solution_ids = list(solution_ids)
                draft.policy_eligible_ids = self.ids_from_mask(eligible_mask)
                draft.query_results = list(query_results) if query_results is not None else None

        errors = validate_matched_families(drafts)
        if errors:
            raise AssertionError("invalid matched task design: " + "; ".join(errors))
        return drafts

    def _leave_one_masks(self, predicates) -> list[int]:
        masks = []
        for omitted in range(len(predicates)):
            mask = self.all_mask
            for index, predicate in enumerate(predicates):
                if index != omitted:
                    mask &= predicate.mask
            masks.append(mask)
        return masks

    def nvr_fragility(self, predicates) -> list[str]:
        """Reasons a reader might call this NVR satisfiable (v2 rule).

        For each predicate, the near-misses satisfy every other predicate.
        Each of these makes the impossibility hinge on a fragile label:
        - a near-miss on a genre carries a neighbor genre, or its description
          names the genre;
        - a near-miss on a country has no country label at all.
        """
        reasons = []
        for predicate, near_mask in zip(predicates, self._leave_one_masks(predicates)):
            near_ids = self.ids_from_mask(near_mask & ~predicate.mask & self.all_mask)
            for item_id in near_ids:
                item = self.items[self.id_to_index[item_id]]
                if predicate.field == "genres":
                    genre = str(predicate.value)
                    if set(item.get("genres", [])) & NEIGHBOR_GENRES.get(genre, set()):
                        reasons.append(f"{item_id}: neighbor genre of {genre}")
                    text = (item.get("description") or "").casefold()
                    if any(keyword in text for keyword in GENRE_KEYWORDS.get(genre, ())):
                        reasons.append(f"{item_id}: description names {genre}")
                elif predicate.field == "countries" and not item.get("countries"):
                    reasons.append(f"{item_id}: no country label (might be {predicate.value})")
        return sorted(set(reasons))

    def hidden_selectivity(self, predicates, hidden_indices) -> float:
        """Share of works consistent with the revealed facts that hidden facts exclude."""
        hidden = set(hidden_indices)
        revealed = self.all_mask
        full = self.all_mask
        for index, predicate in enumerate(predicates):
            full &= predicate.mask
            if index not in hidden:
                revealed &= predicate.mask
        consistent = revealed.bit_count()
        return 1 - full.bit_count() / consistent if consistent else 0.0

    def _history_decoys_exist(
        self, family_id: str, draft: "TaskDraft", hidden: list[int], *, has_format: bool,
        selected_format: str | None,
    ) -> bool:
        mask = self._policy_mask_before_history(draft, has_format=has_format, selected_format=selected_format)
        try:
            self.history_read_ids(family_id, draft.candidate.predicates, hidden_indices=hidden, eligible_mask=mask)
        except ValueError:
            return False
        return True

    def selective_hidden_possible(self, predicates) -> bool:
        """Whether every possible opening fact leaves a selective, leak-free hidden subset.

        Family IDs, and with them the hash-ordered opening, are assigned after
        selection. A candidate is therefore admitted only if any opening still
        allows hidden facts that exclude at least ``HIDDEN_MIN_SELECTIVITY``.
        """
        size = len(predicates)
        hidden_count = max(1, size // 2)
        implied = implied_constraint_pairs([p.as_constraint() for p in predicates], self.catalog.items)
        for opening in range(size):
            others = [index for index in range(size) if index != opening]
            if not any(
                not any(source not in subset and target in subset for source, target in implied)
                and self.hidden_selectivity(predicates, subset) >= HIDDEN_MIN_SELECTIVITY
                for subset in itertools.combinations(others, hidden_count)
            ):
                return False
        return True

    def v2_hidden_indices(self, family_id: str, predicates) -> tuple[list[int], float]:
        """Hidden facts for the hidden arm: no implied-fact leak, selective (v2 rule).

        Subsets are tried in the frozen hash order (latest-revealed first), as in
        :func:`guarded_hidden_indices`. The first one with no leak that excludes
        at least ``HIDDEN_MIN_SELECTIVITY`` of revealed-consistent works wins.
        If none reaches that, the most selective leak-free subset is used, and
        the validator reports it.
        """
        order = matched_reveal_order(len(predicates), family_id=family_id, seed=self.seed)
        hidden_count = max(1, len(predicates) // 2)
        implied = implied_constraint_pairs([p.as_constraint() for p in predicates], self.catalog.items)
        rank = {index: position for position, index in enumerate(order)}
        subsets = sorted(
            itertools.combinations(order[1:], hidden_count),
            key=lambda subset: sorted((rank[index] for index in subset), reverse=True),
            reverse=True,
        )
        leak_free = [
            sorted(subset, key=rank.__getitem__) for subset in subsets
            if not any(source not in subset and target in subset for source, target in implied)
        ]
        if not leak_free:
            raise ValueError(f"{family_id}: no hidden assignment avoids an implied-fact leak")
        for subset in leak_free:
            selectivity = self.hidden_selectivity(predicates, subset)
            if selectivity >= HIDDEN_MIN_SELECTIVITY:
                return subset, selectivity
        best = max(leak_free, key=lambda subset: self.hidden_selectivity(predicates, subset))
        return best, self.hidden_selectivity(predicates, best)

    def history_read_ids(self, family_id: str, predicates, *, hidden_indices, eligible_mask: int) -> list[str]:
        """Trap plus decoys: the read list must not reveal any hidden value (v2 rule).

        The trap is an eligible core-prefix solution the user has read. For each
        hidden predicate, a decoy is a core-prefix work that satisfies every
        other predicate but fails that one. The read list therefore holds both
        values of every hidden fact.
        """
        eligible_ids = self.ids_from_mask(eligible_mask & self.core_mask)
        if len(eligible_ids) < 2:
            raise ValueError(f"{family_id}: fewer than two eligible core solutions for a history trap")
        trap = _stable_choice(eligible_ids, f"{self.seed}:{family_id}:trap")
        read_ids = [trap]
        for hidden_index, near_mask in zip(range(len(predicates)), self._leave_one_masks(predicates)):
            if hidden_index not in set(hidden_indices):
                continue
            decoys = self.ids_from_mask(near_mask & ~predicates[hidden_index].mask & self.core_mask)
            decoys = [item_id for item_id in decoys if item_id not in read_ids]
            if not decoys:
                raise ValueError(f"{family_id}: no core decoy fails only hidden predicate {hidden_index}")
            read_ids.append(_stable_choice(decoys, f"{self.seed}:{family_id}:decoy:{hidden_index}"))
        return read_ids

    def _format_options(self, solution_mask: int) -> list[tuple[str, int]]:
        options = []
        for predicate in self.by_kind["format"]:
            compatible = solution_mask & predicate.mask
            incompatible = solution_mask & ~predicate.mask & self.all_mask
            # The format trap must bite at the smallest catalog size too.
            incompatible &= self.core_mask
            if compatible.bit_count() >= 2 and incompatible and compatible & self.core_mask:
                options.append((str(predicate.value), predicate.mask))
        return options

    def _chosen_format_option(
        self, solution_mask: int, *, required: str | None = None,
    ) -> tuple[str, int]:
        options = self._format_options(solution_mask)
        if required is not None:
            options = [option for option in options if option[0] == required]
        if not options:
            raise AssertionError("format policy selected for a task without a core-prefix-compatible option")
        return max(
            options,
            key=lambda pair: (
                min(
                    (solution_mask & pair[1]).bit_count(),
                    (solution_mask & ~pair[1] & self.all_mask).bit_count(),
                ),
                pair[0],
            ),
        )

    def _policy_mask_before_history(
        self, draft: TaskDraft, *, has_format: bool,
        selected_format: str | None = None,
    ) -> int:
        mask = draft.candidate.solution_mask
        if has_format:
            _, compatible_mask = self._chosen_format_option(
                mask, required=selected_format,
            )
            mask &= compatible_mask
        return mask


def _matched_groups(drafts: list[TaskDraft]) -> dict[str, list[TaskDraft]]:
    groups: dict[str, list[TaskDraft]] = defaultdict(list)
    for draft in drafts:
        if draft.family_id is None:
            raise ValueError("matched task draft is missing family_id")
        groups[draft.family_id].append(draft)
    reveal_order = {reveal: index for index, reveal in enumerate(REVEALS)}
    for variants in groups.values():
        variants.sort(key=lambda draft: reveal_order.get(draft.reveal, len(REVEALS)))
    complexity_order = {
        complexity: index for index, complexity in enumerate(COMPLEXITIES)
    }
    return dict(sorted(
        groups.items(),
        key=lambda pair: (
            complexity_order.get(pair[1][0].complexity, len(COMPLEXITIES)),
            pair[0],
        ),
    ))


def validate_matched_families(drafts: list[TaskDraft]) -> list[str]:
    """Return violations of the matched reveal-family contract."""
    errors: list[str] = []
    try:
        groups = _matched_groups(drafts)
    except ValueError as exc:
        return [str(exc)]

    task_ids: list[str] = []
    for family_id, variants in groups.items():
        reveals = [draft.reveal for draft in variants]
        if len(variants) != len(REVEALS) or set(reveals) != set(REVEALS):
            errors.append(
                f"{family_id}: expected one variant for {REVEALS}, got {tuple(reveals)}"
            )
            continue
        signatures = {draft.candidate.signature for draft in variants}
        solution_masks = {draft.candidate.solution_mask for draft in variants}
        complexities = {draft.complexity for draft in variants}
        nvr_values = {draft.is_nvr for draft in variants}
        if len(signatures) != 1 or len(solution_masks) != 1:
            errors.append(f"{family_id}: variants do not share constraints and answer state")
        if len(complexities) != 1 or len(nvr_values) != 1:
            errors.append(f"{family_id}: variants do not share complexity and NVR state")

        if any(draft.task is None for draft in variants):
            errors.append(f"{family_id}: variants have not all been materialized")
            continue
        tasks = [draft.task for draft in variants]
        assert all(task is not None for task in tasks)
        concrete_tasks = [task for task in tasks if task is not None]
        task_ids.extend(task.id for task in concrete_tasks)

        expected_constraints = [
            tagged.constraint.model_dump(mode="json")
            for tagged in concrete_tasks[0].constraints
        ]
        shared_states = []
        for draft, task in zip(variants, concrete_tasks):
            constraints = [
                tagged.constraint.model_dump(mode="json") for tagged in task.constraints
            ]
            if constraints != expected_constraints:
                errors.append(f"{family_id}: {draft.reveal} changes constraint values or order")
            if task.reveal_difficulty != draft.reveal:
                errors.append(f"{family_id}: {draft.reveal} task label disagrees with its variant")
            if draft.solution_ids != variants[0].solution_ids:
                errors.append(f"{family_id}: answer IDs vary across reveal conditions")
            if draft.policy_eligible_ids != variants[0].policy_eligible_ids:
                errors.append(f"{family_id}: policy-eligible answers vary across reveal conditions")

            state = task.model_dump(mode="json")
            state.pop("id")
            state.pop("constraints")
            state.pop("reveal_difficulty")
            shared_states.append(state)
        if any(state != shared_states[0] for state in shared_states[1:]):
            errors.append(f"{family_id}: non-reveal task state varies across variants")

        reveal_vectors = {
            draft.reveal: [tagged.reveal for tagged in task.constraints]
            for draft, task in zip(variants, concrete_tasks)
        }
        volunteer = reveal_vectors["volunteer"]
        mixed = reveal_vectors["mixed"]
        hidden = reveal_vectors["hidden"]
        if any(tag != RevealTag.VOLUNTEER for tag in volunteer):
            errors.append(f"{family_id}: volunteer arm does not disclose every predicate")
        mixed_volunteer = {
            index for index, tag in enumerate(mixed) if tag == RevealTag.VOLUNTEER
        }
        hidden_volunteer = {
            index for index, tag in enumerate(hidden) if tag == RevealTag.VOLUNTEER
        }
        if len(mixed_volunteer) != 1 or any(
            tag not in {RevealTag.VOLUNTEER, RevealTag.ON_ASK} for tag in mixed
        ):
            errors.append(f"{family_id}: mixed arm must have one volunteer and only on-ask others")
        if hidden_volunteer != mixed_volunteer or RevealTag.HIDDEN not in hidden:
            errors.append(
                f"{family_id}: hidden arm must preserve the mixed volunteer and hide >=1 other"
            )

    if len(task_ids) != len(set(task_ids)):
        errors.append("task IDs are not unique across matched families")
    return errors


def matched_family_manifest(drafts: list[TaskDraft]) -> dict:
    """Create the frozen pairing map used for paired statistical analysis."""
    errors = validate_matched_families(drafts)
    if errors:
        raise ValueError("cannot manifest invalid matched families: " + "; ".join(errors))
    groups = _matched_groups(drafts)
    rows = []
    for family_id, variants in groups.items():
        representative = variants[0]
        assert representative.task is not None
        solution_ids = representative.solution_ids or []
        rows.append({
            "family_id": family_id,
            "complexity": representative.complexity,
            "no_valid_recommendation": representative.is_nvr,
            "constraint_signature": representative.candidate.signature,
            "solution_ids_sha256": hashlib.sha256(
                "\n".join(solution_ids).encode()
            ).hexdigest(),
            "task_ids": {
                draft.reveal: draft.task.id
                for draft in variants
                if draft.task is not None
            },
        })
    return {
        "design_version": MATCHED_DESIGN_VERSION,
        "reveal_assignment_version": MATCHED_REVEAL_ASSIGNMENT_VERSION,
        "sampling_mode": RETRIEVER_INDEPENDENT_SAMPLING,
        "reveal_variants": list(REVEALS),
        "family_count": len(rows),
        "task_count": len(drafts),
        "families": rows,
    }


def _predicate_order(predicate: Predicate) -> tuple:
    field_order = {
        "genres": 0,
        "languages": 1,
        "countries": 2,
        "publication_year": 3,
        "formats": 4,
    }
    op_order = {"contains": 0, ">=": 1, "<=": 2}
    return field_order[predicate.field], op_order.get(predicate.op, 9), str(predicate.value)


def _candidate_genres(candidate: Candidate) -> tuple[str, ...]:
    return tuple(str(predicate.value) for predicate in candidate.predicates if predicate.kind == "genre")


def _genre_balance_key(candidate: Candidate, usage: Counter[str]) -> tuple:
    counts = [usage[genre] for genre in _candidate_genres(candidate)]
    return (
        max(counts, default=0),
        sum(counts),
        hashlib.sha256(candidate.signature.encode()).hexdigest(),
    )


def _stable_choice(values: list[str], salt: str) -> str:
    return min(values, key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest())


def matched_reveal_order(size: int, *, family_id: str, seed: int) -> list[int]:
    """Hash order of predicate indices; the first index is the shared opening fact."""
    return sorted(
        range(size),
        key=lambda index: hashlib.sha256(
            f"{MATCHED_REVEAL_ASSIGNMENT_VERSION}:{seed}:{family_id}:{index}".encode()
        ).hexdigest(),
    )


def _tag_constraints_matched(
    predicates: tuple[Predicate, ...],
    reveal: str,
    *,
    family_id: str,
    seed: int,
    implied: dict[tuple[int, int], dict] | None = None,
    hidden_override: list[int] | None = None,
) -> list[TaskConstraint]:
    """Counterbalance which predicate occupies each reveal role per family.

    With ``implied`` (see :func:`implied_constraint_pairs`), the hidden arm
    never withholds a predicate that a revealed predicate implies.
    """
    if len(predicates) < 2:
        raise ValueError("matched reveal families require at least two predicates")
    if reveal == "volunteer":
        tags = [RevealTag.VOLUNTEER] * len(predicates)
    else:
        order = matched_reveal_order(len(predicates), family_id=family_id, seed=seed)
        tags = [RevealTag.ON_ASK] * len(predicates)
        tags[order[0]] = RevealTag.VOLUNTEER
        if reveal == "hidden":
            hidden_count = max(1, len(predicates) // 2)
            if hidden_override is not None:
                hidden = list(hidden_override)
            elif implied is None:
                hidden = order[-hidden_count:]
            else:
                hidden = guarded_hidden_indices(order, hidden_count, implied)
            for index in hidden:
                tags[index] = RevealTag.HIDDEN
        elif reveal != "mixed":
            raise ValueError(f"unknown reveal difficulty {reveal!r}")
    return [
        TaskConstraint(constraint=predicate.as_constraint(), reveal=tag)
        for predicate, tag in zip(predicates, tags)
    ]


def implied_constraint_pairs(
    constraints: list[Constraint], items: Iterable
) -> dict[tuple[int, int], dict]:
    """Return ``(i, j)`` pairs where knowing constraint ``i`` makes ``j`` likely.

    Among items that satisfy ``i`` and carry any value in ``j``'s field, at
    least ``IMPLIED_FACT_MIN_PROBABILITY`` must satisfy ``j``, over at least
    ``IMPLIED_FACT_MIN_SUPPORT`` items, and at least ``IMPLIED_FACT_MIN_LIFT``
    times the rate among all items carrying that field.  Conditioning on the
    field being present keeps sparse labels from hiding an implication such as
    British English -> United Kingdom.
    """
    items = list(items)
    categorical = [
        index for index, constraint in enumerate(constraints)
        if constraint.op == "contains" and constraint.field in IMPLIED_FACT_FIELDS
    ]
    pairs: dict[tuple[int, int], dict] = {}
    for target_index in categorical:
        target = constraints[target_index]
        carrying = [item for item in items if getattr(item, target.field, None)]
        if not carrying:
            continue
        prior = sum(target.evaluate(item) for item in carrying) / len(carrying)
        for source_index in categorical:
            if source_index == target_index:
                continue
            source = constraints[source_index]
            support = [item for item in carrying if source.evaluate(item)]
            if len(support) < IMPLIED_FACT_MIN_SUPPORT:
                continue
            probability = sum(target.evaluate(item) for item in support) / len(support)
            if (
                probability >= IMPLIED_FACT_MIN_PROBABILITY
                and probability >= IMPLIED_FACT_MIN_LIFT * prior
            ):
                pairs[(source_index, target_index)] = {
                    "probability": round(probability, 4),
                    "prior": round(prior, 4),
                    "support": len(support),
                }
    return pairs


def guarded_hidden_indices(
    order: list[int], hidden_count: int, implied: dict[tuple[int, int], dict]
) -> list[int]:
    """Choose hidden predicates as late in ``order`` as possible without leaks.

    The opening fact ``order[0]`` stays revealed.  The unguarded choice,
    ``order[-hidden_count:]``, is kept whenever no predicate left revealed
    implies a hidden one; otherwise the next subset in the same order is used.
    """
    rank = {index: position for position, index in enumerate(order)}
    subsets = sorted(
        itertools.combinations(order[1:], hidden_count),
        key=lambda subset: sorted((rank[index] for index in subset), reverse=True),
        reverse=True,
    )
    for subset in subsets:
        hidden = set(subset)
        if not any(
            source not in hidden and target in hidden for source, target in implied
        ):
            return sorted(subset, key=rank.__getitem__)
    raise ValueError("no hidden assignment avoids an implied-fact leak")


def _task_json(task: Task) -> str:
    return json.dumps(task.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def write_matched_artifacts(
    drafts: list[TaskDraft],
    catalog: Catalog,
    catalog_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
    extra_manifest: dict | None = None,
    track_id: str = "books-wikidata-v2",
    generator_version: str = WIKIDATA_V2_GENERATOR_VERSION,
    constraint_fields: list[str] | None = None,
) -> dict:
    """Freeze the controlled task set and its pairing information."""
    errors = validate_matched_families(drafts)
    if errors:
        raise ValueError("refusing to write invalid matched families: " + "; ".join(errors))

    output_dir = Path(output_dir)
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    for draft in drafts:
        assert draft.task is not None
        name = f"{draft.task.id}.json"
        expected_names.add(name)
        (tasks_dir / name).write_text(_task_json(draft.task), encoding="utf-8")
    for old in tasks_dir.glob("*.json"):
        if old.name not in expected_names:
            old.unlink()

    validator = CatalogValidator(catalog, BOOKS)
    signatures: dict[str, dict] = {}
    answers: dict[str, dict] = {}
    task_hashes: dict[str, str] = {}
    for draft in drafts:
        task = draft.task
        assert task is not None
        assert draft.solution_ids is not None
        assert draft.policy_eligible_ids is not None
        result = validator.validate_task(task)
        if result.errors or not result.valid_as_designed:
            raise RuntimeError(f"{task.id} failed validator: {result.model_dump()}")
        if result.solution_ids != draft.solution_ids:
            raise RuntimeError(f"{task.id}: generator and validator answers disagree")
        signatures[task.id] = task_signature(task, result)
        task_hashes[task.id] = hashlib.sha256(_task_json(task).encode()).hexdigest()
        answer = {
            "family_id": draft.family_id,
            "no_valid_recommendation": task.no_valid_recommendation,
            "constraint_solutions": draft.solution_ids,
            "policy_eligible_solutions": draft.policy_eligible_ids,
            "user_formats": task.user_formats,
            "read_ids": task.history_values(BOOKS),
            "minimal_unsat_leave_one_counts": list(
                draft.candidate.minimal_unsat_counts
            ),
        }
        if draft.query_results is not None:
            solution_set = set(draft.solution_ids)
            answer["posthoc_retrieval_diagnostic"] = {
                "query": draft.candidate.query,
                "top_k": len(draft.query_results),
                "first_solution_rank": next(
                    (
                        rank for rank, item_id in enumerate(draft.query_results, 1)
                        if item_id in solution_set
                    ),
                    None,
                ),
                "result_ids": draft.query_results,
                "used_for_task_selection": False,
            }
        answers[task.id] = answer

    catalog_sha = hash_file(catalog_path)
    signatures["_meta"] = {
        "catalog_sha256": catalog_sha,
        "track": track_id,
        "generator_version": generator_version,
        "design_version": MATCHED_DESIGN_VERSION,
        "reveal_assignment_version": MATCHED_REVEAL_ASSIGNMENT_VERSION,
    }
    family_manifest = matched_family_manifest(drafts)
    family_json = json.dumps(family_manifest, indent=2, sort_keys=True) + "\n"
    (output_dir / "family_manifest.json").write_text(family_json, encoding="utf-8")
    (output_dir / "answers.json").write_text(
        json.dumps(answers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "task_baseline.json").write_text(
        json.dumps(signatures, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    cells = Counter((draft.complexity, draft.reveal) for draft in drafts)
    policies = Counter(
        flag for draft in drafts for flag in draft.task.policy_flags  # type: ignore[union-attr]
    )
    reveal_field_counts = Counter(
        (draft.reveal, tagged.constraint.field, tagged.reveal.value)
        for draft in drafts
        for tagged in draft.task.constraints  # type: ignore[union-attr]
    )
    reserved = {
        "sampling_mode", "design_version", "reveal_assignment_version", "retrieval_role",
    }
    overlap = reserved & set(extra_manifest or {})
    if overlap:
        raise ValueError(f"extra manifest cannot override controlled fields: {sorted(overlap)}")
    manifest = {
        "generator_version": generator_version,
        "design_version": MATCHED_DESIGN_VERSION,
        "reveal_assignment_version": MATCHED_REVEAL_ASSIGNMENT_VERSION,
        "sampling_mode": RETRIEVER_INDEPENDENT_SAMPLING,
        "retrieval_role": (
            "posthoc outcome only; no retriever output is an inclusion or exclusion criterion"
        ),
        "seed": seed,
        "catalog_sha256": catalog_sha,
        "family_manifest_sha256": hashlib.sha256(family_json.encode()).hexdigest(),
        "n_families": family_manifest["family_count"],
        "n_tasks": len(drafts),
        "grid": {
            f"{complexity}/{reveal}": cells[(complexity, reveal)]
            for complexity in COMPLEXITIES for reveal in REVEALS
        },
        "solution_bands": {key: list(value) for key, value in SOLUTION_BANDS.items()},
        "nvr": {
            "family_count": len({
                draft.family_id for draft in drafts if draft.is_nvr
            }),
            "task_count": sum(draft.is_nvr for draft in drafts),
            "design": (
                "minimal unsatisfiable: dropping any one constraint leaves at least "
                "three matches"
            ),
        },
        "policy_counts": dict(sorted(policies.items())),
        "constraint_fields": constraint_fields or [
            "genres", "languages", "countries", "publication_year"
        ],
        "reveal_field_counts": {
            f"{reveal}/{field}/{tag}": count
            for (reveal, field, tag), count in sorted(reveal_field_counts.items())
        },
        "task_sha256": task_hashes,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (output_dir / "task_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Freeze a family-level audit sample so all three reveal variants are
    # reviewed together.  This prevents per-task spot checks from missing a
    # broken pairing.
    family_rows = family_manifest["families"]
    audit_rows = []
    for complexity in COMPLEXITIES:
        eligible = [row for row in family_rows if row["complexity"] == complexity]
        satisfiable = [row for row in eligible if not row["no_valid_recommendation"]]
        nvr = [row for row in eligible if row["no_valid_recommendation"]]
        audit_rows.extend(satisfiable[:5])
        audit_rows.extend(nvr[:1])
    audit = {
        "description": (
            "Family-level audit sample: five satisfiable and one NVR family per "
            "complexity; inspect all reveal variants together"
        ),
        "families": audit_rows,
        "task_sha256": {
            task_id: task_hashes[task_id]
            for row in audit_rows for task_id in row["task_ids"].values()
        },
    }
    (output_dir / "audit_sample.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
