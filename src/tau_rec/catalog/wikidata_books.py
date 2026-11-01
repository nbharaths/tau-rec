"""Pure helpers for a CC0 Wikidata books catalog.

The Wikidata query module and the catalog exporter share the entity
allowlists and normalization rules in this module.  Network access
deliberately lives outside the package: these helpers operate on
already-fetched Wikidata identifiers and values, so tests and catalog rebuilds
do not depend on a live query service.

Only direct ``instance of`` and ``genre`` statements are accepted.  A
short, reviewed QID allowlist is less inclusive than traversing Wikidata's
changing subclass graph, but it is deterministic and auditable.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Mapping

WIKIDATA_LICENSE_ID = "CC0-1.0"
WIKIDATA_LICENSE_URL = "https://www.wikidata.org/wiki/Wikidata:Licensing"
WIKIDATA_ENTITY_PREFIX = "http://www.wikidata.org/entity/"

# These are work-level classes.  Q571 (book) is excluded from the candidate
# pool because Wikidata uses it for a mixture of works and editions.
WIKIDATA_WORK_TYPES: dict[str, str] = {
    "Q7725634": "literary work",
    "Q47461344": "written work",
}

# Direct P136 values mapped to τ-Rec's compact public taxonomy.  Ambiguous
# values such as generic "fiction" and "romance" (the medieval form) are
# intentionally excluded.  Some unambiguous subgenres map to two useful parent
# labels; this is a declared transform, not an inferred claim about the item.
WIKIDATA_GENRE_MAP: dict[str, tuple[str, ...]] = {
    "Q21802675": ("Adventure",),                 # adventure fiction
    "Q36279": ("Biography",),                    # biography
    "Q56451354": ("Children",),                  # children's fiction
    "Q131539": ("Children",),                    # children's literature
    "Q21615367": ("Children",),                  # children's novel
    "Q8275050": ("Children",),                   # children's book
    "Q5937792": ("Crime",),                      # crime fiction
    "Q20664530": ("Crime",),                     # crime literature
    "Q132311": ("Fantasy",),                     # fantasy
    "Q326439": ("Fantasy",),                     # high fantasy
    "Q1188977": ("Fantasy",),                    # urban fantasy
    "Q1999690": ("Fantasy",),                    # sword and sorcery
    "Q603291": ("Fantasy", "Historical Fiction"),  # historical fantasy
    "Q725377": ("Graphic Novel",),               # graphic novel
    "Q1196408": ("Historical Fiction",),         # historical fiction
    "Q136472": ("Historical Fiction",),          # historical fiction
    "Q10916116": ("History",),                   # history book
    "Q1517777": ("History", "Nonfiction"),      # historical non-fiction
    "Q16575965": ("Horror",),                    # horror fiction
    "Q20669641": ("Horror",),                    # Gothic fiction
    "Q111019576": ("Fantasy", "Horror"),        # vampire fiction
    "Q40831": ("Humor",),                        # comedy
    "Q128758": ("Humor",),                       # satire
    "Q60548032": ("Humor",),                    # humor
    "Q4184": ("Memoir",),                        # autobiography
    "Q112983": ("Memoir",),                      # memoir
    "Q6585139": ("Mystery",),                    # mystery fiction
    "Q186424": ("Mystery",),                     # detective fiction
    "Q213051": ("Nonfiction",),                  # non-fiction
    "Q27801": ("Nonfiction",),                   # non-fiction literature
    "Q482": ("Poetry",),                         # poetry
    "Q19765983": ("Romance",),                   # romantic fiction
    "Q24925": ("Science Fiction",),              # science fiction
    "Q15062348": ("Science Fiction",),           # dystopian fiction
    "Q197949": ("Science Fiction",),             # post-apocalyptic fiction
    "Q3739522": ("Self-Help",),                  # self-help book
    "Q182015": ("Thriller",),                    # thriller
    "Q20664331": ("Thriller",),                  # spy fiction
    "Q111984153": ("Young Adult",),              # young adult fiction
    "Q1233720": ("Young Adult",),                # young adult literature
}

PUBLIC_BOOK_GENRES = tuple(
    sorted({genre for genres in WIKIDATA_GENRE_MAP.values() for genre in genres})
)

# The released map omits two rules that would put canonical literary fiction in
# genre fiction: Gothic fiction is not horror (Wuthering Heights, The Antiquary)
# and dystopian fiction is not science fiction (Animal Farm, Lord of the Flies).
# A work keeps either genre only when Wikidata states it directly.
WIKIDATA_GENRE_MAP_V2: dict[str, tuple[str, ...]] = {
    qid: genres
    for qid, genres in WIKIDATA_GENRE_MAP.items()
    if qid not in {"Q20669641", "Q15062348"}
}

# P7937 (form of creative work) / secondary P31 values that denote a work one
# can reasonably recommend as a book.  Individual short stories, poems, plays,
# chapters and articles are deliberately absent.  Collections are included.
WIKIDATA_BOOKLIKE_FORMS = frozenset(
    {
        "Q571",       # book
        "Q8261",      # novel
        "Q1279564",   # short story collection
        "Q149537",    # novella
        "Q12106333",  # poetry collection
        "Q10916116",  # history book
        "Q747381",    # light novel
        "Q254554",    # picture book
        "Q105420",    # anthology
        "Q125544547", # short story anthology
        "Q5456296",   # fix-up
        "Q16324495",  # essay collection
        "Q20024995",  # novella collection
        "Q108329152", # collection of literary works
        "Q642946",    # gamebook
        "Q12799318",  # short novel
        "Q20540385",  # non-fiction literary work
        "Q133500522", # novel collection
        "Q193495",    # monograph
        "Q725377",    # graphic novel
        "Q5073531",   # chapter book
        "Q8275050",   # children's book
        "Q112983",    # memoir
        "Q36279",     # biography
        "Q104624828", # book form
        "Q128093",    # ebook
        "Q23622",     # dictionary
        "Q185598",    # diary
        "Q4184",      # autobiography
        "Q19357149",  # poetry anthology
        "Q27560760",  # collection of fairy tales
        "Q497976",    # coloring book
        "Q605076",    # cookbook
        "Q5292",      # encyclopedia
        "Q384515",    # treatise
        "Q1164267",   # travel book
        "Q11825892",  # young adult novel
        "Q7797293",   # three-volume novel
        "Q465821",    # epistolary novel
        "Q59126",     # xiaoshuo
        "Q3739522",   # self-help book
        "Q1338914",   # handbook
        "Q13136",     # reference work
    }
)
WIKIDATA_NON_BOOK_FORMS = frozenset(
    {
        "Q3331189",   # version, edition or translation
        "Q57933693",  # book edition
        "Q7553",      # translation
        "Q125309905", # translated work
        "Q732577",    # publication
        "Q49084",     # short story
        "Q472808",    # novelette
        "Q25379",     # play
        "Q5185279",   # poem
        "Q1980247",   # chapter
        "Q1667921",   # novel series
        "Q277759",    # book series
        "Q7725310",   # series of creative works
        "Q47068459",  # children's book series
        "Q21198342",  # manga series
        "Q1347298",   # serialized fiction
        "Q5457615",   # flash fiction
        "Q80056",     # sonnet
        "Q1194583",   # one-act play
        "Q103076",    # screenplay
        "Q191067",    # article
        "Q7366",      # song
        "Q15416",     # television program
        "Q74574",     # fictional book
    }
)

_BOOKLIKE_DESCRIPTION = re.compile(
    r"\b(?:novel|book|memoir|autobiograph(?:y|ical)|biograph(?:y|ical)|"
    r"anthology|collection|monograph|treatise|handbook|encyclopedia|"
    r"dictionary|cookbook|guide|non-fiction work)\b",
    re.IGNORECASE,
)
_NON_BOOK_DESCRIPTION = re.compile(
    r"\b(?:short story|poem|play|chapter|screenplay|television episode|"
    r"television program|song|scholarly article|series of (?:novels|books|"
    r"stories|works))\b",
    re.IGNORECASE,
)
_NON_BOOK_ENTITY_DESCRIPTION = re.compile(
    r"^(?:(?:science[- ]fiction|sci-?fi|fantasy|children(?:'s)?|young adult)\s+)?"
    r"(?:book|novel|light novel|comic book)\s+series\b|"
    r"^(?:\d{4}\s+)?(?:\w+[ -]){0,3}edition of\b",
    re.IGNORECASE,
)

_QID_RE = re.compile(r"Q[1-9][0-9]*$")
_YEAR_RE = re.compile(r"^([+-]?\d+)-")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def qid_from_value(value: str) -> str | None:
    """Return a QID from an RDF IRI, bare QID, or Wikidata JSON value."""

    value = value.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    if value.startswith(WIKIDATA_ENTITY_PREFIX):
        value = value.removeprefix(WIKIDATA_ENTITY_PREFIX)
    return value if _QID_RE.fullmatch(value) else None


def sparql_literal(value: str) -> str:
    """Decode the lexical component of a SPARQL TSV literal.

    QLever emits RDF literals as JSON-compatible quoted strings followed by an
    optional language or datatype suffix.  Avoid interpreting bare values as
    Python literals; identifiers and numeric strings must remain exact.
    """

    value = value.strip()
    if not value.startswith('"'):
        return value
    escaped = False
    for index in range(1, len(value)):
        char = value[index]
        if char == '"' and not escaped:
            import json

            return str(json.loads(value[: index + 1]))
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    raise ValueError(f"unterminated SPARQL literal: {value[:80]!r}")


def year_from_value(value: str) -> int | None:
    """Parse a Wikidata time lexical value and return its signed year."""

    lexical = sparql_literal(value)
    match = _YEAR_RE.match(lexical)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_NON_WORD.sub(" ", value).split())


def stable_order_key(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def canonical_genres(
    raw_genre_ids: Iterable[str], *, genre_map: Mapping[str, tuple[str, ...]] = WIKIDATA_GENRE_MAP,
) -> tuple[str, ...]:
    genres = {
        genre
        for genre_id in raw_genre_ids
        for genre in genre_map.get(genre_id, ())
    }
    return tuple(sorted(genres))


# A trailing "(part II)", "(volume 2)", or "(book 1)" marks a split edition of
# the same work, not a separate work.
_PART_SUFFIX = re.compile(r"\s*\((?:part|vol\.?|volume|book|tome)\s+[\w.]+\)\s*$", re.IGNORECASE)


def exclusion_reason(qid: str, overrides: Mapping, *, title: str, description: str) -> str | None:
    """Why a reviewed override or batch rule removes a work from the catalog, if it does."""
    if overrides.get("overrides", {}).get(qid, {}).get("exclude"):
        return "override_excluded"
    for rule in overrides.get("exclusion_rules", []):
        value = {"title": title, "description": description}[rule["field"]]
        if "equals" in rule and value == rule["equals"]:
            return "exclusion_rule"
        if "startswith" in rule and value.startswith(rule["startswith"]):
            return "exclusion_rule"
    return None


def apply_label_overrides(
    qid: str, fields: dict, overrides: Mapping, *, title: str, description: str,
) -> dict | None:
    """Apply reviewed corrections to one work's public fields; ``None`` drops the work.

    A work is dropped when an override or a batch rule excludes it, or when no
    mapped genre remains after a correction. Exclusion rules match the source
    title and description; an override's ``set`` corrects them afterwards.
    """
    if exclusion_reason(qid, overrides, title=title, description=description):
        return None
    fix = overrides.get("overrides", {}).get(qid)
    if not fix:
        return fields
    fields = dict(fields)
    for field, values in fix.get("remove", {}).items():
        fields[field] = [value for value in fields[field] if value not in values]
    for field, values in fix.get("add", {}).items():
        fields[field] = sorted(set(fields[field]) | set(values))
    if "publication_year" in fix:
        fields["publication_year"] = int(fix["publication_year"])
    unknown = set(fix.get("set", {})) - {"title", "description"}
    if unknown:
        raise ValueError(f"{qid}: override can only set title or description, not {sorted(unknown)}")
    fields.update(fix.get("set", {}))
    return fields if fields["genres"] else None


def dedupe_key_v2(profile: "WikidataBookProfile") -> tuple[str, tuple[str, ...]]:
    return normalize_title(_PART_SUFFIX.sub("", profile.title)), profile.author_ids


@dataclass(frozen=True)
class WikidataBookProfile:
    item_id: str
    title: str
    author_ids: tuple[str, ...]
    year: int
    genre_ids: tuple[str, ...]
    language_ids: tuple[str, ...]
    country_ids: tuple[str, ...] = ()
    series_ids: tuple[str, ...] = ()
    subject_ids: tuple[str, ...] = ()
    pages: int | None = None
    format_ids: tuple[str, ...] = ()
    isbn13: tuple[str, ...] = ()
    publisher_ids: tuple[str, ...] = ()
    instance_ids: tuple[str, ...] = ()
    form_ids: tuple[str, ...] = ()
    description: str = ""

    @property
    def genres(self) -> tuple[str, ...]:
        return canonical_genres(self.genre_ids)

    @property
    def dedupe_key(self) -> tuple[str, tuple[str, ...]]:
        return normalize_title(self.title), self.author_ids


def booklike_reason(profile: WikidataBookProfile) -> str | None:
    """Return the deterministic inclusion reason for a book-length work.

    Explicit non-book types and clearly non-book descriptions take precedence.
    The short English description is otherwise only a fallback for the
    sizeable part of Wikidata that has the generic ``literary work`` type but
    no P7937 statement. Collections and anthologies remain eligible.
    """

    declared = set(profile.instance_ids) | set(profile.form_ids)
    if declared & WIKIDATA_NON_BOOK_FORMS:
        return None
    description = profile.description.strip()
    if _NON_BOOK_ENTITY_DESCRIPTION.search(description):
        return None
    if _NON_BOOK_DESCRIPTION.search(description) and not re.search(
        r"\b(?:collection|anthology)\b", description, re.IGNORECASE
    ):
        return None
    if declared & WIKIDATA_BOOKLIKE_FORMS:
        return "structured_book_form"
    if not description:
        return None
    if _BOOKLIKE_DESCRIPTION.search(description):
        return "description_fallback"
    return None


def deduplicate_profiles(
    profiles: Iterable[WikidataBookProfile],
    *,
    key: Callable[[WikidataBookProfile], tuple[str, tuple[str, ...]]] | None = None,
) -> tuple[list[WikidataBookProfile], dict[str, int]]:
    """Keep one stable QID per normalized title + author-set group."""

    groups: dict[tuple[str, tuple[str, ...]], list[WikidataBookProfile]] = defaultdict(list)
    for profile in profiles:
        groups[key(profile) if key else profile.dedupe_key].append(profile)
    chosen = [min(group, key=lambda row: (int(row.item_id[1:]), row.item_id)) for group in groups.values()]
    chosen.sort(key=lambda row: (int(row.item_id[1:]), row.item_id))
    duplicate_groups = sum(len(group) > 1 for group in groups.values())
    return chosen, {
        "input_records": sum(len(group) for group in groups.values()),
        "unique_records": len(chosen),
        "duplicate_records_removed": sum(len(group) - 1 for group in groups.values()),
        "duplicate_groups": duplicate_groups,
    }


def _mask_for_values(
    profiles: list[WikidataBookProfile], attribute: str,
) -> dict[str, int]:
    masks: dict[str, int] = defaultdict(int)
    for index, profile in enumerate(profiles):
        values = getattr(profile, attribute)
        for value in values:
            masks[value] |= 1 << index
    return dict(masks)


def _year_masks(profiles: list[WikidataBookProfile]) -> tuple[dict[int, int], dict[int, int]]:
    years = range(1450, 2027, 10)
    lower: dict[int, int] = {}
    upper: dict[int, int] = {}
    for threshold in years:
        lower[threshold] = sum(
            1 << index for index, profile in enumerate(profiles) if profile.year >= threshold
        )
        upper[threshold] = sum(
            1 << index for index, profile in enumerate(profiles) if profile.year <= threshold
        )
    return lower, upper


def _keep_frequent(masks: Mapping[str, int], minimum: int) -> dict[str, int]:
    return {key: mask for key, mask in masks.items() if mask.bit_count() >= minimum}


def task_viability(
    profiles: Iterable[WikidataBookProfile],
    *,
    catalog_size: int = 20_000,
    core_size: int = 10_000,
    seed: int = 20260929,
) -> dict:
    """Count source-backed constraint signatures in τ-Rec solution bands.

    This is a feasibility diagnostic, not the task generator.  It only uses
    genre, language, country, and publication-year predicates.  A signature
    must have a witness in the nested core prefix, mirroring the controlled
    study's scale contract.
    """

    ordered = sorted(profiles, key=lambda row: stable_order_key(row.item_id, seed))
    selected = ordered[: min(catalog_size, len(ordered))]
    if not selected:
        return {"catalog_size": 0, "core_size": 0, "bands": {}}
    core_size = min(core_size, len(selected))
    core_mask = (1 << core_size) - 1
    all_mask = (1 << len(selected)) - 1

    genre_masks: dict[str, int] = defaultdict(int)
    for index, profile in enumerate(selected):
        for genre in profile.genres:
            genre_masks[genre] |= 1 << index
    genre_masks = _keep_frequent(genre_masks, 3)
    language_masks = _keep_frequent(_mask_for_values(selected, "language_ids"), 20)
    country_masks = _keep_frequent(_mask_for_values(selected, "country_ids"), 20)
    year_lower, year_upper = _year_masks(selected)

    bands = {
        "simple": (100, 500),
        "medium": (20, 100),
        "complex": (3, 20),
    }
    signatures: dict[str, set[str]] = {name: set() for name in bands}
    genres_used: dict[str, set[str]] = {name: set() for name in bands}

    def consider(complexity: str, name: str, mask: int, genres: Iterable[str]) -> None:
        low, high = bands[complexity]
        count = mask.bit_count()
        if low <= count <= high and mask & core_mask:
            signatures[complexity].add(name)
            genres_used[complexity].update(genres)

    # Simple candidates: one semantic category plus one broad discriminator.
    for genre, genre_mask in genre_masks.items():
        for year, mask in year_lower.items():
            consider("simple", f"g={genre}|year>={year}", genre_mask & mask, (genre,))
        for year, mask in year_upper.items():
            consider("simple", f"g={genre}|year<={year}", genre_mask & mask, (genre,))
        for language, mask in language_masks.items():
            consider("simple", f"g={genre}|lang={language}", genre_mask & mask, (genre,))
        for country, mask in country_masks.items():
            consider("simple", f"g={genre}|country={country}", genre_mask & mask, (genre,))

    # Medium candidates: a bounded era or three source-backed predicates.
    year_windows = [
        (start, min(start + width, 2026))
        for start in range(1450, 2027, 10)
        for width in (20, 40, 80)
        if start + width <= 2026
    ]
    for genre, genre_mask in genre_masks.items():
        for start, end in year_windows:
            mask = genre_mask & year_lower[start] & year_upper[end]
            consider("medium", f"g={genre}|{start}<=year<={end}", mask, (genre,))
        for language, language_mask in language_masks.items():
            for year, year_mask in year_lower.items():
                consider(
                    "medium",
                    f"g={genre}|lang={language}|year>={year}",
                    genre_mask & language_mask & year_mask,
                    (genre,),
                )
        for country, country_mask in country_masks.items():
            for year, year_mask in year_lower.items():
                consider(
                    "medium",
                    f"g={genre}|country={country}|year>={year}",
                    genre_mask & country_mask & year_mask,
                    (genre,),
                )

    # Complex candidates: multi-genre or a bounded era plus another dimension.
    for (genre_a, mask_a), (genre_b, mask_b) in combinations(genre_masks.items(), 2):
        both = mask_a & mask_b
        for start, end in year_windows:
            consider(
                "complex",
                f"g={genre_a}+{genre_b}|{start}<=year<={end}",
                both & year_lower[start] & year_upper[end],
                (genre_a, genre_b),
            )
    for genre, genre_mask in genre_masks.items():
        for start, end in year_windows:
            era = genre_mask & year_lower[start] & year_upper[end]
            for country, country_mask in country_masks.items():
                consider(
                    "complex",
                    f"g={genre}|{start}<=year<={end}|country={country}",
                    era & country_mask,
                    (genre,),
                )
            for language, language_mask in language_masks.items():
                consider(
                    "complex",
                    f"g={genre}|{start}<=year<={end}|lang={language}",
                    era & language_mask,
                    (genre,),
                )

    return {
        "catalog_size": len(selected),
        "core_size": core_size,
        "source_backed_dimensions": ["genres", "language", "country", "publication_year"],
        "predicate_cardinality": {
            "genres": len(genre_masks),
            "languages_with_at_least_20_items": len(language_masks),
            "countries_with_at_least_20_items": len(country_masks),
            "year_thresholds": len(year_lower),
        },
        "bands": {
            complexity: {
                "solution_range": list(bands[complexity]),
                "viable_signatures": len(signatures[complexity]),
                "genres_represented": len(genres_used[complexity]),
                "example_signatures": sorted(signatures[complexity])[:5],
            }
            for complexity in bands
        },
        "unused_mask_bits": max(all_mask.bit_length() - len(selected), 0),
    }
