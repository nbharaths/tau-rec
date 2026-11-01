"""Domain vocabulary the harness renders into everything a model sees.

The harness was written for movies, with the words "movie", "streaming
service", and "watch history" and the set {R, NC-17} spelled out at each site
that needed them. A second domain needs the same sites to say "book" and
"formats". This module is the one place those words and sets live; every
consumer takes a spec and defaults to MOVIES.

Two rules keep this honest. The MOVIES spec must render every string
byte-for-byte as the hardcoded harness did — tests/test_goldens.py pins that,
and a difference is a new harness generation, not a refactor. And a spec
carries *vocabulary*, never behaviour: the tool loop, the scoring, and the
reveal mechanics are the same on every track.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from tau_rec.data_model.catalog import Book, Movie


@dataclass(frozen=True)
class DomainSpec:
    id: str
    item_noun: str                  # "movie"
    item_plural: str                # "movies"
    history_noun: str               # "watch history"
    consumed_adj: str               # "watched"
    history_key: str                # serialized history list: watched | read
    history_contents: str           # what the history tool returns: "watch history and ratings"

    availability_field: str         # item attribute holding where it can be had
    availability_noun: str          # "streaming services"
    availability_singular: str      # "streaming service"
    availability_kind: str          # "streaming availability"
    task_availability_field: str    # user_services | user_formats
    tool_availability_argument: str  # services | formats
    memory_availability_argument: str  # recommend() disambiguator name

    restricted_ratings: tuple[str, ...]   # ordered; a set renders nondeterministically
    age_policy_enabled: bool
    restricted_examples: str        # "R or NC-17"
    rating_examples: str            # "'PG-13', 'R', 'NC-17'"
    rating_note: str                # paragraph appended to the agent prompt

    # BM25 tokenisation per field: "split" a string on whitespace, "whole"
    # keeps a string as one token, "each" lowercases every element of a list.
    search_fields: tuple[tuple[str, str], ...]
    search_result_fields: tuple[str, ...]

    item_model: type
    id_prefix: str
    greeting: str

    # Simulator prompt sections. The first takes {services} and {item_noun}.
    sim_availability_have: str
    sim_availability_none: str

    # Alternate spellings a model may use for an availability value, mapped to
    # the catalog's canonical string. None means exact match only, which is
    # what movies-v1 scored under and must keep.
    service_aliases: dict[str, str] | None = None
    # Match availability values case-insensitively ("Audiobook"); the alias
    # keys are then lowercase.
    casefold_services: bool = False

    @property
    def item_title(self) -> str:
        return self.item_noun.capitalize()

    def normalize_service(self, name: str) -> str:
        if self.service_aliases is None:
            return name
        if self.casefold_services:
            name = name.strip().casefold()
        return self.service_aliases.get(name, name)

    def mentions_service(self, text: str, service: str) -> bool:
        """Whether ``text`` names ``service`` in any spelling normalize_service accepts."""
        spellings = {service} | {
            alias for alias, value in (self.service_aliases or {}).items() if value == service
        }
        flags = re.IGNORECASE if self.casefold_services else 0
        return any(
            re.search(rf"(?<!\w){re.escape(spelling)}s?(?!\w)", text, flags) for spelling in spellings
        )


MOVIES = DomainSpec(
    id="movies",
    item_noun="movie",
    item_plural="movies",
    history_noun="watch history",
    consumed_adj="watched",
    history_key="watched",
    history_contents="watch history and ratings",
    availability_field="streaming_services",
    availability_noun="streaming services",
    availability_singular="streaming service",
    availability_kind="streaming availability",
    task_availability_field="user_services",
    tool_availability_argument="services",
    memory_availability_argument="platform",
    restricted_ratings=("R", "NC-17"),
    age_policy_enabled=True,
    restricted_examples="R or NC-17",
    rating_examples="'PG-13', 'R', 'NC-17'",
    rating_note=(
        'Note on content ratings: "NR" (Not Rated) is NOT the same as "R". Only treat\n'
        "them as interchangeable if the user explicitly says they are fine with\n"
        "unrated content."
    ),
    search_fields=(
        ("title", "split"), ("overview", "split"), ("genres", "each"),
        ("director", "whole"), ("cast", "each"),
    ),
    search_result_fields=("id", "title", "genres", "release_date", "rating", "overview"),
    item_model=Movie,
    id_prefix="tmdb_",
    greeting=(
        "Hi! I can help you find a movie to watch. "
        "What kind of movie are you in the mood for?"
    ),
    sim_availability_have=(
        "\n## Streaming Services You Have\n"
        "You subscribe to: {services}.\n"
        "If the assistant asks which streaming services you have access to, share this list. "
        "If the assistant recommends a {item_noun}, you may assume they've verified availability through their tool."
    ),
    sim_availability_none=(
        "\n## Streaming Services You Have\n"
        "Streaming availability is not a concern for you.\n"
        "If the assistant asks which streaming services you have, say you have access to all major services."
    ),
    service_aliases=None,
)

# The books vocabulary. "General" is a benchmark label,
# not a source age rating, and books currently have no age gate.
BOOKS = DomainSpec(
    id="books",
    item_noun="book",
    item_plural="books",
    history_noun="reading history",
    consumed_adj="read",
    history_key="read",
    history_contents="reading history",
    availability_field="formats",
    availability_noun="formats",
    availability_singular="format",
    availability_kind="format availability",
    task_availability_field="user_formats",
    tool_availability_argument="formats",
    memory_availability_argument="format",
    restricted_ratings=("Adult",),
    age_policy_enabled=False,
    restricted_examples="Adult",
    rating_examples="'General'",
    rating_note=(
        'Note on content labels: "General" is a benchmark default, not an age rating.\n'
        "This track scores no age-gating rule. Catalog metadata can be incomplete; treat\n"
        "publication_year as the source's earliest recorded year and inspect recorded\n"
        "formats rather than inferring availability."
    ),
    search_fields=(
        ("title", "split"), ("description", "split"), ("genres", "each"),
        ("authors", "each"), ("languages", "each"), ("countries", "each"),
    ),
    search_result_fields=(
        "id", "title", "authors", "publication_year", "genres", "languages", "countries",
    ),
    item_model=Book,
    id_prefix="wd_",
    greeting=(
        "Hi! I can help you find a book to read. "
        "What kind of book are you in the mood for?"
    ),
    # The books user prompt appends when to share them (books-llm-user-v3).
    sim_availability_have="\n## Formats You Read\nYou read in these formats: {services}.",
    sim_availability_none="\n## Formats You Read\nFormat is not a concern for you.",
    service_aliases={
        "e-book": "ebook", "e book": "ebook", "ebooks": "ebook", "e-books": "ebook",
        "kindle": "ebook", "kindle edition": "ebook",
        "audio": "audiobook", "audible": "audiobook", "audio book": "audiobook",
        "audio-book": "audiobook", "audio cd": "audiobook", "audiobooks": "audiobook",
        "paper back": "paperback", "paperbacks": "paperback", "softcover": "paperback",
        "soft cover": "paperback",
        "hard cover": "hardcover", "hardcovers": "hardcover", "hardback": "hardcover",
        "hardbacks": "hardcover",
    },
    casefold_services=True,
)

SPECS: dict[str, DomainSpec] = {s.id: s for s in (MOVIES, BOOKS)}


def get_spec(domain: str) -> DomainSpec:
    try:
        return SPECS[domain]
    except KeyError:
        raise ValueError(f"unknown domain {domain!r}; known: {sorted(SPECS)}") from None
