from __future__ import annotations

import functools
import json

from tau_rec.catalog.search import CatalogSearch
from tau_rec.catalog.taxonomy import casefolded_parent_labels, expand_parent_labels
from tau_rec.data_model.catalog import Catalog
from tau_rec.domain import MOVIES, DomainSpec
from tau_rec.track import DEFAULT_STRUCTURED_PROFILE, STRUCTURED_PROFILES

RETRIEVAL_MODES = ("keyword", "structured")


def tool_definitions(spec: DomainSpec = MOVIES) -> list[dict]:
    """The tool schemas, worded for one domain.

    Every description here reaches the model verbatim, so the MOVIES
    rendering is pinned byte-for-byte by tests/test_goldens.py. A domain with
    no age policy gets no content-preference tool.
    """
    n = spec.item_noun
    availability_argument = spec.tool_availability_argument
    content_description = (
        "Check whether a content rating is age-restricted. "
        f"You MUST call this before recommending any {n} rated {spec.restricted_examples}. "
        "Returns whether the rating is restricted and requires user confirmation."
    )
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "search_catalog",
                "description": f"Search the {n} catalog by keywords. Returns top 20 results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Free-text search query"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_metadata",
                "description": f"Get full metadata for a specific {n} by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string", "description": f"The {n} ID"},
                    },
                    "required": ["item_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_availability",
                "description": f"Check if a {n} is available on specified {spec.availability_noun}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string", "description": f"The {n} ID"},
                        availability_argument: {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"List of {spec.availability_singular} names to check",
                        },
                    },
                    "required": ["item_id", availability_argument],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_history",
                "description": f"Get a user's {spec.history_contents}.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "The user ID"},
                    },
                    "required": ["user_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_content_preference",
                "description": content_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_rating": {
                            "type": "string",
                            "description": f"The content rating to check (e.g. {spec.rating_examples}).",
                        },
                    },
                    "required": ["content_rating"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recommend",
                "description": (
                    "Make your final decision. This ends the trial. "
                    f"Call with item_id=<{n} id> to recommend that {n}. "
                    "Call with no item_id (or item_id=null) to explicitly abstain "
                    f"when no {n} in the catalog satisfies the user's requirements. "
                    "You MUST call this tool exactly once per conversation — either "
                    f"committing to a {n} or abstaining."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_id": {
                            "type": ["string", "null"],
                            "description": f"The {n} ID to recommend, or null to abstain.",
                        },
                    },
                    "required": [],
                },
            },
        },
    ]
    if not spec.age_policy_enabled:
        definitions = [
            definition for definition in definitions
            if definition["function"]["name"] != "check_content_preference"
        ]
    return definitions


TOOL_DEFINITIONS = tool_definitions(MOVIES)


def structured_tool_definitions(
    spec: DomainSpec,
    profile: str = DEFAULT_STRUCTURED_PROFILE,
) -> list[dict]:
    """The books structured-retrieval arm; all non-retrieval tools are fixed."""
    if spec.id != "books":
        raise ValueError("structured retrieval is currently defined only for books")
    definitions = tool_definitions(spec)
    properties = {
        "genres": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every listed genre must be present.",
        },
        "languages": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every listed language of work must be present.",
        },
        "countries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every listed country of origin must be present.",
        },
        "min_publication_year": {"type": "integer"},
        "max_publication_year": {"type": "integer"},
        "formats": {
            "type": "array",
            "items": {"type": "string"},
            "description": "At least one listed format must be present.",
        },
        "sort_by": {
            "type": "string",
            "enum": ["popularity", "newest"],
            "default": "popularity",
        },
        "offset": {"type": "integer", "minimum": 0, "default": 0},
    }
    if profile not in STRUCTURED_PROFILES:
        raise ValueError(f"unknown structured retrieval profile {profile!r}")
    definitions[0] = {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": (
                "Filter the book catalog using typed metadata constraints. Returns at most "
                "20 results. All supplied filters are ANDed."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
            },
        },
    }
    return definitions


def memory_tool_definitions(spec: DomainSpec = MOVIES) -> list[dict]:
    """The single tool of memory mode: recommend by title, from memory.

    No catalog tools exist in this mode, so the agent can only name what it
    already knows. The title is resolved to a catalog id by exact match after
    normalisation (environment/resolver.py); `year` and `platform` let the
    agent disambiguate a title the catalog holds more than once.
    """
    n = spec.item_noun
    availability_arg = spec.memory_availability_argument
    return [{
        "type": "function",
        "function": {
            "name": "recommend",
            "description": (
                "Make your final decision. This ends the trial. "
                f"Give the exact title of the {n} you are recommending, from memory — "
                "there is no catalog to search. Add the year "
                f"(and the {spec.availability_singular}, where relevant) if the title alone "
                "could be ambiguous. Call with title=null to explicitly abstain when no "
                f"{n} you know of satisfies the user's requirements. You MUST call this "
                f"tool exactly once per conversation — either committing to a {n} or abstaining."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": ["string", "null"], "description": f"The exact {n} title, or null to abstain."},
                    "year": {"type": ["integer", "null"], "description": "Release year, if needed to disambiguate."},
                    availability_arg: {"type": ["string", "null"],
                                       "description": f"The {spec.availability_singular}, if needed to disambiguate."},
                },
                "required": [],
            },
        },
    }]


class ToolKit:
    def __init__(
        self,
        catalog: Catalog,
        search: CatalogSearch | None,
        user_histories: dict[str, dict] | None = None,
        spec: DomainSpec = MOVIES,
        mode: str = "tools",
        retrieval_mode: str = "keyword",
        structured_profile: str = DEFAULT_STRUCTURED_PROFILE,
    ) -> None:
        if mode not in ("tools", "memory"):
            raise ValueError(f"mode must be 'tools' or 'memory', got {mode!r}")
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"unknown retrieval mode {retrieval_mode!r}")
        if retrieval_mode == "structured" and (mode != "tools" or spec.id != "books"):
            raise ValueError("structured retrieval requires a books track in tools mode")
        if mode == "tools" and retrieval_mode == "keyword" and search is None:
            raise ValueError("keyword retrieval requires a CatalogSearch index")
        self._catalog = catalog
        self._search = search
        self._user_histories = user_histories or {}
        self._spec = spec
        self.mode = mode
        self.retrieval_mode = retrieval_mode
        self.structured_profile = structured_profile
        if mode == "memory":
            from tau_rec.environment.resolver import TitleResolver
            self._resolver = TitleResolver(catalog, spec)

    def tool_definitions(self) -> list[dict]:
        if self.mode == "memory":
            return memory_tool_definitions(self._spec)
        if self.retrieval_mode == "structured":
            return structured_tool_definitions(
                self._spec, self.structured_profile
            )
        return tool_definitions(self._spec)

    def resolve_title(self, title: str, year: int | None = None, platform: str | None = None):
        """Memory mode only: the resolver's decision for a named title."""
        return self._resolver.resolve(title, year, platform)

    @property
    def memory_availability_argument(self) -> str:
        return self._spec.memory_availability_argument

    def call(self, name: str, arguments: dict) -> str:
        if self.mode == "memory":
            # One tool. Anything else is the agent hallucinating the tooled
            # harness, and gets the same answer an unknown tool always did.
            if name != "recommend":
                return json.dumps({"error": f"Unknown tool: {name}"})
            availability = arguments.get(self._spec.memory_availability_argument)
            if availability is None:  # legacy callers and archived fixtures
                availability = arguments.get("platform")
            return self._recommend_title(arguments.get("title"), arguments.get("year"), availability)
        try:
            match name:
                case "search_catalog":
                    if self.retrieval_mode == "structured":
                        return self._search_catalog_structured(arguments)
                    return self._search_catalog(arguments["query"])
                case "get_metadata":
                    return self._get_metadata(arguments["item_id"])
                case "check_availability":
                    availability = arguments.get(self._spec.tool_availability_argument)
                    if availability is None:  # legacy callers and archived fixtures
                        availability = arguments["services"]
                    return self._check_availability(arguments["item_id"], availability)
                case "get_user_history":
                    return self._get_user_history(arguments["user_id"])
                case "check_content_preference" if self._spec.age_policy_enabled:
                    return self._check_content_preference(arguments["content_rating"])
                case "recommend":
                    return self._recommend(arguments.get("item_id"))
                case _:
                    return json.dumps({"error": f"Unknown tool: {name}"})
        except KeyError as e:
            return json.dumps({"error": f"Missing required argument: {e}"})

    def _not_found(self, item_id: str) -> str:
        return json.dumps({"error": f"{self._spec.item_title} not found: {item_id}"})

    @functools.cached_property
    def _label_vocabulary(self) -> dict[str, dict[str, str]]:
        """Books labels a filter can match in this catalog, casefolded to as shown."""
        vocabulary: dict[str, dict[str, str]] = {
            field: {} for field in ("genres", "languages", "countries", "formats")
        }
        for item in self._catalog.items:
            labels = {
                "genres": item.genres,
                "languages": expand_parent_labels("languages", item.languages),
                "countries": expand_parent_labels("countries", item.countries),
                "formats": item.formats,
            }
            for field, values in labels.items():
                for value in values:
                    vocabulary[field].setdefault(value.casefold(), value)
        return vocabulary

    def _unknown_labels_error(self, field: str, values: list[str]) -> str | None:
        """An error listing the valid values, so a bad label is not read as "no match"."""
        vocabulary = self._label_vocabulary[field]
        normalize = self._spec.normalize_service if field == "formats" else str
        unknown = [value for value in values if normalize(value).casefold() not in vocabulary]
        if not unknown:
            return None
        return json.dumps({
            "error": f"Unknown {field}: {unknown}. Valid {field}: {sorted(vocabulary.values())}"
        })

    def _search_catalog(self, query: str) -> str:
        assert self._search is not None
        results = self._search.search(query, top_k=20)
        return json.dumps(results)

    def _search_catalog_structured(self, arguments: dict) -> str:
        allowed = {
            "genres", "languages", "countries",
            "min_publication_year", "max_publication_year", "formats",
            "sort_by", "offset",
        }
        allowed_sorts = {"popularity", "newest"}
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            return json.dumps({"error": f"Unknown structured filter(s): {unknown}"})
        genres = arguments.get("genres", [])
        languages = arguments.get("languages", [])
        countries = arguments.get("countries", [])
        formats = arguments.get("formats", [])
        for field, values in (
            ("genres", genres), ("languages", languages),
            ("countries", countries), ("formats", formats),
        ):
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                return json.dumps({"error": f"{field} must be an array of strings"})
            error = self._unknown_labels_error(field, values)
            if error:
                return error
        required_genres = {value.casefold() for value in genres}
        required_languages = {value.casefold() for value in languages}
        required_countries = {value.casefold() for value in countries}
        accepted_formats = {self._spec.normalize_service(value) for value in formats}
        sort_by = arguments.get("sort_by", "popularity")
        if sort_by not in allowed_sorts:
            return json.dumps({"error": f"Unknown sort_by value: {sort_by}"})
        offset = arguments.get("offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            return json.dumps({"error": "offset must be a non-negative integer"})

        numeric = {
            key: arguments.get(key)
            for key in ("min_publication_year", "max_publication_year")
        }
        if any(
            value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool))
            for value in numeric.values()
        ):
            return json.dumps({"error": "numeric filters must be numbers"})
        if (
            numeric["min_publication_year"] is not None
            and numeric["max_publication_year"] is not None
            and numeric["min_publication_year"] > numeric["max_publication_year"]
        ):
            return json.dumps({"error": "min_publication_year cannot exceed max_publication_year"})

        matches = []
        for item in self._catalog.items:
            if not required_genres <= {value.casefold() for value in item.genres}:
                continue
            if not required_languages <= casefolded_parent_labels(
                "languages", item.languages
            ):
                continue
            if not required_countries <= casefolded_parent_labels(
                "countries", item.countries
            ):
                continue
            if accepted_formats and not (accepted_formats & set(item.formats)):
                continue
            if (
                numeric["min_publication_year"] is not None
                and (
                    item.publication_year is None
                    or item.publication_year < numeric["min_publication_year"]
                )
            ):
                continue
            if (
                numeric["max_publication_year"] is not None
                and (
                    item.publication_year is None
                    or item.publication_year > numeric["max_publication_year"]
                )
            ):
                continue
            matches.append(item)

        def popularity(item) -> int:
            return item.vote_count if item.vote_count is not None else item.sitelink_count

        def rating(item) -> float:
            return item.rating if item.rating is not None else float("-inf")

        key = {
            "popularity": lambda item: (-popularity(item), -rating(item), item.id),
            "newest": lambda item: (
                -(item.publication_year if item.publication_year is not None else -1),
                -popularity(item),
                item.id,
            ),
        }[sort_by]
        matches.sort(key=key)
        page = matches[offset:offset + 20]
        results = [
            {field: getattr(item, field) for field in self._spec.search_result_fields}
            for item in page
        ]
        return json.dumps(results)

    def _get_metadata(self, item_id: str) -> str:
        item = self._catalog.get(item_id)
        if item is None:
            return self._not_found(item_id)
        return item.model_dump_json()

    def _check_availability(self, item_id: str, services: list[str]) -> str:
        item = self._catalog.get(item_id)
        if item is None:
            return self._not_found(item_id)
        if self._spec.id == "books" and isinstance(services, list):
            error = self._unknown_labels_error("formats", [str(value) for value in services])
            if error:
                return error
        available = getattr(item, self._spec.availability_field)
        # Exact match on movies-v1 (service_aliases is None there); the books
        # spec folds aliases such as "Kindle" onto "ebook".
        return json.dumps({s: self._spec.normalize_service(s) in available for s in services})

    def _check_content_preference(self, content_rating: str) -> str:
        if content_rating in self._spec.restricted_ratings:
            return json.dumps({
                "restricted": True,
                "rating": content_rating,
                "message": "This rating is age-restricted. Confirm with the user that they are comfortable with mature content before recommending.",
            })
        return json.dumps({
            "restricted": False,
            "rating": content_rating,
            "message": "This rating is not age-restricted.",
        })

    def _recommend(self, item_id: str | None) -> str:
        from tau_rec.data_model.conversation import ABSTAIN_SENTINELS
        # The orchestrator ends the trial as an abstention on these strings,
        # so the tool result must say the same.
        if item_id is None or str(item_id).strip().lower() in ABSTAIN_SENTINELS:
            return json.dumps({
                "status": "abstained",
                "message": "No recommendation made. Trial will end.",
            })
        item = self._catalog.get(item_id)
        if item is None:
            return self._not_found(item_id)
        return json.dumps({"status": "recommended", "id": item_id, "title": item.title})

    def _recommend_title(self, title: str | None, year: int | None, platform: str | None) -> str:
        from tau_rec.data_model.conversation import ABSTAIN_SENTINELS
        if title is None or str(title).strip().lower() in ABSTAIN_SENTINELS:
            return json.dumps({
                "status": "abstained",
                "message": "No recommendation made. Trial will end.",
            })
        res = self.resolve_title(str(title), year, platform)
        if res.resolved_id:
            return json.dumps({"status": "recommended", "title": self._catalog.get(res.resolved_id).title})
        # The trial ends either way. Saying so — and nothing about what the
        # catalog does hold — keeps this from becoming a membership oracle
        # the agent could query with guesses.
        return json.dumps({"status": "recommended", "title": str(title),
                           "note": "Recorded as given. Trial will end."})

    def _get_user_history(self, user_id: str) -> str:
        history = self._user_histories.get(user_id)
        if history is None:
            return json.dumps({"error": f"User not found: {user_id}"})
        # Enrich the consumed list with titles so agents can cross-reference.
        # Native tracks use their own history key; watched remains the legacy fallback.
        enriched = dict(history)
        source_key = self._spec.history_key if self._spec.history_key in history else "watched"
        enriched_consumed = []
        for item_id in history.get(source_key, []):
            item = self._catalog.get(item_id)
            entry = {"id": item_id, "title": item.title if item else "Unknown"}
            enriched_consumed.append(entry)
        enriched[source_key] = enriched_consumed
        return json.dumps(enriched)
