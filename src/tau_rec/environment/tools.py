from __future__ import annotations
import json
from tau_rec.data_model.catalog import Catalog
from tau_rec.catalog.search import CatalogSearch

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Search the movie catalog by keywords. Returns top 20 results.",
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
            "description": "Get full metadata for a specific movie by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "The movie ID"},
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check if a movie is available on specified streaming services.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "The movie ID"},
                    "services": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of streaming service names to check",
                    },
                },
                "required": ["item_id", "services"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_history",
            "description": "Get a user's watch history and ratings.",
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
            "description": (
                "Check whether a content rating is age-restricted. "
                "You MUST call this before recommending any movie rated R or NC-17. "
                "Returns whether the rating is restricted and requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content_rating": {
                        "type": "string",
                        "description": "The content rating to check (e.g. 'PG-13', 'R', 'NC-17').",
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
                "Call with item_id=<movie id> to recommend that movie. "
                "Call with no item_id (or item_id=null) to explicitly abstain "
                "when no movie in the catalog satisfies the user's requirements. "
                "You MUST call this tool exactly once per conversation — either "
                "committing to a movie or abstaining."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": ["string", "null"],
                        "description": "The movie ID to recommend, or null to abstain.",
                    },
                },
                "required": [],
            },
        },
    },
]


class ToolKit:
    def __init__(
        self,
        catalog: Catalog,
        search: CatalogSearch,
        user_histories: dict[str, dict] | None = None,
    ) -> None:
        self._catalog = catalog
        self._search = search
        self._user_histories = user_histories or {}

    def tool_definitions(self) -> list[dict]:
        return TOOL_DEFINITIONS

    def call(self, name: str, arguments: dict) -> str:
        try:
            match name:
                case "search_catalog":
                    return self._search_catalog(arguments["query"])
                case "get_metadata":
                    return self._get_metadata(arguments["item_id"])
                case "check_availability":
                    return self._check_availability(arguments["item_id"], arguments["services"])
                case "get_user_history":
                    return self._get_user_history(arguments["user_id"])
                case "check_content_preference":
                    return self._check_content_preference(arguments["content_rating"])
                case "recommend":
                    return self._recommend(arguments.get("item_id"))
                case _:
                    return json.dumps({"error": f"Unknown tool: {name}"})
        except KeyError as e:
            return json.dumps({"error": f"Missing required argument: {e}"})

    def _search_catalog(self, query: str) -> str:
        results = self._search.search(query, top_k=20)
        return json.dumps(results)

    def _get_metadata(self, item_id: str) -> str:
        movie = self._catalog.get(item_id)
        if movie is None:
            return json.dumps({"error": f"Movie not found: {item_id}"})
        return movie.model_dump_json()

    def _check_availability(self, item_id: str, services: list[str]) -> str:
        movie = self._catalog.get(item_id)
        if movie is None:
            return json.dumps({"error": f"Movie not found: {item_id}"})
        return json.dumps({s: s in movie.streaming_services for s in services})

    def _check_content_preference(self, content_rating: str) -> str:
        restricted = {"R", "NC-17"}
        if content_rating in restricted:
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
        if not item_id:
            return json.dumps({
                "status": "abstained",
                "message": "No recommendation made. Trial will end.",
            })
        movie = self._catalog.get(item_id)
        if movie is None:
            return json.dumps({"error": f"Movie not found: {item_id}"})
        return json.dumps({"status": "recommended", "id": item_id, "title": movie.title})

    def _get_user_history(self, user_id: str) -> str:
        history = self._user_histories.get(user_id)
        if history is None:
            return json.dumps({"error": f"User not found: {user_id}"})
        # Enrich watched list with titles so agents can cross-reference
        enriched = dict(history)
        enriched_watched = []
        for item_id in history.get("watched", []):
            movie = self._catalog.get(item_id)
            entry = {"id": item_id, "title": movie.title if movie else "Unknown"}
            enriched_watched.append(entry)
        enriched["watched"] = enriched_watched
        return json.dumps(enriched)
