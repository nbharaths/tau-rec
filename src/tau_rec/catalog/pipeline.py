from __future__ import annotations
import json
from pathlib import Path
import httpx
from tau_rec.data_model.catalog import Movie, Catalog


_SERVICE_SUFFIXES = (
    " Standard with Ads", " with Ads",
    " Apple TV Channel", " Amazon Channel",
    " Roku Premium Channel", " Roku Channel",
    " Premium Channel",
    " Premium Plus", " Premium", " Essential", " Basic",
)

# Aliases that collapse alternate brand spellings to a single canonical form.
_SERVICE_ALIASES = {
    "AMC Plus": "AMC+",
    "Paramount Plus": "Paramount+",
    "Disney Plus": "Disney+",
    "Apple TV Plus": "Apple TV+",
    "Apple TV": "Apple TV+",
    "MGM Plus": "MGM+",
}


def normalize_service_name(name: str) -> str:
    """Collapse TMDB tier variants onto the base service name.

    Examples:
        "Netflix Standard with Ads" -> "Netflix"
        "Starz Apple TV Channel"    -> "Starz"
        "AMC Plus Apple TV Channel" -> "AMC+"
        "Peacock Premium Plus"      -> "Peacock"
        "Disney Plus"               -> "Disney+"
    """
    s = name.strip()
    changed = True
    while changed:
        changed = False
        for suf in _SERVICE_SUFFIXES:
            if s.endswith(suf):
                s = s[: -len(suf)].strip()
                changed = True
                break
    return _SERVICE_ALIASES.get(s, s)


def normalize_services(services: list[str]) -> list[str]:
    """Normalize and deduplicate a list of service names, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for s in services:
        n = normalize_service_name(s)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


class TMDBPipeline:
    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str, region: str = "US") -> None:
        self._api_key = api_key
        self._region = region

    async def build_catalog(
        self, cutoff_date: str, min_vote_count: int = 5, output_path: str | None = None,
    ) -> Catalog:
        async with httpx.AsyncClient() as client:
            movie_ids = await self._discover_movies(client, cutoff_date, min_vote_count)
            movies = []
            for mid in movie_ids:
                raw = await self._fetch_movie_detail(client, mid)
                if raw:
                    movie = self._parse_movie(raw, self._region)
                    if movie:
                        movies.append(movie)
        catalog = Catalog(movies=movies)
        if output_path:
            Path(output_path).write_text(json.dumps([m.model_dump() for m in movies], indent=2))
        return catalog

    async def _discover_movies(self, client: httpx.AsyncClient, cutoff_date: str, min_vote_count: int) -> list[int]:
        ids = []
        page = 1
        while True:
            resp = await client.get(f"{self.BASE_URL}/discover/movie", params={
                "api_key": self._api_key, "primary_release_date.gte": cutoff_date,
                "vote_count.gte": min_vote_count, "sort_by": "primary_release_date.asc", "page": page,
            })
            data = resp.json()
            for item in data.get("results", []):
                ids.append(item["id"])
            if page >= data.get("total_pages", 1):
                break
            page += 1
        return ids

    async def _fetch_movie_detail(self, client: httpx.AsyncClient, movie_id: int) -> dict | None:
        resp = await client.get(f"{self.BASE_URL}/movie/{movie_id}", params={
            "api_key": self._api_key, "append_to_response": "credits,watch/providers,release_dates",
        })
        if resp.status_code == 200:
            return resp.json()
        return None

    @staticmethod
    def _parse_movie(raw: dict, region: str = "US") -> Movie | None:
        try:
            director = "Unknown"
            credits = raw.get("credits", {})
            for crew in credits.get("crew", []):
                if crew.get("job") == "Director":
                    director = crew["name"]
                    break
            cast = [c["name"] for c in credits.get("cast", [])[:5]]
            genres = [g["name"] for g in raw.get("genres", [])]
            providers = raw.get("watch/providers", {}).get("results", {})
            region_data = providers.get(region, {})
            streaming = normalize_services(
                [p["provider_name"] for p in region_data.get("flatrate", [])]
            )
            content_rating = "NR"
            for rd in raw.get("release_dates", {}).get("results", []):
                if rd.get("iso_3166_1") == region:
                    for d in rd.get("release_dates", []):
                        if d.get("certification"):
                            content_rating = d["certification"]
                            break
            return Movie(
                id=f"tmdb_{raw['id']}", title=raw["title"],
                release_date=raw.get("release_date", ""), runtime=raw.get("runtime", 0),
                genres=genres, overview=raw.get("overview", ""), cast=cast, director=director,
                rating=raw.get("vote_average", 0.0), vote_count=raw.get("vote_count", 0),
                streaming_services=streaming, sponsored=False, content_rating=content_rating,
            )
        except (KeyError, TypeError):
            return None
