"""Query and freeze the Wikidata slice used by ``export_wikidata_books_catalog.py``.

Queries a read-only Wikidata mirror (QLever), caches every response by hash,
and builds per-work profiles from the cached responses.  The release contract
is the committed catalog plus the pinned query-response hashes; the published
source snapshot replays the same responses offline.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from tau_rec.catalog.wikidata_books import (
    WIKIDATA_GENRE_MAP,
    WIKIDATA_WORK_TYPES,
    WikidataBookProfile,
    canonical_genres,
    qid_from_value,
    sparql_literal,
    year_from_value,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / ".cache" / "wikidata-books"
DEFAULT_ENDPOINT = "https://qlever.dev/api/wikidata"
USER_AGENT = "tau-rec-wikidata-audit/0.1 (https://github.com/nbharaths/tau-rec)"
SEED = 20260929
MIN_YEAR = 1450
MAX_YEAR = 2026

PREFIXES = """\
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
"""

RELATIONS = {
    "titles": ("rdfs:label", 'FILTER(LANG(?value) = "en")'),
    "descriptions": ("schema:description", 'FILTER(LANG(?value) = "en")'),
    "authors": ("wdt:P50", ""),
    "dates": ("wdt:P577", ""),
    "genres": ("wdt:P136", ""),
    "languages": ("wdt:P407", ""),
    "countries": ("wdt:P495", ""),
    "series": ("wdt:P179", ""),
    "subjects": ("wdt:P921", ""),
    "pages": ("wdt:P1104", ""),
    "formats": ("wdt:P437", ""),
    "isbn13": ("wdt:P212", ""),
    "publishers": ("wdt:P123", ""),
    "instances": ("wdt:P31", ""),
    "forms": ("wdt:P7937", ""),
}


def _values(values: Iterable[str]) -> str:
    return " ".join(f"wd:{value}" for value in values)


def _candidate_body() -> str:
    return f"""
  VALUES ?type {{ {_values(WIKIDATA_WORK_TYPES)} }}
  VALUES ?mappedGenre {{ {_values(WIKIDATA_GENRE_MAP)} }}
  ?item wdt:P31 ?type;
        wdt:P50 ?author;
        wdt:P577 ?date; wdt:P407 ?language;
        wdt:P136 ?mappedGenre;
        rdfs:label ?title.
  FILTER(LANG(?title) = "en")
  FILTER(YEAR(?date) >= {MIN_YEAR} && YEAR(?date) <= {MAX_YEAR})
"""


def _candidate_subquery() -> str:
    return f"SELECT DISTINCT ?item WHERE {{ {_candidate_body()} }}"


def ids_query() -> str:
    return PREFIXES + f"""
SELECT DISTINCT ?item WHERE {{
{_candidate_body()}
}}
"""


def relation_query(predicate: str, suffix: str = "") -> str:
    return PREFIXES + f"""
SELECT DISTINCT ?item ?value WHERE {{
  {{ {_candidate_subquery()} }}
  ?item {predicate} ?value.
  {suffix}
}}
"""


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def parse_tsv(text: str) -> list[dict[str, str]]:
    if not text.strip():
        return []
    rows = csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
    try:
        header = [column.lstrip("?") for column in next(rows)]
    except StopIteration:
        return []
    return [dict(zip(header, row, strict=True)) for row in rows if row]


class CachedHttp:
    def __init__(
        self,
        cache_dir: Path,
        *,
        offline: bool,
        refresh: bool,
        timeout: float,
        request_delay: float,
    ) -> None:
        self.cache_dir = cache_dir
        self.offline = offline
        self.refresh = refresh
        self.timeout = timeout
        self.request_delay = request_delay
        self.last_request = 0.0
        self.records: list[dict] = []
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _pace(self) -> None:
        remaining = self.request_delay - (time.monotonic() - self.last_request)
        if remaining > 0:
            time.sleep(remaining)

    def request(
        self,
        *,
        name: str,
        url: str,
        data: bytes | None,
        extension: str,
    ) -> bytes:
        request_fingerprint = _sha256(url.encode() + b"\0" + (data or b""))
        path = self.cache_dir / f"{name}-{request_fingerprint[:16]}.{extension}"
        if path.is_file() and not self.refresh:
            payload = path.read_bytes()
            self.records.append(
                {
                    "name": name,
                    "cache": str(path.name),
                    "request_sha256": request_fingerprint,
                    "response_sha256": _sha256(payload),
                    "bytes": len(payload),
                    "cache_hit": True,
                }
            )
            return payload
        if self.offline:
            raise RuntimeError(f"offline cache miss for {name}: {path}")

        headers = {"User-Agent": USER_AGENT}
        last_error: Exception | None = None
        for attempt in range(6):
            self._pace()
            request = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                self.last_request = time.monotonic()
                path.write_bytes(payload)
                self.records.append(
                    {
                        "name": name,
                        "cache": str(path.name),
                        "request_sha256": request_fingerprint,
                        "response_sha256": _sha256(payload),
                        "bytes": len(payload),
                        "cache_hit": False,
                    }
                )
                return payload
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {
                    429, 500, 502, 503, 504,
                }
                if not retryable or attempt == 5:
                    raise
                retry_after = 0.0
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        retry_after = float(exc.headers.get("Retry-After", 0))
                    except (TypeError, ValueError):
                        pass
                time.sleep(max(retry_after, min(2**attempt, 16)))
        raise RuntimeError(f"request failed: {last_error}")


class QLeverClient:
    def __init__(self, endpoint: str, http: CachedHttp) -> None:
        self.endpoint = endpoint
        self.http = http

    def query(self, name: str, query: str) -> tuple[list[dict[str, str]], str]:
        print(f"query: {name}", flush=True)
        body = urllib.parse.urlencode({"query": query, "action": "tsv_export"}).encode()
        text = self.http.request(
            name=f"sparql-{name}", url=self.endpoint, data=body, extension="tsv"
        ).decode("utf-8")
        if text.lstrip().startswith("{"):
            error = json.loads(text)
            if error.get("status") == "ERROR" or "exception" in error:
                raise RuntimeError(f"SPARQL query {name} failed: {error.get('exception', error)}")
        return parse_tsv(text), _sha256(query)


def _int_literal(value: str) -> int | None:
    lexical = sparql_literal(value).lstrip("+")
    try:
        number = int(float(lexical))
    except ValueError:
        return None
    return number if number > 0 else None


def _relation_sets(rows: list[dict[str, str]], *, literal: bool = False) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        item_id = qid_from_value(row.get("item", ""))
        value = sparql_literal(row.get("value", "")) if literal else qid_from_value(row.get("value", ""))
        if item_id and value:
            values[item_id].add(value)
    return dict(values)


def build_profiles(
    candidate_ids: list[str], relation_rows: dict[str, list[dict[str, str]]],
    *, genre_map=None,
) -> tuple[list[WikidataBookProfile], dict]:
    genre_map = WIKIDATA_GENRE_MAP if genre_map is None else genre_map
    titles = _relation_sets(relation_rows["titles"], literal=True)
    descriptions = _relation_sets(relation_rows["descriptions"], literal=True)
    authors = _relation_sets(relation_rows["authors"])
    genres = _relation_sets(relation_rows["genres"])
    languages = _relation_sets(relation_rows["languages"])
    countries = _relation_sets(relation_rows["countries"])
    series = _relation_sets(relation_rows["series"])
    subjects = _relation_sets(relation_rows["subjects"])
    formats = _relation_sets(relation_rows["formats"])
    publishers = _relation_sets(relation_rows["publishers"])
    instances = _relation_sets(relation_rows["instances"])
    forms = _relation_sets(relation_rows["forms"])
    isbn13 = _relation_sets(relation_rows["isbn13"], literal=True)

    dates: dict[str, set[int]] = defaultdict(set)
    for row in relation_rows["dates"]:
        item_id = qid_from_value(row.get("item", ""))
        year = year_from_value(row.get("value", ""))
        if item_id and year is not None and MIN_YEAR <= year <= MAX_YEAR:
            dates[item_id].add(year)

    pages: dict[str, set[int]] = defaultdict(set)
    for row in relation_rows["pages"]:
        item_id = qid_from_value(row.get("item", ""))
        value = _int_literal(row.get("value", ""))
        if item_id and value is not None:
            pages[item_id].add(value)

    profiles: list[WikidataBookProfile] = []
    rejected: Counter[str] = Counter()
    for item_id in candidate_ids:
        title_options = titles.get(item_id, set())
        author_ids = tuple(sorted(authors.get(item_id, set())))
        genre_ids = tuple(sorted(genres.get(item_id, set())))
        language_ids = tuple(sorted(languages.get(item_id, set())))
        year_options = dates.get(item_id, set())
        if not title_options:
            rejected["missing_english_title"] += 1
        if not author_ids:
            rejected["missing_author"] += 1
        if not canonical_genres(genre_ids, genre_map=genre_map):
            rejected["missing_mapped_genre"] += 1
        if not language_ids:
            rejected["missing_language"] += 1
        if not year_options:
            rejected["missing_plausible_year"] += 1
        if not (title_options and author_ids and canonical_genres(genre_ids, genre_map=genre_map) and language_ids and year_options):
            continue
        title = min(title_options, key=lambda value: (len(value), value))
        profile = WikidataBookProfile(
            item_id=item_id,
            title=title,
            author_ids=author_ids,
            year=min(year_options),
            genre_ids=genre_ids,
            language_ids=language_ids,
            country_ids=tuple(sorted(countries.get(item_id, set()))),
            series_ids=tuple(sorted(series.get(item_id, set()))),
            subject_ids=tuple(sorted(subjects.get(item_id, set()))),
            pages=max(pages.get(item_id, set()), default=None),
            format_ids=tuple(sorted(formats.get(item_id, set()))),
            isbn13=tuple(sorted(isbn13.get(item_id, set()))),
            publisher_ids=tuple(sorted(publishers.get(item_id, set()))),
            instance_ids=tuple(sorted(instances.get(item_id, set()))),
            form_ids=tuple(sorted(forms.get(item_id, set()))),
            description=min(
                descriptions.get(item_id, {""}), key=lambda value: (len(value), value)
            ),
        )
        profiles.append(profile)
    profiles.sort(key=lambda row: (int(row.item_id[1:]), row.item_id))
    return profiles, {"rejected": dict(sorted(rejected.items()))}
