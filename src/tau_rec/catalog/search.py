from __future__ import annotations

import heapq
import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

from tau_rec.data_model.catalog import Catalog
from tau_rec.domain import MOVIES, DomainSpec

LEGACY_PROFILE = "bm25-v1"
BM25F_PROFILE = "bm25f-v1"

# Fixed as part of the books retrieval contract, not exposed as tuning
# knobs. Separate field indexes preserve meaningful length normalization.
BM25F_WEIGHTS = {
    "title": 4.0,
    "genres": 3.0,
    "authors": 2.0,
    "languages": 2.0,
    "countries": 2.0,
    "description": 1.0,
}

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(value, mode: str) -> list[str]:
    """Legacy tokenisation; changing it would move existing track behavior."""
    if value is None:
        return []
    if mode == "split":
        return str(value).lower().split()
    if mode == "whole":
        return [str(value).lower()]
    if mode == "each":
        return [str(v).lower() for v in value]
    raise ValueError(f"unknown tokenisation mode {mode!r}")


def _normalized_text_tokens(value: object, *, phrases: bool = True) -> list[str]:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    words = _WORD_RE.findall(text)
    if not phrases:
        return words
    bigrams = [f"__phrase__:{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


def _field_tokens(value: object, mode: str, *, phrases: bool) -> list[str]:
    if value is None:
        return []
    if mode == "each":
        out: list[str] = []
        for element in value:
            out.extend(_normalized_text_tokens(element, phrases=phrases))
        return out
    return _normalized_text_tokens(value, phrases=phrases)


class CatalogSearch:
    RESULT_FIELDS = MOVIES.search_result_fields

    def __init__(
        self,
        catalog: Catalog,
        spec: DomainSpec = MOVIES,
        profile: str = LEGACY_PROFILE,
    ) -> None:
        if profile not in (LEGACY_PROFILE, BM25F_PROFILE):
            raise ValueError(f"unknown search profile {profile!r}")
        self._catalog = catalog
        self._result_fields = spec.search_result_fields
        self.profile = profile
        self._docs: list[list[str]] = []
        self._bm25_fields: list[tuple[float, BM25Okapi]] = []
        if profile == LEGACY_PROFILE:
            for item in catalog.items:
                tokens: list[str] = []
                for field, mode in spec.search_fields:
                    tokens += _tokens(getattr(item, field), mode)
                self._docs.append(tokens)
            self._bm25 = BM25Okapi(self._docs)
        else:
            for field, mode in spec.search_fields:
                # Bigrams materially help multiword titles and fixed genre
                # labels. On descriptions they inflate the index for little
                # retrieval value, so that field remains word-only.
                phrases = field in {"title", "genres"}
                docs = [_field_tokens(getattr(item, field), mode, phrases=phrases) for item in catalog.items]
                # A domain-wide spec may include a field a catalog does not
                # populate.  An all-empty BM25 corpus is undefined, so the
                # field is skipped when every document lacks it.
                if not any(docs):
                    continue
                weight = BM25F_WEIGHTS.get(field, 1.0)
                self._bm25_fields.append((weight, BM25Okapi(docs)))

    def _scores(self, query: str):
        if self.profile == LEGACY_PROFILE:
            return self._bm25.get_scores(query.lower().split())
        query_tokens = _normalized_text_tokens(query)
        scores = np.zeros(len(self._catalog.items), dtype=float)
        for weight, index in self._bm25_fields:
            scores += weight * index.get_scores(query_tokens)
        return scores

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        scores = self._scores(query)
        # Equivalent to stable score-descending/index-ascending sorting, but
        # O(N log k) rather than sorting all N scores for every query.
        ranked = heapq.nsmallest(
            min(top_k, len(scores)),
            enumerate(scores),
            key=lambda pair: (-float(pair[1]), pair[0]),
        )
        results = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                break
            item = self._catalog.items[idx]
            results.append({f: getattr(item, f) for f in self._result_fields})
        return results
