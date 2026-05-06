from __future__ import annotations
from rank_bm25 import BM25Okapi
from tau_rec.data_model.catalog import Catalog

class CatalogSearch:
    RESULT_FIELDS = ("id", "title", "genres", "release_date", "rating", "overview")

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog
        self._docs: list[list[str]] = []
        for movie in catalog.movies:
            tokens = (
                movie.title.lower().split()
                + movie.overview.lower().split()
                + [g.lower() for g in movie.genres]
                + [movie.director.lower()]
                + [c.lower() for c in movie.cast]
            )
            self._docs.append(tokens)
        self._bm25 = BM25Okapi(self._docs)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                break
            movie = self._catalog.movies[idx]
            results.append({f: getattr(movie, f) for f in self.RESULT_FIELDS})
        return results
