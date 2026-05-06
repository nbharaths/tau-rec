from __future__ import annotations
from pydantic import BaseModel, model_validator

class Movie(BaseModel):
    id: str
    title: str
    release_date: str
    runtime: int
    genres: list[str]
    overview: str
    cast: list[str]
    director: str
    rating: float
    vote_count: int
    streaming_services: list[str]
    sponsored: bool = False
    content_rating: str = "NR"

class Catalog(BaseModel):
    movies: list[Movie]
    _index: dict[str, Movie] = {}

    @model_validator(mode="after")
    def _build_index(self) -> "Catalog":
        object.__setattr__(self, "_index", {m.id: m for m in self.movies})
        return self

    def __len__(self) -> int:
        return len(self.movies)

    def get(self, item_id: str) -> Movie | None:
        return self._index.get(item_id)

    @classmethod
    def from_json(cls, path: str) -> Catalog:
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text())
        return cls(movies=[Movie(**m) for m in data])
