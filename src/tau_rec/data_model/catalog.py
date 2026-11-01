from __future__ import annotations
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

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

# Fields whose meaning is shared across domains keep the shared name —
# rating, vote_count, content_rating, sponsored, genres — so the evaluator
# reads them without indirection. Fields whose meaning differs get their own
# name (pages, formats) and are reached through DomainSpec. Ratings, when
# present, stay on each source's native scale (e.g. TMDB 0-10); task
# construction matches source semantics.

class Book(BaseModel):
    id: str
    title: str
    authors: list[str]
    publication_year: int | None = None
    pages: int | None = None
    genres: list[str]              # from a committed source->genre map
    description: str
    publisher: str
    rating: float | None = None
    vote_count: int | None = None
    formats: list[str]             # source editions or declared synthetic environment state
    sponsored: bool = False
    content_rating: str = "General"  # a benchmark label, not a source age rating
    languages: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    sitelink_count: int = 0
    source_url: str | None = None
    formats_source: str = "source"

Item = Movie | Book

class Catalog(BaseModel):
    # The field is `items` because the harness is growing tracks whose items
    # are not movies. `movies` stays accepted as a constructor keyword and as
    # a read-only property, so the ten existing call sites and every test
    # keep working; the on-disk format (a bare JSON list) is untouched.
    # The union is for typing: from_json builds items with the spec's model,
    # so pydantic never has to pick a member from a raw dict.
    model_config = ConfigDict(populate_by_name=True)

    items: list[Item] = Field(validation_alias=AliasChoices("items", "movies"))
    _index: dict[str, Item] = {}

    @model_validator(mode="after")
    def _build_index(self) -> "Catalog":
        object.__setattr__(self, "_index", {m.id: m for m in self.items})
        return self

    @property
    def movies(self) -> list[Item]:
        return self.items

    def __len__(self) -> int:
        return len(self.items)

    def get(self, item_id: str) -> Item | None:
        return self._index.get(item_id)

    @classmethod
    def from_json(cls, path: str, spec=None) -> Catalog:
        """Load a bare JSON list of items.

        `spec` is a DomainSpec (duck-typed here to avoid an import cycle);
        None means movies, which is every existing caller.
        """
        import json
        from pathlib import Path
        model = spec.item_model if spec is not None else Movie
        data = json.loads(Path(path).read_text())
        return cls(items=[model(**m) for m in data])
