"""Conservative parent-label semantics for source-backed book metadata.

Wikidata represents broad languages/countries and several dialect,
constituent, or historical child labels as separate values. A generic task
predicate such as "languages contains English" should accept an item labelled
"American English"; a specific predicate remains exact. The raw source labels
stay in the catalog for provenance and display.
"""

from __future__ import annotations

from collections.abc import Iterable

BOOK_TAXONOMY_VERSION = "books-source-parent-labels-v1"

LANGUAGE_PARENTS = {
    "American English": "English",
    "Australian English": "English",
    "British English": "English",
    "Canadian English": "English",
    "Hiberno-English": "English",
    "New Zealand English": "English",
    "Scottish English": "English",
    "Austrian German": "German",
    "Berlin German": "German",
    "Swiss German": "German",
    "Brazilian Portuguese": "Portuguese",
    "Canadian French": "French",
    "French of France": "French",
    "Quebec French": "French",
    "Latin American Spanish": "Spanish",
    "Peninsular Spanish": "Spanish",
    "Philippine Spanish": "Spanish",
    "Finland Swedish": "Swedish",
    "Mandarin": "Chinese",
    "Putonghua": "Chinese",
    "Simplified Chinese": "Chinese",
    "Standard Chinese": "Chinese",
    "Traditional Chinese": "Chinese",
    "written Chinese language": "Chinese",
}

COUNTRY_PARENTS = {
    "England": "United Kingdom",
    "Scotland": "United Kingdom",
    "Wales": "United Kingdom",
    "Kingdom of England": "United Kingdom",
    "Kingdom of Great Britain": "United Kingdom",
    "United Kingdom of Great Britain and Ireland": "United Kingdom",
    "United States of America": "United States",
}

PARENTS_BY_FIELD = {
    "languages": LANGUAGE_PARENTS,
    "countries": COUNTRY_PARENTS,
}

# books-wikidata-v2 bakes a wider parent table into the catalog itself: each
# record's languages/countries hold the source labels plus these derived
# parents, and the raw labels move to source_languages/source_countries. Only
# a direct predecessor state maps to a modern country. Multinational states
# (Soviet Union, Ottoman Empire, Yugoslavia, Austria-Hungary, Holy Roman
# Empire) and colonial administrations get no parent. Tracks without derived
# labels keep the runtime parent semantics above.
BOOK_TAXONOMY_V2_VERSION = "books-derived-parent-labels-v2"
LANGUAGE_PARENTS_V2 = {
    **LANGUAGE_PARENTS,
    "Classical Chinese": "Chinese",
    "Early Modern English": "English",
    "Early Modern Spanish": "Spanish",
    "Modern Greek": "Greek",
    "Neo-Latin": "Latin",
}
COUNTRY_PARENTS_V2 = {
    **COUNTRY_PARENTS,
    "People's Republic of China": "China",
    "Ming dynasty": "China",
    "Qing dynasty": "China",
    "Kingdom of the Netherlands": "Netherlands",
    "German Empire": "Germany",
    "German Reich": "Germany",
    "Weimar Republic": "Germany",
    "German Democratic Republic": "Germany",
    "German Confederation": "Germany",
    "Free City of Frankfurt": "Germany",
    "Kingdom of Württemberg": "Germany",
    "Empire of Japan": "Japan",
    "Kingdom of Italy": "Italy",
    "Grand Duchy of Tuscany": "Italy",
    "Republic of Venice": "Italy",
    "July Monarchy": "France",
    "French Third Republic": "France",
    "Austrian Empire": "Austria",
    "First Republic of Austria": "Austria",
    "Russian Empire": "Russia",
    "Russian Socialist Federative Soviet Republic": "Russia",
    "Russian Soviet Federative Socialist Republic": "Russia",
    "Azerbaijan Soviet Socialist Republic": "Azerbaijan",
    "Kirghiz Soviet Socialist Republic": "Kyrgyzstan",
}
_PARENTS_V2_BY_FIELD = {"languages": LANGUAGE_PARENTS_V2, "countries": COUNTRY_PARENTS_V2}


def derived_labels_v2(field: str, values: Iterable[str]) -> list[str]:
    """Source labels plus their v2 parents, sorted, for a v2 catalog record."""
    labels = {str(value) for value in values}
    parents = _PARENTS_V2_BY_FIELD[field]
    pending = list(labels)
    while pending:
        parent = parents.get(pending.pop())
        if parent is not None and parent not in labels:
            labels.add(parent)
            pending.append(parent)
    return sorted(labels)


def expand_parent_labels(field: str, values: Iterable[str]) -> set[str]:
    """Return raw values plus their conservative parents."""
    expanded = {str(value) for value in values}
    parents = PARENTS_BY_FIELD.get(field, {})
    pending = list(expanded)
    while pending:
        value = pending.pop()
        parent = parents.get(value)
        if parent is not None and parent not in expanded:
            expanded.add(parent)
            pending.append(parent)
    return expanded


def contains_label(field: str, values: Iterable[str], required: str) -> bool:
    return required in expand_parent_labels(field, values)


def casefolded_parent_labels(field: str, values: Iterable[str]) -> set[str]:
    return {value.casefold() for value in expand_parent_labels(field, values)}
