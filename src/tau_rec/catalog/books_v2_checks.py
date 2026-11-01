"""Independent checks of the books-wikidata-v2 generation rules.

The rules are re-implemented here from their written statement rather than by
calling the generator, so a generator bug cannot certify itself. The only
shared inputs are the label-parent tables. Used by
``scripts/validate_books_controlled.py`` for v2 artifacts.
"""
from __future__ import annotations

from itertools import combinations

from tau_rec.catalog.taxonomy import COUNTRY_PARENTS_V2, LANGUAGE_PARENTS_V2
from tau_rec.data_model.task import RevealTag

OVERLAPPING_GENRES = {
    frozenset(pair) for pair in (
        ("Nonfiction", "Memoir"), ("Nonfiction", "Biography"), ("Nonfiction", "History"),
        ("Nonfiction", "Self-Help"), ("Biography", "Memoir"),
    )
}
NEIGHBOR_GENRES = {
    "History": {"Nonfiction"}, "Nonfiction": {"History", "Memoir", "Biography", "Self-Help"},
    "Memoir": {"Biography", "Nonfiction"}, "Biography": {"Memoir", "Nonfiction"},
    "Self-Help": {"Nonfiction"}, "Crime": {"Mystery", "Thriller"}, "Mystery": {"Crime", "Thriller"},
    "Thriller": {"Crime", "Mystery"}, "Children": {"Young Adult"}, "Young Adult": {"Children"},
}
GENRE_KEYWORDS = {
    "Adventure": ("adventure",), "Biography": ("biography",), "Children": ("children",),
    "Crime": ("crime",), "Fantasy": ("fantasy",), "Graphic Novel": ("graphic novel", "comic"),
    "Historical Fiction": ("historical novel", "historical fiction"), "History": ("history",),
    "Horror": ("horror",), "Humor": ("humor", "humour", "comic", "comedy", "satir"),
    "Memoir": ("memoir", "autobiograph"), "Mystery": ("mystery", "detective"),
    "Nonfiction": ("non-fiction", "nonfiction"), "Poetry": ("poem", "poetry", "verse"),
    "Romance": ("romance", "love story"), "Science Fiction": ("science fiction", "sci-fi"),
    "Self-Help": ("self-help",), "Thriller": ("thriller",), "Young Adult": ("young adult",),
}
NON_MODERN_STATES = {
    "Soviet Union", "Ottoman Empire", "Czechoslovakia", "Yugoslavia",
    "Socialist Federal Republic of Yugoslavia", "Democratic Federal Yugoslavia",
    "Kingdom of Serbs, Croats and Slovenes", "Austria–Hungary", "British Raj",
    "Holy Roman Empire", "Mughal Empire", "Dutch East Indies", "Republic of China",
}
MIN_HIDDEN_SELECTIVITY = 0.5


def v2_rule_errors(items, tasks, families, *, core_size: int) -> list[str]:
    """Return one message per violated v2 rule; empty means the artifact complies."""
    errors: list[str] = []
    by_id = {item.id: item for item in items}
    for family in families:
        family_id = family["family_id"]
        volunteer = tasks[family["task_ids"]["volunteer"]]
        hidden_task = tasks[family["task_ids"]["hidden"]]
        constraints = [tagged.constraint for tagged in volunteer.constraints]
        size = len(constraints)

        genres = [c.value for c in constraints if c.field == "genres" and c.op == "contains"]
        for pair in combinations(genres, 2):
            if frozenset(pair) in OVERLAPPING_GENRES:
                errors.append(f"{family_id}: overlapping genres {sorted(pair)}")
        for c in constraints:
            if c.field == "languages" and c.value in LANGUAGE_PARENTS_V2:
                errors.append(f"{family_id}: non-base language {c.value}")
            if c.field == "countries" and (c.value in COUNTRY_PARENTS_V2 or c.value in NON_MODERN_STATES):
                errors.append(f"{family_id}: non-modern country {c.value}")

        truth = [[c.evaluate(item) for c in constraints] for item in items]
        if family["no_valid_recommendation"]:
            for index, c in enumerate(constraints):
                for item, row in zip(items, truth):
                    if row[index] or not all(row[j] for j in range(size) if j != index):
                        continue
                    if c.field == "genres":
                        if set(item.genres) & NEIGHBOR_GENRES.get(c.value, set()):
                            errors.append(f"{family_id}: fragile NVR, {item.id} has a neighbor genre of {c.value}")
                        text = (item.description or "").casefold()
                        if any(keyword in text for keyword in GENRE_KEYWORDS.get(c.value, ())):
                            errors.append(f"{family_id}: fragile NVR, {item.id} description names {c.value}")
                    elif c.field == "countries" and not item.countries:
                        errors.append(f"{family_id}: fragile NVR, {item.id} has no country label")
            continue

        hidden = [i for i, tagged in enumerate(hidden_task.constraints) if tagged.reveal == RevealTag.HIDDEN]
        consistent = [row for row in truth if all(row[j] for j in range(size) if j not in hidden)]
        answers = sum(all(row) for row in consistent)
        selectivity = 1 - answers / len(consistent) if consistent else 0.0
        if selectivity < MIN_HIDDEN_SELECTIVITY:
            errors.append(
                f"{family_id}: hidden facts exclude only {selectivity:.2f} of revealed-consistent works"
            )
        if "history" in volunteer.policy_flags:
            read = [by_id[item_id] for item_id in volunteer.user_history["user_1"]["read"] if item_id in by_id]
            for h in hidden:
                if not any(
                    not constraints[h].evaluate(item)
                    and all(constraints[j].evaluate(item) for j in range(size) if j != h)
                    for item in read
                ):
                    errors.append(f"{family_id}: no decoy in the read list for hidden predicate {h}")
        if volunteer.user_formats:
            formats = set(volunteer.user_formats)
            if not any(
                all(row) and not formats & set(item.formats)
                for item, row in zip(items[:core_size], truth[:core_size])
            ):
                errors.append(f"{family_id}: no format trap in the {core_size:,}-work prefix")
    return errors
