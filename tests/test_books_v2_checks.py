"""Independent v2 rule checks (used by scripts/validate_books_controlled.py)."""
from tau_rec.catalog.books_v2_checks import v2_rule_errors
from tau_rec.data_model.catalog import Book
from tau_rec.data_model.task import Constraint, RevealTag, Task, TaskConstraint


def _book(index: int, **fields) -> Book:
    base = {
        "id": f"wd_{index:03d}", "title": f"Work {index}", "authors": ["A"], "publication_year": 2000,
        "pages": None, "genres": ["Mystery"], "description": "a novel", "publisher": "P",
        "rating": None, "vote_count": None, "formats": ["ebook"], "languages": ["English"],
        "countries": ["United States"], "sitelink_count": 1,
    }
    base.update(fields)
    return Book(**base)


def _task(task_id, constraints, *, reveal, nvr=False, read=(), formats=(), flags=("single_recommendation",)):
    return Task(
        id=task_id, persona="p", complexity="simple", reveal_difficulty=reveal,
        no_valid_recommendation=nvr, policy_flags=list(flags), user_formats=list(formats),
        user_history={"user_1": {"read": list(read), "ratings": {}}}, user_id="user_1",
        constraints=[TaskConstraint(constraint=Constraint(field=f, op=o, value=v), reveal=t) for f, o, v, t in constraints],
    )


def _family(tasks):
    return {"family_id": "books_simple_001", "no_valid_recommendation": tasks[0].no_valid_recommendation,
            "task_ids": {"volunteer": tasks[0].id, "mixed": tasks[1].id, "hidden": tasks[2].id}}


def _variants(constraints, nvr=False, read=(), formats=(), flags=("single_recommendation",)):
    tags = {
        "volunteer": [RevealTag.VOLUNTEER] * len(constraints),
        "mixed": [RevealTag.VOLUNTEER] + [RevealTag.ON_ASK] * (len(constraints) - 1),
        "hidden": [RevealTag.VOLUNTEER] + [RevealTag.HIDDEN] * (len(constraints) - 1),
    }
    return [
        _task(f"task_001_{reveal[0]}", [(*c, tag) for c, tag in zip(constraints, tags[reveal])],
              reveal=reveal, nvr=nvr, read=read, formats=formats, flags=flags)
        for reveal in ("volunteer", "mixed", "hidden")
    ]


def test_overlapping_genres_and_non_base_values_are_errors():
    items = [_book(i, genres=["Nonfiction", "Memoir"]) for i in range(10)]
    tasks = _variants([("genres", "contains", "Nonfiction"), ("genres", "contains", "Memoir"),
                       ("languages", "contains", "British English")])
    errors = v2_rule_errors(items, {t.id: t for t in tasks}, [_family(tasks)], core_size=10)
    assert any("overlapping genres" in e for e in errors)
    assert any("non-base language" in e for e in errors)


def test_fragile_nvr_and_unselective_hidden_facts_are_errors():
    items = [_book(0, genres=["Nonfiction"], languages=["French"], description="a graphic novel")] + [
        _book(1 + i, genres=["Graphic Novel"]) for i in range(5)] + [_book(10 + i, languages=["French"]) for i in range(5)]
    nvr = _variants([("genres", "contains", "Graphic Novel"), ("languages", "contains", "French")], nvr=True)
    errors = v2_rule_errors(items, {t.id: t for t in nvr}, [_family(nvr)], core_size=11)
    assert any("fragile NVR" in e for e in errors)

    items = [_book(i, countries=["United States"]) for i in range(20)]
    weak = _variants([("genres", "contains", "Mystery"), ("countries", "contains", "United States")])
    errors = v2_rule_errors(items, {t.id: t for t in weak}, [_family(weak)], core_size=20)
    assert any("hidden facts exclude" in e for e in errors)


def test_history_needs_a_decoy_per_hidden_fact_and_formats_need_a_5k_trap():
    items = ([_book(i, countries=["Canada"], publication_year=2010) for i in range(6)]
             + [_book(10 + i, countries=["United States"], publication_year=2010) for i in range(20)])
    constraints = [("genres", "contains", "Mystery"), ("countries", "contains", "Canada")]
    no_decoy = _variants(constraints, read=["wd_000"], flags=("history",))
    errors = v2_rule_errors(items, {t.id: t for t in no_decoy}, [_family(no_decoy)], core_size=26)
    assert any("no decoy" in e for e in errors)
    decoy = _variants(constraints, read=["wd_000", "wd_010"], flags=("history",))
    assert not any("no decoy" in e for e in v2_rule_errors(items, {t.id: t for t in decoy}, [_family(decoy)], core_size=26))
    every_answer_has_ebook = _variants(constraints, formats=["ebook"], flags=("format_verification",))
    errors = v2_rule_errors(items, {t.id: t for t in every_answer_has_ebook}, [_family(every_answer_has_ebook)], core_size=26)
    assert any("format trap" in e for e in errors)
