"""Description-year disagreements (scripts/report_description_years.py)."""
from tau_rec.catalog.description_years import description_year, year_mismatches
from tau_rec.data_model.catalog import Book
from tau_rec.data_model.task import Constraint, RevealTag, Task, TaskConstraint


def _book(index: int, year: int, description: str) -> Book:
    return Book(
        id=f"wd_{index:03d}", title=f"Work {index}", authors=["A"], publication_year=year, pages=None,
        genres=["Science Fiction"], description=description, publisher="P", rating=None, vote_count=None,
        formats=["ebook"], languages=["English"], countries=["United States"], sitelink_count=1,
    )


def _task(task_id: str, year_lower: int) -> Task:
    return Task(
        id=task_id, persona="p", complexity="simple", reveal_difficulty="volunteer",
        no_valid_recommendation=False, policy_flags=["single_recommendation"], user_formats=[],
        user_history={"user_1": {"read": [], "ratings": {}}}, user_id="user_1",
        constraints=[
            TaskConstraint(constraint=Constraint(field="genres", op="contains", value="Science Fiction"),
                           reveal=RevealTag.VOLUNTEER),
            TaskConstraint(constraint=Constraint(field="publication_year", op=">=", value=year_lower),
                           reveal=RevealTag.VOLUNTEER),
        ],
    )


def test_description_year_parsing():
    assert description_year("2013 novel by Tui T. Sutherland") == 2013
    assert description_year("1994 children's fantasy book") == 1994
    assert description_year("2008 crime novel by Håkan Nesser") == 2008
    assert description_year("199 biography by Carlo Feltrinelli") is None  # typo, not a year
    assert description_year("novel by Charles Bukowski") is None
    assert description_year(None) is None


def test_mismatch_reports_family_changes():
    items = [
        _book(1, 1999, "2019 novel by Sue Burke"),     # Interference: description year joins the family
        _book(2, 2018, "2018 novel"),                  # agrees: not reported
        _book(3, 2016, "2017 novel by Katharine McGee"),  # disagrees, but both years are >= 2010: no change
    ]
    tasks = {"task_001_v": _task("task_001_v", 2010)}
    families = [{"family_id": "books_simple_001", "task_ids": {"volunteer": "task_001_v"}}]
    rows = {row.item_id: row for row in year_mismatches(items, tasks, families)}
    assert set(rows) == {"wd_001", "wd_003"}
    assert rows["wd_001"].family_changes == {"books_simple_001": "+"}
    assert rows["wd_003"].family_changes == {}
