from types import SimpleNamespace

import pytest

from tau_rec.catalog.books_tasks import (
    Predicate,
    _tag_constraints_matched,
    guarded_hidden_indices,
    implied_constraint_pairs,
    matched_reveal_order,
)
from tau_rec.data_model.task import Constraint, RevealTag


def _book(languages=(), countries=(), genres=("Fantasy",)):
    return SimpleNamespace(
        languages=list(languages), countries=list(countries), genres=list(genres)
    )


BRITISH = Constraint(field="languages", op="contains", value="British English")
UK = Constraint(field="countries", op="contains", value="United Kingdom")
ENGLISH = Constraint(field="languages", op="contains", value="English")
HORROR = Constraint(field="genres", op="contains", value="Horror")


def test_unguarded_hidden_arm_keeps_the_hash_schedule():
    predicates = tuple(
        Predicate("genres", "contains", genre, "genre", 0)
        for genre in ("Fantasy", "Horror", "Mystery", "Poetry", "Romance")
    )
    order = matched_reveal_order(5, family_id="family", seed=7)
    tags = [
        tagged.reveal
        for tagged in _tag_constraints_matched(predicates, "hidden", family_id="family", seed=7)
    ]
    assert tags[order[0]] == RevealTag.VOLUNTEER
    assert {i for i, tag in enumerate(tags) if tag == RevealTag.HIDDEN} == set(order[-2:])


def test_implication_conditions_on_items_carrying_the_field():
    items = [_book(["British English"], ["United Kingdom"]) for _ in range(6)]
    items += [_book(["British English"]) for _ in range(4)]
    items += [_book(["English"], ["United States"]) for _ in range(30)]
    pairs = implied_constraint_pairs([BRITISH, UK], items)
    assert (0, 1) in pairs
    assert pairs[(0, 1)]["probability"] == 1.0 and pairs[(0, 1)]["support"] == 6


def test_low_support_is_not_an_implication():
    items = [_book(["British English"], ["United Kingdom"]) for _ in range(3)]
    items += [_book(["English"], ["United States"]) for _ in range(30)]
    assert implied_constraint_pairs([BRITISH, UK], items) == {}


def test_base_rate_is_not_an_implication():
    items = [_book(["English"], genres=["Horror"]) for _ in range(9)]
    items += [_book(["French"], genres=["Horror"])]
    items += [_book(["English"]) for _ in range(81)] + [_book(["French"]) for _ in range(9)]
    assert (0, 1) not in implied_constraint_pairs([HORROR, ENGLISH], items)


def test_guard_keeps_the_schedule_without_leaks():
    assert guarded_hidden_indices([2, 0, 4, 1, 3], 2, {}) == [1, 3]


def test_guard_redraws_without_hiding_the_opening_fact():
    order = [0, 1, 2, 3, 4]
    assert guarded_hidden_indices(order, 2, {(1, 4): {}}) == [1, 4]
    assert guarded_hidden_indices(order, 2, {(0, 4): {}}) == [2, 3]


def test_guard_fails_loudly_when_no_assignment_avoids_a_leak():
    with pytest.raises(ValueError):
        guarded_hidden_indices([0, 1], 1, {(0, 1): {}})
