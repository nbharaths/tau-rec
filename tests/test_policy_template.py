from pathlib import Path

from tau_rec.domain import MOVIES
from tau_rec.policy_template import render_memory_policy, render_policy

ROOT = Path(__file__).resolve().parent.parent


def test_movies_rendering_is_the_checked_in_policy_byte_for_byte():
    """data/policy.md is scored content whose digest sits on every g1 entry."""
    assert render_policy(MOVIES) == (ROOT / "data" / "policy.md").read_text()
    assert render_memory_policy(MOVIES) == (ROOT / "data" / "policy_memory.md").read_text()
