import pytest

from pathlib import Path
from tau_rec.leaderboard.hashing import hash_file, hash_task_dir, file_manifest


def test_hash_task_dir_is_stable():
    assert hash_task_dir("data/tasks") == hash_task_dir("data/tasks")


def test_hash_task_dir_differs_from_concatenation():
    """Pin the canonical convention: manifest-of-digests, not byte concatenation."""
    import hashlib
    from pathlib import Path
    concat = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path("data/tasks").glob("*.json")))
    ).hexdigest()
    assert hash_task_dir("data/tasks") != concat


def test_manifest_has_one_line_per_task():
    lines = file_manifest("data/tasks").strip().split("\n")
    assert len(lines) == 60
    assert all(":" in line for line in lines)
    assert lines == sorted(lines)


def test_hash_task_dir_detects_content_change(tmp_path):
    (tmp_path / "a.json").write_text('{"x": 1}')
    before = hash_task_dir(tmp_path)
    (tmp_path / "a.json").write_text('{"x": 2}')
    assert hash_task_dir(tmp_path) != before


def test_hash_task_dir_detects_rename(tmp_path):
    (tmp_path / "a.json").write_text('{"x": 1}')
    before = hash_task_dir(tmp_path)
    (tmp_path / "a.json").rename(tmp_path / "b.json")
    assert hash_task_dir(tmp_path) != before


def test_hash_task_dir_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        hash_task_dir(tmp_path)


def test_hash_file_matches_known_digest(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    assert hash_file(p) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


# --- entry model ---
from pydantic import ValidationError
from tau_rec.leaderboard.entry import LeaderboardEntry, TaskCount


def _entry(**overrides) -> LeaderboardEntry:
    base = dict(
        submission_id="g0-example", display_name="Example", submitted_by="tester",
        model_id="deepseek/deepseek-v4-flash", pinned=False, run_date="2026-05-02",
        simulator_model="gpt-5-mini", harness_generation="g0",
        tau_rec_version="0.1.0", trials_per_task=4,
        policy_sha256="a" * 64, tasks_sha256="b" * 64, catalog_sha256="c" * 64,
        per_task={"task_001": {"n": 4, "c": 4}, "task_002": {"n": 4, "c": 2}},
    )
    base.update(overrides)
    return LeaderboardEntry(**base)


def test_entry_roundtrip():
    e = _entry()
    assert LeaderboardEntry.model_validate_json(e.model_dump_json()) == e
    assert e.n_tasks == 2


def test_entry_stores_no_aggregates():
    fields = set(LeaderboardEntry.model_fields)
    assert not fields & {"pass_1", "pass_2", "pass_4", "primary_reward", "score"}


def test_task_count_rejects_c_over_n():
    with pytest.raises(ValidationError):
        TaskCount(n=4, c=5)


def test_entry_rejects_floating_model_tag():
    with pytest.raises(ValidationError):
        _entry(model_id="anthropic/claude-sonnet-latest")


def test_entry_allows_unpinned_alias_with_run_date():
    e = _entry(model_id="gpt-5-mini", pinned=False, run_date="2026-05-02")
    assert e.pinned is False and e.run_date


def test_entry_auto_flags_same_model_as_simulator():
    e = _entry(model_id="gpt-5-mini", simulator_model="gpt-5-mini")
    assert "same_model_as_simulator" in e.caveats


def test_entry_task_results_shape_feeds_aggregate_pass_k():
    from tau_rec.metrics.pass_k import aggregate_pass_k
    e = _entry()
    assert aggregate_pass_k(e.task_results, 1) == pytest.approx(0.75)


# --- renderer ---
from tau_rec.leaderboard import render as render_mod


def _write_entry(d, **kw):
    e = _entry(**kw)
    (d / f"{e.submission_id}.json").write_text(e.model_dump_json())
    return e


def test_render_is_deterministic(tmp_path):
    _write_entry(tmp_path, submission_id="a", per_task={f"t{i}": {"n": 4, "c": i % 5} for i in range(12)})
    assert render_mod.render(tmp_path) == render_mod.render(tmp_path)


def test_render_sorts_best_first(tmp_path):
    _write_entry(tmp_path, submission_id="weak", display_name="Weak",
                 per_task={"t1": {"n": 4, "c": 0}, "t2": {"n": 4, "c": 0}})
    _write_entry(tmp_path, submission_id="strong", display_name="Strong",
                 per_task={"t1": {"n": 4, "c": 4}, "t2": {"n": 4, "c": 4}})
    out = render_mod.render(tmp_path)
    assert out.index("Strong") < out.index("Weak")


def test_render_separates_nonstandard_simulator(tmp_path):
    _write_entry(tmp_path, submission_id="std", display_name="Std")
    _write_entry(tmp_path, submission_id="odd", display_name="Odd",
                 simulator_model="some/other-sim")
    out = render_mod.render(tmp_path)
    assert "## Non-standard simulator" in out
    assert out.index("## Non-standard simulator") < out.index("Odd")


def test_render_omits_nonstandard_section_when_all_standard(tmp_path):
    _write_entry(tmp_path, submission_id="std")
    assert "## Non-standard simulator" not in render_mod.render(tmp_path)


def test_render_check_detects_staleness(tmp_path):
    _write_entry(tmp_path, submission_id="a")
    out = tmp_path / "BOARD.md"
    render_mod.write(tmp_path, out)
    assert render_mod.check(tmp_path, out)
    out.write_text("tampered\n")
    assert not render_mod.check(tmp_path, out)


def test_render_matches_paper_for_seed_cohort():
    """The DS V4 Flash row must reproduce Table 3: 0.546 / 0.433 / 0.333."""
    entries = Path("leaderboard/entries")
    if not entries.is_dir():
        pytest.skip("seed entries not present")
    e = next(x for x in render_mod.load_entries(entries) if x.submission_id == "g0-dsv4-flash")
    s = render_mod.summarize(e)
    assert (round(s["pass_1"], 3), round(s["pass_2"], 3), round(s["pass_4"], 3)) == (0.546, 0.433, 0.333)
