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
    assert "non-standard simulator" in out
    assert out.index("non-standard simulator") < out.index("Odd")


def test_render_omits_nonstandard_section_when_all_standard(tmp_path):
    _write_entry(tmp_path, submission_id="std")
    assert "non-standard simulator" not in render_mod.render(tmp_path)


def test_render_ranks_generations_in_separate_tables(tmp_path):
    """A g0 row must never share a ranking with a g1 row."""
    _write_entry(tmp_path, submission_id="old", display_name="OldGen",
                 harness_generation="g0",
                 per_task={"t1": {"n": 4, "c": 4}, "t2": {"n": 4, "c": 4}})
    _write_entry(tmp_path, submission_id="new", display_name="NewGen",
                 harness_generation="g1",
                 per_task={"t1": {"n": 4, "c": 0}, "t2": {"n": 4, "c": 0}})
    out = render_mod.render(tmp_path)
    # Newest generation leads, even though its scores are worse.
    assert out.index("Generation `g1`") < out.index("Generation `g0`")
    assert out.index("NewGen") < out.index("OldGen")
    assert "most recent" in out
    assert "cannot be compared across tables" in out


def test_render_omits_cross_generation_warning_for_single_generation(tmp_path):
    _write_entry(tmp_path, submission_id="only", harness_generation="g0")
    out = render_mod.render(tmp_path)
    assert "cannot be compared across tables" not in out
    # A lone generation must not be labelled the newest — it may predate HEAD.
    assert "most recent" not in out


def test_generation_sort_is_numeric_not_lexicographic(tmp_path):
    assert render_mod._generation_sort_key("g10") < render_mod._generation_sort_key("g9")


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


# --- make-entry reads the run manifest -------------------------------------
#
# Nothing in traces/ or task_results.json records reasoning_effort, so a run
# without a manifest cannot be labelled after the fact. These pin the two rules
# that make a manifest worth having: it beats the CLI flag, and it refuses to
# certify a run whose content has since changed.

import json
import shutil

from click.testing import CliRunner

from tau_rec.cli import main as cli
from tau_rec.leaderboard.hashing import hash_file as _hash_file
from tau_rec.leaderboard.hashing import hash_task_dir as _hash_task_dir

FIXTURE_TRACE = Path(__file__).parent / "fixtures" / "sample_trace.json"


def _run_dir(tmp_path, **manifest_overrides):
    run = tmp_path / "run" / "20260910_000000"
    (run / "traces").mkdir(parents=True)
    shutil.copy(FIXTURE_TRACE, run / "traces" / "task_001_trial1.json")
    manifest = {
        "model": json.loads(FIXTURE_TRACE.read_text())["model"],
        "reasoning_effort": "low",
        "simulator_model": "gpt-5-mini",
        "policy_sha256": _hash_file("data/policy.md"),
        "tasks_sha256": _hash_task_dir("data/tasks"),
        "catalog_sha256": _hash_file("data/catalog.json"),
    }
    manifest.update(manifest_overrides)
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    return run.parent


def _make_entry(tmp_path, run_root, *extra):
    entries = tmp_path / "entries"
    entries.mkdir(exist_ok=True)
    result = CliRunner().invoke(cli, [
        "leaderboard", "make-entry",
        "--run-dir", str(run_root),
        "--submission-id", "t1", "--display-name", "T", "--submitted-by", "T",
        "--run-date", "2026-09-10", "--entries", str(entries), *extra,
    ])
    return result, entries / "t1.json"


def test_make_entry_prefers_manifest_effort_over_flag(tmp_path):
    result, path = _make_entry(
        tmp_path, _run_dir(tmp_path), "--reasoning-effort", "xhigh"
    )
    assert result.exit_code == 0, result.output
    assert json.loads(path.read_text())["reasoning_effort"] == "low"


def test_make_entry_records_null_effort_from_manifest(tmp_path):
    """A manifest saying 'no effort set' must override a flag, not be treated
    as absent — otherwise the fallback silently reinstates the typed value."""
    run = _run_dir(tmp_path, reasoning_effort=None)
    result, path = _make_entry(tmp_path, run, "--reasoning-effort", "high")
    assert result.exit_code == 0, result.output
    assert json.loads(path.read_text())["reasoning_effort"] is None


def test_make_entry_rejects_run_whose_tasks_changed(tmp_path):
    run = _run_dir(tmp_path, tasks_sha256="0" * 64)
    result, _ = _make_entry(tmp_path, run)
    assert result.exit_code != 0
    assert "tasks changed since this run" in result.output


def test_make_entry_falls_back_to_flags_without_manifest(tmp_path):
    run = _run_dir(tmp_path)
    (run / "20260910_000000" / "run_manifest.json").unlink()
    result, path = _make_entry(tmp_path, run, "--reasoning-effort", "high")
    assert result.exit_code == 0, result.output
    assert "no run_manifest.json" in result.output
    assert json.loads(path.read_text())["reasoning_effort"] == "high"
