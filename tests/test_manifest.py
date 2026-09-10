import json

from tau_rec.data_model.manifest import RunManifest


def _manifest(**overrides) -> RunManifest:
    base = dict(
        model="deepseek/deepseek-v4-flash",
        simulator_model="gpt-5-mini",
        trials=4,
        max_turns=20,
        concurrency=16,
        no_tools=False,
        reasoning_effort=None,
        tasks_limit=None,
        tau_rec_version="0.1.0",
        harness_generation="g1",
        policy_sha256="a" * 64,
        tasks_sha256="b" * 64,
        catalog_sha256="c" * 64,
        run_started_at="2026-09-10T12:00:00+00:00",
        n_tasks=60,
    )
    base.update(overrides)
    return RunManifest(**base)


def test_manifest_roundtrip():
    m = _manifest()
    assert RunManifest.model_validate_json(m.model_dump_json()) == m


def test_manifest_optional_fields_default_to_none():
    m = _manifest()
    assert m.run_finished_at is None
    assert m.wall_clock_s is None
    assert m.n_trials_completed is None
    assert m.token_usage == {}


def test_manifest_records_thinking_mode():
    """The three DeepSeek thinking modes share one model string; only
    reasoning_effort distinguishes them, so it must survive serialization."""
    m = _manifest(reasoning_effort="xhigh")
    assert json.loads(m.model_dump_json())["reasoning_effort"] == "xhigh"


def test_manifest_rejects_bad_digest():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _manifest(policy_sha256="not-a-sha")
