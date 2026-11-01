"""Opt-in agent prompt variants for exploratory runs; the default prompt is unchanged.

No test here can reach a provider: the CLI tests replace _run_benchmark, and the
manifest test runs zero trials with every provider entry point patched to fail.
"""
import asyncio
import json
from pathlib import Path

import litellm
import pytest
from click.testing import CliRunner

from tau_rec import cli as cli_module
from tau_rec.agents import litellm_agent
from tau_rec.agents.litellm_agent import (
    AGENT_PROMPT_VARIANTS, BOOKS_CANDIDATE_PROTOCOL, KEYWORD_TOOL_VARIANTS, build_agent_system_prompt,
)
from tau_rec.domain import BOOKS, MOVIES
from tau_rec.track import load_track

ROOT = Path(__file__).resolve().parent.parent
TRACK = ROOT / "data/tracks/books-wikidata-v2"
POLICY = (TRACK / "policy.md").read_text()


def _books_prompt(**kwargs) -> str:
    return build_agent_system_prompt(
        POLICY, "user_1", spec=BOOKS, retrieval_mode="structured",
        structured_profile="books-wikidata-core-v3", **kwargs,
    )


def test_variant_continues_the_elicitation_paragraph_and_changes_nothing_else():
    default = _books_prompt()
    variant = _books_prompt(prompt_variant="ask-requirements-v1")
    sentence = AGENT_PROMPT_VARIANTS["ask-requirements-v1"].format(item_noun="book")
    assert "not stated yet" not in default
    assert variant == default.replace(BOOKS_CANDIDATE_PROTOCOL, sentence + BOOKS_CANDIDATE_PROTOCOL)
    assert "making a recommendation. The user may have requirements" in variant


def test_unknown_or_unsupported_variants_are_refused():
    with pytest.raises(ValueError):
        _books_prompt(prompt_variant="no-such-variant")
    with pytest.raises(ValueError):
        build_agent_system_prompt(POLICY, None, spec=BOOKS, mode="memory", prompt_variant="ask-requirements-v1")


def _keyword_prompt(**kwargs) -> str:
    return build_agent_system_prompt(
        POLICY, "user_1", spec=BOOKS, retrieval_mode="keyword",
        structured_profile="books-wikidata-core-v3", **kwargs,
    )


def test_keyword_field_list_variant_names_the_structured_fields_in_the_keyword_tool_line():
    default = _keyword_prompt()
    variant = _keyword_prompt(prompt_variant="keyword-field-list-v1")
    tool_line = "search_catalog(query): Search the book catalog by keywords."
    addition = KEYWORD_TOOL_VARIANTS["keyword-field-list-v1"]
    assert tool_line in default and addition not in default
    assert variant == default.replace(tool_line, tool_line + addition)
    for field in ("genres", "languages", "countries", "publication year", "formats"):
        assert field in addition


def test_keyword_field_list_variant_needs_the_books_keyword_arm():
    with pytest.raises(ValueError):
        _books_prompt(prompt_variant="keyword-field-list-v1")
    with pytest.raises(ValueError):
        build_agent_system_prompt(POLICY, None, spec=MOVIES, prompt_variant="keyword-field-list-v1")


def test_cli_offers_exactly_the_defined_variants():
    option = next(param for param in cli_module.run.params if param.name == "agent_prompt_variant")
    assert sorted(option.type.choices) == sorted([*AGENT_PROMPT_VARIANTS, *KEYWORD_TOOL_VARIANTS])


def test_cli_refuses_the_keyword_variant_on_the_structured_arm(monkeypatch, tmp_path):
    seen = []

    async def fake_run_benchmark(*args, **kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(cli_module, "_run_benchmark", fake_run_benchmark)
    runner = CliRunner()
    base = ["run", "--track", str(TRACK), "--model", "m", "--simulator-mode", "symbolic",
            "--agent-prompt-variant", "keyword-field-list-v1"]
    result = runner.invoke(cli_module.main, [*base, "--output", str(tmp_path / "a"), "--retrieval-mode", "structured"])
    assert result.exit_code != 0 and "keyword-field-list-v1 requires --retrieval-mode keyword" in result.output
    result = runner.invoke(cli_module.main, [*base, "--output", str(tmp_path / "b"), "--retrieval-mode", "keyword"])
    assert result.exit_code == 0, result.output
    assert seen[-1]["agent_prompt_variant"] == "keyword-field-list-v1"


def test_new_runs_pass_the_variant_and_resumes_take_it_from_the_manifest(monkeypatch, tmp_path):
    seen = []

    async def fake_run_benchmark(*args, **kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(cli_module, "_run_benchmark", fake_run_benchmark)
    runner = CliRunner()
    result = runner.invoke(cli_module.main, [
        "run", "--track", str(TRACK), "--model", "m", "--output", str(tmp_path / "out"),
        "--simulator-mode", "symbolic", "--agent-prompt-variant", "ask-requirements-v1",
    ])
    assert result.exit_code == 0, result.output
    assert seen[-1]["agent_prompt_variant"] == "ask-requirements-v1"

    run = tmp_path / "run" / "20261016_000000"
    (run / "traces").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({
        "model": "m", "trials_per_task": 1, "mode": "tools", "track": "books-wikidata-v2",
        "search_profile": "bm25f-v1", "agent_prompt_variant": "ask-requirements-v1",
    }))
    result = runner.invoke(cli_module.main, ["run", "--resume", str(run)])
    assert result.exit_code == 0, result.output
    assert seen[-1]["agent_prompt_variant"] == "ask-requirements-v1"

    result = runner.invoke(cli_module.main, [
        "run", "--track", str(ROOT / "data"), "--model", "m", "--output", str(tmp_path / "memory"),
        "--mode", "memory", "--agent-prompt-variant", "ask-requirements-v1",
    ])
    assert result.exit_code != 0 and "--agent-prompt-variant requires" in result.output


def test_manifest_records_the_variant_only_when_set(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise AssertionError("no provider call or agent may be created in this test")

    monkeypatch.setattr(litellm, "acompletion", refuse)
    monkeypatch.setattr(litellm, "aresponses", refuse)
    monkeypatch.setattr(litellm_agent, "LiteLLMAgent", refuse)
    exits = []
    monkeypatch.setattr("os._exit", exits.append)
    track = load_track(str(TRACK))
    for variant, out in (("ask-requirements-v1", tmp_path / "variant"), (None, tmp_path / "default")):
        asyncio.run(cli_module._run_benchmark(
            "m", str(track.require_catalog()), str(track.tasks_path), str(track.policy_path), 1, str(out),
            "unused", 20, 1, tasks_filter="no_such_task", track=track,
            simulator_interaction_mode="symbolic", retrieval_mode="structured", agent_prompt_variant=variant,
        ))
    variant_manifest = json.loads(next((tmp_path / "variant").glob("*/run_manifest.json")).read_text())
    default_manifest = json.loads(next((tmp_path / "default").glob("*/run_manifest.json")).read_text())
    assert variant_manifest["agent_prompt_variant"] == "ask-requirements-v1"
    assert "agent_prompt_variant" not in default_manifest
    assert exits == [0, 0]


def test_ask_and_verify_variant_adds_the_verify_sentence_after_the_ask_sentence():
    default = _books_prompt()
    variant = _books_prompt(prompt_variant="ask-and-verify-v1")
    ask = AGENT_PROMPT_VARIANTS["ask-requirements-v1"].format(item_noun="book")
    both = AGENT_PROMPT_VARIANTS["ask-and-verify-v1"].format(item_noun="book")
    assert both.startswith(ask) and "verdict" in both[len(ask):]
    assert variant == default.replace(BOOKS_CANDIDATE_PROTOCOL, both + BOOKS_CANDIDATE_PROTOCOL)


def test_protocol_variant_explains_declines_and_rejections_after_ask_and_verify():
    default = _books_prompt()
    variant = _books_prompt(prompt_variant="ask-verify-protocol-v1")
    verify = AGENT_PROMPT_VARIANTS["ask-and-verify-v1"].format(item_noun="book")
    protocol = AGENT_PROMPT_VARIANTS["ask-verify-protocol-v1"].format(item_noun="book")
    added = protocol[len(verify):]
    # Only the protocol's meaning is added: a declined field holds a requirement, a rejection fails one.
    assert protocol.startswith(verify)
    assert "rather not say" in added and "requirement on that field" in added
    assert "rejected book fails at least one" in added
    assert variant == default.replace(BOOKS_CANDIDATE_PROTOCOL, protocol + BOOKS_CANDIDATE_PROTOCOL)
