"""The books release trace packer (scripts/pack_books_release.py)."""
import importlib.util
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKER = ROOT / "scripts/pack_books_release.py"


def _packer():
    spec = importlib.util.spec_from_file_location("pack_books_release", PACKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_root(base: Path) -> Path:
    root = base / "out" / "books-run"
    run_dir = root / "reliability" / "agent" / "structured" / "20000" / "20261014_000000"
    (run_dir / "traces").mkdir(parents=True)
    (run_dir / "quarantine").mkdir()
    (run_dir / "traces" / "task_001_v_trial0.json").write_text(json.dumps({
        "task_id": "task_001_v",
        "api_calls": {"agent": [
            {"error": 'Timeout - {"error":{"code":504},"user_id":"org_ExampleAccount0000000000000"}'},
            {"error": "APIError - 408, message='Request Timeout', url='http://localhost:8080'"},
        ]},
    }))
    (run_dir / "quarantine" / "task_001_h_trial0.json").write_text('{"task_id": "task_001_h"}')
    (run_dir / "run_manifest.json").write_text("{}")
    rel = run_dir.relative_to(base).as_posix()
    (root / "reliability_v4_state.json").write_text(json.dumps({"cells": [{"run_dir": rel}]}))
    smoke = root / "smoke" / "agent"
    smoke.mkdir(parents=True)
    (smoke / "trace.json").write_text("{}")
    (root / "smoke_v4_state.json").write_text(json.dumps({"cells": [{"run_dir": smoke.relative_to(base).as_posix()}]}))
    (root / "full_launcher.log").write_text("ok\n/Users/someone/tau-rec/.venv/lib/x.py:1: Warning\n")
    (root / "smoke_launcher.log").write_text("smoke\n")
    return root


def _members(archive: Path) -> dict[str, bytes]:
    with tarfile.open(archive) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def test_packs_cells_states_and_logs_but_not_smoke(tmp_path):
    packer = _packer()
    root = _fake_root(tmp_path)
    archive = packer.pack(root, tmp_path / "assets", base=tmp_path)
    members = _members(archive)
    prefix = "out/books-run/reliability/agent/structured/20000/20261014_000000/"
    assert prefix + "traces/task_001_v_trial0.json" in members
    assert prefix + "quarantine/task_001_h_trial0.json" in members
    assert prefix + "run_manifest.json" in members
    assert "out/books-run/reliability_v4_state.json" in members
    assert "out/books-run/full_launcher.log" in members
    assert not any("smoke" in name for name in members)


def test_logs_lose_local_paths(tmp_path):
    packer = _packer()
    archive = packer.pack(_fake_root(tmp_path), tmp_path / "assets", base=tmp_path)
    log = _members(archive)["out/books-run/full_launcher.log"].decode()
    assert "/Users/" not in log
    assert "tau-rec/.venv/lib/x.py" in log


def test_traces_lose_account_ids_and_local_proxy_addresses(tmp_path):
    packer = _packer()
    archive = packer.pack(_fake_root(tmp_path), tmp_path / "assets", base=tmp_path)
    member = "out/books-run/reliability/agent/structured/20000/20261014_000000/traces/task_001_v_trial0.json"
    trace = json.loads(_members(archive)[member])
    errors = [call["error"] for call in trace["api_calls"]["agent"]]
    assert "org_Example" not in errors[0] and '"user_id":"<redacted>"' in errors[0]
    assert "8080" not in errors[1] and "url='<redacted>'" in errors[1]
    assert trace["task_id"] == "task_001_v"


def test_archives_are_byte_reproducible_and_carry_no_local_metadata(tmp_path):
    packer = _packer()
    root = _fake_root(tmp_path)
    first = packer.pack(root, tmp_path / "a", base=tmp_path).read_bytes()
    second = packer.pack(root, tmp_path / "b", base=tmp_path).read_bytes()
    assert first == second
    with tarfile.open(fileobj=io.BytesIO(first)) as tar:
        for member in tar.getmembers():
            assert not Path(member.name).name.startswith("._")
            assert (member.uid, member.gid, member.uname, member.gname, member.mtime) == (0, 0, "", "", 0)
