"""Redact catalog data from traces before publishing them.

A published trace carries every `get_metadata` result and every
`search_catalog` result list. Across a few thousand traces that reconstructs
the catalog, so a track whose source data may not be redistributed should
publish redacted traces.

The redaction replaces those two tools' `result` strings with a digest and
keeps everything else. `get_metadata` also preserves one boolean saying
whether the call succeeded because the books format-verification policy uses
that fact. No source metadata is retained. Tests cover score equivalence for
both legacy archives and the books policy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REDACTED_TOOLS = ("search_catalog", "get_metadata")


def _redacted_result(tool_name: str, result: str) -> str:
    payload = {
        "redacted": True,
        "sha256": hashlib.sha256(result.encode("utf-8")).hexdigest(),
        "length": len(result),
    }
    if tool_name == "get_metadata":
        try:
            original = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            original = {"error": "invalid"}
        payload["ok"] = isinstance(original, dict) and "error" not in original
    return json.dumps(payload)


def redact_trace(doc: dict) -> dict:
    """A copy of a trace document with catalog tool results replaced by digests."""
    out = json.loads(json.dumps(doc))
    for ev in out.get("events", []):
        if ev.get("type") == "tool_call" and ev.get("name") in REDACTED_TOOLS:
            result = ev.get("result") or ""
            ev["result"] = _redacted_result(ev["name"], result)
    return out


def redact_dir(src: Path, dst: Path) -> int:
    """Redact every trace under src/ into dst/, keeping file names. Returns the count."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(src.glob("*.json")):
        doc = json.loads(path.read_text())
        (dst / path.name).write_text(json.dumps(redact_trace(doc), indent=2, ensure_ascii=False) + "\n")
        n += 1
    return n
