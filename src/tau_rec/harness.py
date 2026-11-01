"""Harness profiles: the run-time limits a track runs under.

"legacy" is how every published movie run executed, and movies keep it: no
tool-call budget, provider-default output limits, only litellm's own request
timeout, a trial cap on total wall time, and every early stop recorded as
"timeout". A track declares its profile in track.json; the run manifest
records it, and a resumed run keeps the profile it started with.

"books-v4" is the profile the books study ran under:

- 150 tool calls per agent step. A call past the budget is not executed; it
  returns an error asking the agent to answer the user or recommend, and the
  trial ends after ten such errors in one step.
- 32,768 output tokens per agent call.
- Each model call is cancelled after 300 s and retried.
- The trial cap counts only the agent's own time, not the user simulator's.
- Early stops are recorded as wall_cap, turn_limit, context_overflow or
  tool_budget instead of one "timeout".
- An empty agent reply is resampled up to three times, and each call's
  finish_reason is logged.
- Text the agent sends alongside tool calls is logged in the trace. It is not
  shown to the user.
"""
from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Awaitable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarnessProfile:
    id: str
    max_tool_calls_per_step: int | None = None
    budget_errors_before_stop: int = 10
    agent_max_tokens: int | None = None
    call_timeout_seconds: float | None = None
    cap_counts_agent_time_only: bool = False
    detailed_stop_reasons: bool = False
    empty_reply_resamples: int = 0
    log_finish_reason: bool = False
    log_tool_call_text: bool = False

    def manifest(self) -> dict:
        return asdict(self)


LEGACY = HarnessProfile(id="legacy")
BOOKS_V4 = HarnessProfile(
    id="books-v4",
    max_tool_calls_per_step=150,
    agent_max_tokens=32_768,
    call_timeout_seconds=300.0,
    cap_counts_agent_time_only=True,
    detailed_stop_reasons=True,
    empty_reply_resamples=3,
    log_finish_reason=True,
    log_tool_call_text=True,
)
HARNESS_PROFILES = {profile.id: profile for profile in (LEGACY, BOOKS_V4)}


def get_harness_profile(profile_id: str) -> HarnessProfile:
    try:
        return HARNESS_PROFILES[profile_id]
    except KeyError:
        raise ValueError(
            f"unknown harness profile {profile_id!r}; known: {sorted(HARNESS_PROFILES)}"
        ) from None


class CallTimeout(Exception):
    """A model call outlived the profile's call timeout; callers retry it."""


async def timed_call(call: Awaitable, timeout_seconds: float | None):
    """Await a model call, cancelling it after ``timeout_seconds`` if set."""
    if timeout_seconds is None:
        return await call
    try:
        return await asyncio.wait_for(call, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise CallTimeout(f"no response in {timeout_seconds:g} s") from None


def harness_commit() -> dict | None:
    """The git commit the harness runs from, and whether tracked files differ from it."""
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"sha": sha, "dirty": bool(changes)}
