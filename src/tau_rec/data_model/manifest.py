from __future__ import annotations
from typing import Annotated
from pydantic import BaseModel, Field

# A hex sha256 digest. Validated so a truncated or placeholder hash cannot
# silently enter a leaderboard entry and defeat the comparability check.
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RunManifest(BaseModel):
    """Harness metadata for one `tau-rec run`, written to run_manifest.json.

    Run directories used to contain only task_results.json and
    trial_results.json, and traces carry no usage fields, so nothing recorded
    which simulator, policy, or thinking mode produced a set of numbers. The
    three DeepSeek thinking modes in particular all report an identical model
    string and were only distinguishable by which directory they landed in.
    """

    # What was run
    model: str
    simulator_model: str
    reasoning_effort: str | None = None
    no_tools: bool = False

    # How it was run
    trials: int
    max_turns: int
    concurrency: int
    tasks_limit: int | None = None
    n_tasks: int

    # What it was run against
    tau_rec_version: str
    harness_generation: str
    policy_sha256: Sha256
    tasks_sha256: Sha256
    catalog_sha256: Sha256

    # Outcome — filled in when the run finishes
    run_started_at: str
    run_finished_at: str | None = None
    wall_clock_s: float | None = None
    n_trials_completed: int | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
