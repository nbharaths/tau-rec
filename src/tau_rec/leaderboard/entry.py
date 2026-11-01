"""The leaderboard submission format.

An entry stores raw per-task `{n, c}` counts and harness metadata. It stores
**no aggregates** — pass^k, confidence intervals, cost and efficiency are all
derived at render time. That is what lets the board be re-derived when the
evaluator changes, which is not hypothetical: the rec-then-abstain scoring
fix moved a published number.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

# Every entry that predates tracks and modes was a tooled run on movies-v1,
# so those are the defaults and the 25 archived entries need no edit.
DEFAULT_TRACK = "movies-v1"
Mode = Literal["tools", "memory"]

# Model ids ending in a floating tag are refused: they silently re-point at a
# different model and make an entry unreproducible. An unpinned dated alias
# (gpt-5-mini) is allowed but must declare pinned=false + run_date.
FLOATING_SUFFIXES = ("-latest", ":latest", "-preview", "@latest")

SAME_MODEL_AS_SIMULATOR = "same_model_as_simulator"
CONTENT_DIGESTS_UNRECORDED = "content_digests_unrecorded"
# Tasks in the entry do not all have the same n, or some have fewer than the
# standard 4. pass^k is 0 for any task with n < k, so an entry built from a
# run that lost trials to rate limits silently under-reports unless it says so.
UNEVEN_TRIALS = "uneven_trials"
STANDARD_TRIALS = 4


class TaskCount(BaseModel):
    """Trials attempted and trials where primary_reward == 1.0."""

    n: int = Field(ge=0)
    c: int = Field(ge=0)

    @model_validator(mode="after")
    def _c_within_n(self) -> "TaskCount":
        if self.c > self.n:
            raise ValueError(f"c={self.c} exceeds n={self.n}")
        return self


class LeaderboardEntry(BaseModel):
    submission_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    display_name: str
    submitted_by: str

    model_id: str
    pinned: bool
    run_date: str
    reasoning_effort: str | None = None
    # Superseded by `mode`. Kept so old entry files still parse; an entry
    # claiming no_tools=True is the vacuous ablation and is refused below.
    no_tools: bool = False

    # Which scored content set (track) and which agent configuration (mode)
    # produced the counts. `memory` entries score constraints only and are
    # rendered in their own table; they never rank against tooled entries.
    track: str = DEFAULT_TRACK
    mode: Mode = "tools"

    simulator_model: str
    harness_generation: str
    tau_rec_version: str
    trials_per_task: int = Field(ge=1)

    # Null only for archival entries whose content predates digest recording.
    # New submissions must supply all three; see CONTENT_DIGESTS_UNRECORDED.
    policy_sha256: Sha256 | None = None
    tasks_sha256: Sha256 | None = None
    catalog_sha256: Sha256 | None = None

    per_task: dict[str, TaskCount]
    traces_url: str | None = None
    caveats: list[str] = Field(default_factory=list)

    model_config = {"protected_namespaces": ()}

    @model_validator(mode="after")
    def _check_model_id(self) -> "LeaderboardEntry":
        lowered = self.model_id.lower()
        for suffix in FLOATING_SUFFIXES:
            if lowered.endswith(suffix):
                raise ValueError(
                    f"model_id {self.model_id!r} ends in floating tag {suffix!r}; "
                    "pin a dated snapshot or record the alias with pinned=false"
                )
        return self

    @model_validator(mode="after")
    def _refuse_no_tools_ablation(self) -> "LeaderboardEntry":
        """`--no-tools` removed recommend() along with everything else, so no
        recommendation could ever register and every satisfiable task scored 0
        by construction. Those runs measure nothing; memory mode replaces them."""
        if self.no_tools:
            raise ValueError(
                "no_tools=True is the vacuous ablation (no recommend() tool, so no "
                "recommendation can register); re-run under mode='memory' instead"
            )
        return self

    @model_validator(mode="after")
    def _flag_same_model_as_simulator(self) -> "LeaderboardEntry":
        """The simulator is worth ~13.7 pass^1 points against a 28-point board
        span, so an agent graded by itself is confounded and must say so."""
        if self.model_id == self.simulator_model:
            if SAME_MODEL_AS_SIMULATOR not in self.caveats:
                self.caveats.append(SAME_MODEL_AS_SIMULATOR)
        return self

    @model_validator(mode="after")
    def _flag_missing_digests(self) -> "LeaderboardEntry":
        """An entry with no content digests cannot be checked for comparability,
        so it must say so on the board rather than look equivalent."""
        missing = not all(
            (self.policy_sha256, self.tasks_sha256, self.catalog_sha256)
        )
        if missing and CONTENT_DIGESTS_UNRECORDED not in self.caveats:
            self.caveats.append(CONTENT_DIGESTS_UNRECORDED)
        return self

    @property
    def n_tasks(self) -> int:
        return len(self.per_task)

    @property
    def task_results(self) -> dict[str, dict[str, int]]:
        """Shape expected by `tau_rec.metrics.pass_k.aggregate_pass_k`."""
        return {tid: {"n": tc.n, "c": tc.c} for tid, tc in self.per_task.items()}
