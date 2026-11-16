"""τ-Rec movies-v1 as an Inspect task.

    uv run inspect eval src/tau_rec/inspect_task.py@tau_rec_movies \\
      --model openai/gpt-5-mini -M responses_api=false

The agent under test is ``--model``; the simulated user is the ``user`` role
(default ``openai/gpt-5-mini``). Each sample is one native τ-Rec trial, scored
by the native evaluator; the default 4 epochs report pass^1, pass^2 and
pass^4. For another epoch count N, name only reducers with k <= N:
``--epochs N --epochs-reducer pass_k_1[,pass_k_2...]``.
"""
from __future__ import annotations

from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model._model import active_model
from inspect_ai.scorer import pass_k

import tau_rec
from tau_rec.inspect_support.metadata import comparability_metadata
from tau_rec.inspect_support.scorer import tau_rec_scorer
from tau_rec.inspect_support.simulator import user_model
from tau_rec.inspect_support.solver import environment, tau_rec_solver


def repo_root() -> Path:
    """The repository root, whose ``data/`` holds the movies-v1 track."""
    root = Path(__file__).resolve().parents[2]
    if not (root / "data" / "track.json").exists():
        raise RuntimeError(
            f"τ-Rec data not found at {root / 'data'}: run the task from a clone of "
            "the repository installed with `uv sync` (an editable install), not "
            "from a wheel or site-packages install."
        )
    return root


@task
def tau_rec_movies(max_turns: int = 20) -> Task:
    repo_root()  # fail first, and clearly, outside a repository checkout
    env = environment()

    active = active_model()
    if active is None:
        raise RuntimeError(
            "tau_rec_movies records comparability metadata for the model under "
            "test, so it must be created by Inspect: run `inspect eval "
            "src/tau_rec/inspect_task.py@tau_rec_movies`, or pass the task "
            "function (not an instance) to inspect_ai.eval()."
        )

    dataset = MemoryDataset(
        [
            Sample(
                id=task_id,
                input=f"τ-Rec {env.track.id} {task_id}",
                metadata={
                    "complexity": t.complexity,
                    "reveal_difficulty": t.reveal_difficulty,
                },
            )
            for task_id, t in env.tasks.items()
        ],
        name=env.track.id,
    )
    return Task(
        dataset=dataset,
        solver=tau_rec_solver(max_turns),
        scorer=tau_rec_scorer(),
        epochs=Epochs(4, [pass_k(1), pass_k(2), pass_k(4)]),
        fail_on_error=False,
        version=tau_rec.__version__,
        metadata=comparability_metadata(active, user_model()),
    )
