"""What a log ran: content digests, versions and resolved model settings.

Recorded on the task's metadata, which Inspect writes into the log header
(``log.eval.metadata``), so every log says which τ-Rec it ran and under what
settings (REQ-023).
"""
from __future__ import annotations

import json
from typing import Any

from inspect_ai._util.error import PrerequisiteError
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.model._generate_config import active_generate_config
from inspect_ai.model._model_config import model_args_for_log

from tau_rec import __version__
from tau_rec.agents.request_settings import (
    agent_request_settings,
    native_model_name,
    simulator_request_settings,
)
from tau_rec.harness import harness_commit
from tau_rec.inspect_support.config import fill_unset
from tau_rec.inspect_support.simulator import DEFAULT_USER_MODEL, user_model
from tau_rec.inspect_support.solver import MODE, environment
from tau_rec.leaderboard.hashing import hash_file, hash_task_dir

# The leaderboard generation the movies-v1 harness scores under
# (leaderboard/GENERATIONS.md); the g1 entries record the same digests.
GENERATION = "g1"

SAME_MODEL_AS_SIMULATOR = "same_model_as_simulator"
# The task was built outside an Inspect eval (no active model): nothing about
# the models is known, so the model fields are None.
MODEL_SETTINGS_UNRECORDED = "model_settings_unrecorded"
# The user role's client could not be built (no provider credentials): its
# name is recorded, its settings and model args are None.
SIMULATOR_SETTINGS_UNRESOLVED = "simulator_settings_unresolved"


def _settings(config: GenerateConfig) -> dict[str, Any]:
    return config.model_dump(mode="json", exclude_none=True)


def _model_args(model: Model) -> dict[str, Any]:
    # Inspect's own log redaction (no api_key, no generators); anything else
    # that is not plain JSON (a mock's callable) is recorded by its repr.
    return json.loads(json.dumps(model_args_for_log(model.model_args), default=repr))


def resolved_user_model() -> Model | None:
    """The ``user`` role model, or None when its client cannot be built.

    Building the default ``openai/gpt-5-mini`` client needs OPENAI_API_KEY.
    Recording metadata must never need credentials: without them the task
    still builds, and fails only when the simulator is actually called.
    """
    try:
        return user_model()
    except PrerequisiteError:
        return None


def comparability_metadata(active: Model | None, user: Model | None) -> dict[str, Any]:
    """The comparability record for a run of ``active`` against the ``user`` role.

    The request settings are what each side sends: the native settings for the
    model, filling only what the user left unset, resolved as
    ``InspectAgent._generate_config`` and ``InspectSimulator._call_once`` do.

    With no ``active`` model (the task was built outside an Inspect eval) the
    model fields are None and the caveat ``model_settings_unrecorded`` is set.
    With no ``user`` model (its client could not be built) the simulator is
    recorded by its default name, its settings and args are None, and the
    caveat ``simulator_settings_unresolved`` is set.
    """
    env = environment()
    track = env.track
    harness = env.harness

    record: dict[str, Any] = {
        "track": track.id,
        "generation": GENERATION,
        "digests": {
            "policy_sha256": hash_file(track.policy_for(MODE)),
            "tasks_sha256": hash_task_dir(track.tasks_path),
            "catalog_sha256": hash_file(track.catalog_path),
        },
        "tau_rec_version": __version__,
        "harness_commit": harness_commit(),
        "request_settings": None,
        "model_args": None,
        "simulator_model": None,
        "caveats": [],
    }
    caveats: list[str] = record["caveats"]
    if active is None:
        caveats.append(MODEL_SETTINGS_UNRECORDED)
        return record

    agent_explicit = active.config.merge(active_generate_config())
    agent_config = fill_unset(
        agent_request_settings(
            native_model_name(str(active)), agent_explicit.reasoning_effort, harness
        ),
        agent_explicit,
    )
    if user is None:
        simulator_model = DEFAULT_USER_MODEL
        simulator_settings = simulator_args = None
        caveats.append(SIMULATOR_SETTINGS_UNRESOLVED)
    else:
        simulator_model = str(user)
        simulator_settings = _settings(fill_unset(
            simulator_request_settings(native_model_name(str(user))), user.config
        ))
        simulator_args = _model_args(user)
    if str(active) == simulator_model:
        caveats.append(SAME_MODEL_AS_SIMULATOR)

    record.update({
        "request_settings": {"agent": _settings(agent_config), "simulator": simulator_settings},
        "model_args": {"agent": _model_args(active), "simulator": simulator_args},
        "simulator_model": simulator_model,
    })
    return record
