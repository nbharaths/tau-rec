from tau_rec.simulator.user_simulator import build_simulator_prompt
from tau_rec.data_model.task import Task, TaskConstraint, Constraint, RevealTag

def test_build_simulator_prompt():
    task = Task(
        id="t1",
        constraints=[
            TaskConstraint(
                constraint=Constraint(field="runtime", op="<=", value=120),
                reveal=RevealTag.VOLUNTEER,
            ),
            TaskConstraint(
                constraint=Constraint(field="genres", op="contains", value="Thriller"),
                reveal=RevealTag.ON_ASK,
            ),
            TaskConstraint(
                constraint=Constraint(field="streaming_services", op="contains_any", value=["Netflix"]),
                reveal=RevealTag.HIDDEN,
            ),
        ],
        persona="You are a college student looking for a weekend movie.",
        soft_preferences=["likes plot twists"],
        complexity="medium", reveal_difficulty="mixed",
    )
    prompt = build_simulator_prompt(task)
    assert "###ACCEPTED###" in prompt
    assert "###REJECTED###" in prompt
    assert "college student" in prompt
    assert "runtime" in prompt

