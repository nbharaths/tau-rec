from __future__ import annotations
import asyncio
from tau_rec.data_model.task import Task, RevealTag
from tau_rec.data_model.conversation import Message, Role

MAX_RETRIES = 8
RETRY_DELAY = 30

BASE_GUIDELINES_SATISFIABLE = """\
You are simulating a real person looking for a movie recommendation.
Behave naturally. Do not be adversarial. Respond conversationally.

A recommendation means the assistant has committed to a specific movie — naming it
as THE pick for you, not offering multiple candidates or discussing possibilities.

When the assistant makes a recommendation:
- If it fits your preferences and constraints, respond with ###ACCEPTED###
- If it does not, respond with ###REJECTED### and briefly explain why.

If the assistant stalls, keeps asking clarifying questions without committing, or says
they cannot find a match, DO NOT accept. Push back: tell them to just pick their best
current option and commit to a single recommendation. Stay in the conversation — do
not emit ###ACCEPTED### until they actually name a specific movie as their pick.

If the assistant has named a specific title but hedges (e.g. "the closest match is X,
but I can't verify availability"), treat that as a concrete recommendation of X and
judge it on your constraints.

Follow the reveal rules below strictly.
"""

BASE_GUIDELINES_NO_VALID = """\
You are simulating a real person looking for a movie recommendation, but your
requirements are intentionally impossible — nothing in the catalog will satisfy
them. Behave naturally. Do not be adversarial.

When the assistant tells you no movie matches your criteria (either in plain words
or by declining to recommend anything), respond with ###ACCEPTED###.

If the assistant attempts to recommend a specific movie anyway, respond with
###REJECTED### and explain which requirement it fails.

Follow the reveal rules below strictly.
"""


def _describe_constraint(tc) -> str:
    c = tc.constraint
    return f"{c.field} {c.op} {c.value}"


def build_simulator_prompt(task: Task) -> str:
    base = BASE_GUIDELINES_NO_VALID if task.no_valid_recommendation else BASE_GUIDELINES_SATISFIABLE
    sections = [base]

    # Persona
    sections.append(f"\n## Your Persona\n{task.persona}")

    # Soft preferences
    if task.soft_preferences:
        prefs = "\n".join(f"- {p}" for p in task.soft_preferences)
        sections.append(f"\n## Your Soft Preferences\nUse these to judge recommendations:\n{prefs}")

    # Streaming services (share when asked)
    if task.user_services:
        svc_list = ", ".join(task.user_services)
        sections.append(
            f"\n## Streaming Services You Have\n"
            f"You subscribe to: {svc_list}.\n"
            f"If the assistant asks which streaming services you have access to, share this list. "
            f"If the assistant recommends a movie, you may assume they've verified availability through their tool."
        )
    else:
        sections.append(
            f"\n## Streaming Services You Have\n"
            f"Streaming availability is not a concern for you.\n"
            f"If the assistant asks which streaming services you have, say you have access to all major services."
        )

    # Reveal rules
    volunteer, on_ask, hidden = [], [], []
    for tc in task.constraints:
        desc = _describe_constraint(tc)
        match tc.reveal:
            case RevealTag.VOLUNTEER:
                volunteer.append(desc)
            case RevealTag.ON_ASK:
                on_ask.append(desc)
            case RevealTag.HIDDEN:
                hidden.append(desc)

    rules = ["\n## Reveal Rules"]
    if volunteer:
        items = "\n".join(f"- {d}" for d in volunteer)
        rules.append(f"\n**Share proactively** (these are your requirements for THIS conversation — state them clearly in your first message, even if they differ from your general persona):\n{items}")
    if on_ask:
        items = "\n".join(f"- {d}" for d in on_ask)
        rules.append(
            f"\n**Share only if asked** (only reveal these if the assistant asks a relevant question. "
            f"However, ALWAYS reject recommendations that violate these, even if you have not shared them yet):\n{items}"
        )
    if hidden:
        items = "\n".join(f"- {d}" for d in hidden)
        rules.append(
            f"\n**Never share explicitly** (use only to internally evaluate recommendations. "
            f"If a recommendation violates these, reject it and hint at why without stating the constraint):\n{items}"
        )
    sections.append("\n".join(rules))

    return "\n".join(sections)


class UserSimulator:
    def __init__(self, model: str, task: Task) -> None:
        self.model = model
        self.task = task
        self.system_prompt = build_simulator_prompt(task)

    async def respond(self, conversation: list[Message]) -> Message:
        import litellm

        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in conversation:
            role = "assistant" if msg.role == Role.USER else "user"
            messages.append({"role": role, "content": msg.content})

        kwargs = {"model": self.model, "messages": messages, "temperature": 1.0}
        if "deepseek" in self.model.lower():
            kwargs["thinking"] = {"type": "disabled"}
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        for attempt in range(MAX_RETRIES):
            try:
                response = await litellm.acompletion(**kwargs)
                from tau_rec.cost import record_usage
                record_usage("sim", response)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1 and ("503" in str(e) or "429" in str(e) or "rate_limit" in str(e).lower() or "overloaded" in str(e).lower() or "unavailable" in str(e).lower() or "connection" in str(e).lower()):
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise
        content = response.choices[0].message.content
        return Message(role=Role.USER, content=content)
