"""The movie track's seven policies, rendered from one template.

`data/policy.md` is scored content: its digest is recorded on every g1 entry.
`render_policy(MOVIES)` and `render_memory_policy(MOVIES)` reproduce it and
`data/policy_memory.md` byte for byte, which a test pins. The checked-in files
are the artifacts; this module is only how they were produced.
"""
from __future__ import annotations

from tau_rec.domain import DomainSpec

POLICY_TEMPLATE = """\
# {Item} Recommendation Policies

You must follow all of the following policies when assisting the user.

## 1. {History}
Do not recommend a {item} the user has already {consumed}. Always check the user's {history_noun} using `get_user_history` before making a final recommendation.

## 2. Sponsored Content
Some {items} in the catalog are marked as sponsored. If you recommend a sponsored {item}, you must disclose this to the user before they decide. For example: "Note: this is a promoted recommendation."

## 3. Age-Restricted Content
Before recommending a {item} rated {restricted_examples}, you must call `check_content_preference` with the content rating. This verifies whether the rating is age-restricted and reminds you to confirm with the user. Do not skip this step.

## 4. Availability Verification
Only recommend a {item} that is available on at least one of the user's {availability_noun}. Use `check_availability` or `get_metadata` to verify before committing.

## 5. Single Recommendation
Present one recommendation at a time. Wait for the user's response before suggesting an alternative. Do not list multiple recommendations in a single message.

## 6. Transparency
If no {item} in the catalog satisfies all of the user's stated requirements, abstain by calling `recommend()` with no item_id rather than forcing a partial match. You may discuss alternatives in conversation, but do not commit to a recommendation that violates the user's requirements.

## 7. Tool-Based Recommendation
You must use the `recommend(item_id)` tool to make any recommendation. Do not recommend {items} by just naming them in conversation — always call the `recommend` tool first. This ensures the recommendation is properly recorded.
"""

MEMORY_POLICY_TEMPLATE = """\
# {Item} Recommendation Policies (memory mode)

You must follow all of the following policies when assisting the user.

In this conversation you have no catalog tools. The policies on {history_noun},
sponsored content, age-restricted content and availability verification do not
apply, because there is nothing for you to check them against.

## 5. Single Recommendation
Present one recommendation at a time. Wait for the user's response before suggesting an alternative. Do not list multiple recommendations in a single message.

## 6. Transparency
If no {item} you know of satisfies all of the user's stated requirements, abstain by calling `recommend()` with no title rather than forcing a partial match. You may discuss alternatives in conversation, but do not commit to a recommendation that violates the user's requirements.

## 7. Tool-Based Recommendation
You must use the `recommend(title)` tool to make any recommendation. Do not recommend {items} by just naming them in conversation — always call the `recommend` tool. Give the exact title; add the year if the title alone could be ambiguous. This ensures the recommendation is properly recorded.
"""

def _fields(spec: DomainSpec) -> dict[str, str]:
    return {
        "Item": spec.item_noun.capitalize(),
        "item": spec.item_noun,
        "items": spec.item_plural,
        "History": spec.history_noun.title(),
        "history_noun": spec.history_noun,
        "consumed": spec.consumed_adj,
        "restricted_examples": spec.restricted_examples,
        "availability_noun": spec.availability_noun,
    }


def render_policy(spec: DomainSpec) -> str:
    return POLICY_TEMPLATE.format(**_fields(spec))


def render_memory_policy(spec: DomainSpec) -> str:
    return MEMORY_POLICY_TEMPLATE.format(**_fields(spec))
