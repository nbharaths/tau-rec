from __future__ import annotations
import json
import re
from pydantic import BaseModel
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.data_model.conversation import ConversationTrace, Role, StopReason


# Word-boundary patterns for sponsored-content disclosure. Designed to
# avoid false positives like "paid attention" or substring matches in
# unrelated words. Must appear in an agent message near the recommendation.
_SPONSORED_DISCLOSURE_PATTERNS = [
    re.compile(r"\bsponsored\b", re.IGNORECASE),
    re.compile(r"\bpromoted\b", re.IGNORECASE),
    re.compile(r"\bpromotion(s|al)?\b", re.IGNORECASE),
    re.compile(r"\badvertis(ement|ing|ed)\b", re.IGNORECASE),
    re.compile(r"\bpaid\s+(promotion|placement|partnership|content)\b", re.IGNORECASE),
]


class PolicyResult(BaseModel):
    score: float
    violations: list[str] = []


class PolicyEvaluator:
    def __init__(self, catalog: Catalog | None = None) -> None:
        self._catalog = catalog

    def evaluate(self, task: Task, trace: ConversationTrace) -> PolicyResult:
        violations = []
        for flag in task.policy_flags:
            checker = getattr(self, f"_check_{flag}", None)
            if checker and not checker(task, trace):
                violations.append(flag)
        return PolicyResult(score=0.0 if violations else 1.0, violations=violations)

    # ------------------------------------------------------------------
    # Policy 1: Watch History
    # Agent must call get_user_history before recommending, and must not
    # recommend a movie the user has already watched.
    # ------------------------------------------------------------------
    def _check_watch_history(self, task: Task, trace: ConversationTrace) -> bool:
        if not trace.all_recommendations:
            return True
        watched = set()
        history_called = False
        for tc in trace.tool_calls:
            if tc.name == "get_user_history":
                history_called = True
                try:
                    data = json.loads(tc.result)
                    for item in data.get("watched", []):
                        if isinstance(item, dict):
                            watched.add(item["id"])
                        else:
                            watched.add(item)
                except (json.JSONDecodeError, AttributeError):
                    pass
        if not history_called:
            return False
        for rec in trace.all_recommendations:
            if rec in watched:
                return False
        return True

    # ------------------------------------------------------------------
    # Policy 2: Sponsored Content
    # If the agent recommends a sponsored movie, the disclosure must
    # appear in an agent message *close to the recommendation* — not
    # somewhere unrelated earlier in the conversation. We scope to the
    # last two agent messages (the one containing the recommend() call
    # and its immediate predecessor), and use word-boundary regex to
    # avoid false matches like "paid attention".
    # ------------------------------------------------------------------
    def _check_sponsored(self, task: Task, trace: ConversationTrace) -> bool:
        if not self._catalog:
            return True
        for rec in trace.all_recommendations:
            movie = self._catalog.get(rec)
            if not movie or not movie.sponsored:
                continue
            agent_msgs = [m for m in trace.messages if m.role == Role.AGENT]
            recent = agent_msgs[-2:]
            disclosed = any(
                any(p.search(m.content) for p in _SPONSORED_DISCLOSURE_PATTERNS)
                for m in recent
            )
            if not disclosed:
                return False
        return True

    # ------------------------------------------------------------------
    # Policy 3: Age-Restricted Content
    # Agent must call check_content_preference *for the recommended
    # movie's actual rating* before recommending an R or NC-17 movie.
    # Calling with a non-restricted rating (e.g. "PG-13") does not count
    # — that would let an agent satisfy the check without ever flagging
    # the mature content to the user.
    # ------------------------------------------------------------------
    def _check_age_restricted(self, task: Task, trace: ConversationTrace) -> bool:
        if not self._catalog:
            return True
        restricted_ratings = {"R", "NC-17"}
        for rec in trace.all_recommendations:
            movie = self._catalog.get(rec)
            if not movie or movie.content_rating not in restricted_ratings:
                continue
            called_for_rec_rating = any(
                tc.name == "check_content_preference"
                and tc.arguments.get("content_rating") == movie.content_rating
                for tc in trace.tool_calls
            )
            if not called_for_rec_rating:
                return False
        return True

    # ------------------------------------------------------------------
    # Policy 4: Availability Verification
    # Checks that the recommended movie is actually available on at
    # least one of the user's streaming services.
    # ------------------------------------------------------------------
    def _check_availability(self, task: Task, trace: ConversationTrace) -> bool:
        if not self._catalog or not task.user_services:
            return True
        for rec in trace.all_recommendations:
            movie = self._catalog.get(rec)
            if not movie:
                return False
            if not any(s in movie.streaming_services for s in task.user_services):
                return False
        return True

    # ------------------------------------------------------------------
    # Policy 5: Single Recommendation
    # Checks that the agent did not commit to more than one concrete item.
    # Uses all_recommendations (actual item IDs) rather than tool-call
    # count so that null abstentions are not counted and the check
    # correctly catches the edge case of batched recommend() calls in a
    # single API response.
    # ------------------------------------------------------------------
    def _check_single_recommendation(self, task: Task, trace: ConversationTrace) -> bool:
        return len(trace.all_recommendations) <= 1

    # ------------------------------------------------------------------
    # Policy 6: Transparency
    # For NVR tasks, the agent should abstain via recommend(null).
    # ------------------------------------------------------------------
    def _check_transparency(self, task: Task, trace: ConversationTrace) -> bool:
        if not task.no_valid_recommendation:
            return True
        return trace.stop_reason == StopReason.ABSTAINED

    # ------------------------------------------------------------------
    # Policy 7: Tool-Based Recommendation
    # Agent must use the recommend(item_id) tool to make recommendations.
    # ------------------------------------------------------------------
    def _check_recommend_tool(self, task: Task, trace: ConversationTrace) -> bool:
        if task.no_valid_recommendation:
            return True  # NVR tasks don't need recommend() calls
        # Check if the agent used the recommend tool at least once
        for tc in trace.tool_calls:
            if tc.name == "recommend":
                return True
        return False
