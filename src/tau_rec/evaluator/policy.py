from __future__ import annotations
import json
import re
from pydantic import BaseModel
from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.task import Task
from tau_rec.data_model.conversation import (
    ConversationTrace, MessageEvent, Role, StopReason, ToolCallEvent,
)
from tau_rec.domain import MOVIES, DomainSpec


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
    def __init__(self, catalog: Catalog | None = None, spec: DomainSpec = MOVIES) -> None:
        self._catalog = catalog
        self._spec = spec

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
    def _check_history(self, task: Task, trace: ConversationTrace) -> bool:
        """Domain-native history policy: lookup must precede commitment."""
        if not trace.all_recommendations:
            return True
        consumed = set()
        history_called = False
        recommended_seen: set[str] = set()
        for event in trace.events:
            if not isinstance(event, ToolCallEvent):
                continue
            if event.name == "get_user_history":
                try:
                    data = json.loads(event.result)
                except (json.JSONDecodeError, TypeError):
                    continue
                if (not isinstance(data, dict) or data.get("error")
                        or event.arguments.get("user_id") != task.user_id):
                    continue
                history_key = self._spec.history_key if self._spec.history_key in data else "watched"
                history = data.get(history_key)
                if not isinstance(history, list):
                    continue
                history_called = True
                for item in history:
                    if isinstance(item, dict):
                        if "id" in item:
                            consumed.add(item["id"])
                    else:
                        consumed.add(item)
            elif event.name == "recommend" and event.arguments.get("item_id"):
                item_id = event.arguments["item_id"]
                recommended_seen.add(item_id)
                if not history_called or item_id in consumed:
                    return False
        return all(rec in recommended_seen and rec not in consumed for rec in trace.all_recommendations)

    def _check_watch_history(self, task: Task, trace: ConversationTrace) -> bool:
        """Legacy outcome retained byte-for-byte for archived movie traces."""
        if not trace.all_recommendations:
            return True
        consumed = set()
        history_called = False
        for tc in trace.tool_calls:
            if tc.name == "get_user_history":
                history_called = True
                try:
                    data = json.loads(tc.result)
                    for item in data.get("watched", []):
                        consumed.add(item["id"] if isinstance(item, dict) else item)
                except (json.JSONDecodeError, AttributeError):
                    pass
        return history_called and not any(rec in consumed for rec in trace.all_recommendations)

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
        restricted_ratings = self._spec.restricted_ratings
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
        availability = task.availability_values(self._spec)
        if not self._catalog or not availability:
            return True
        for rec in trace.all_recommendations:
            movie = self._catalog.get(rec)
            if not movie:
                return False
            available = getattr(movie, self._spec.availability_field)
            if not any(s in available for s in availability):
                return False
        return True

    def _check_format_verification(self, task: Task, trace: ConversationTrace) -> bool:
        """Book-native availability policy: outcome *and* prior verification.

        The legacy availability rule intentionally remains outcome-only so no
        published score moves. New book tasks use this flag and must inspect
        the chosen work with get_metadata or check it against an accepted
        format before the terminal recommend() call.
        """
        accepted = set(task.availability_values(self._spec))
        if not self._catalog or not accepted or not trace.all_recommendations:
            return True

        verified: set[str] = set()
        recommended_seen: set[str] = set()
        for event in trace.events:
            if not isinstance(event, ToolCallEvent):
                continue
            item_id = event.arguments.get("item_id")
            if event.name == "get_metadata" and item_id:
                try:
                    payload = json.loads(event.result)
                except (json.JSONDecodeError, TypeError):
                    payload = {"error": "invalid"}
                if payload.get("redacted") is True:
                    metadata_ok = payload.get("ok") is True
                else:
                    metadata_ok = "error" not in payload
                if metadata_ok:
                    verified.add(item_id)
            elif event.name == "check_availability" and item_id:
                try:
                    payload = json.loads(event.result)
                except (json.JSONDecodeError, TypeError):
                    payload = {}
                requested = event.arguments.get(self._spec.tool_availability_argument)
                if requested is None:  # legacy tool-call fixtures
                    requested = event.arguments.get("services", [])
                if not isinstance(requested, list):
                    requested = []
                confirmed = any(
                    self._spec.normalize_service(str(value)) in accepted
                    and bool(payload.get(str(value)))
                    for value in requested
                )
                if confirmed:
                    verified.add(item_id)
            elif event.name == "recommend" and item_id:
                recommended_seen.add(item_id)
                item = self._catalog.get(item_id)
                if item is None:
                    return False
                available = set(getattr(item, self._spec.availability_field))
                if not (available & accepted) or item_id not in verified:
                    return False

        return all(rec in recommended_seen and rec in verified for rec in trace.all_recommendations)

    def _check_format_verification_v2(self, task: Task, trace: ConversationTrace) -> bool:
        """Format verification as the books-wikidata-v2 policy text states it.

        Version 2 of the rule: it applies only once a user message before the
        commitment names one of the user's accepted formats, in any spelling
        the tools accept. From then on it is the v1 rule above.
        """
        accepted = task.availability_values(self._spec)
        for event in trace.events:
            if isinstance(event, ToolCallEvent) and event.name == "recommend":
                return True
            if (
                isinstance(event, MessageEvent) and event.role == Role.USER
                and any(self._spec.mentions_service(event.content, value) for value in accepted)
            ):
                return self._check_format_verification(task, trace)
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
        # A batched [recommend(X), recommend()] leaves stop_reason == ABSTAINED
        # alongside a concrete recommendation. Abstaining and recommending at
        # once is not transparent.
        return trace.stop_reason == StopReason.ABSTAINED and not trace.recommendations

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
