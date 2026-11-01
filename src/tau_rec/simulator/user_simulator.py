from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from tau_rec.data_model.catalog import Catalog
from tau_rec.data_model.conversation import Message, Role
from tau_rec.data_model.task import RevealTag, Task
from tau_rec.domain import MOVIES, DomainSpec
from tau_rec.harness import LEGACY, CallTimeout, HarnessProfile, timed_call

MAX_RETRIES = 8
RETRY_DELAY = 30
REQUEST_TIMEOUT_SECONDS = 120

# Verdicts come only from candidate markers, parsed tolerantly; ACCEPT only for
# eligible books: unread, and in a stated format once the user has named formats.
BOOKS_GROUNDING_VERSION = "candidate-id-v5"
# "I'd rather not say" for hidden fields, "No preference on that" for fields
# with no requirement, restated facts for re-asked fields; formats in the
# volunteer opening; a neutral connector before answers in candidate turns.
BOOKS_CONTROLLER_VERSION = "book-disclosure-v4"
# LLM users answer questions asked alongside a candidate, as the symbolic user
# does; each such call is tagged with this purpose in the simulator call log.
# The model sees the whole turn and the verdict already given.
BOOKS_LLM_EMBEDDED_ANSWERS_VERSION = "embedded-answer-v2"
# One contract for satisfiable and no-valid-answer tasks: no push-back and no
# hints at why a candidate failed, only volunteer facts in the first message,
# the same reply rules as the symbolic user, and no verdicts from the model.
BOOKS_LLM_USER_VERSION = "books-llm-user-v3"
SIMULATOR_INTERACTION_MODES = ("llm", "symbolic")
# Tolerant of one bracket or two on either side, "=" for ":", a space for "_"
# and backticks around the ID. A marker start with no parsable marker is a
# truncated one.
_CANDIDATE_MARKER = re.compile(
    r"\[\[?\s*candidate[_ ]?id\s*[:=]\s*`?([^\]\s`]+)`?\s*\]\]?", re.IGNORECASE
)
_MARKER_START = re.compile(r"\[\[?\s*candidate[_ ]?id\b", re.IGNORECASE)
_GENRE_NAMES = (
    "adventure", "biography", "biographies", "children", "childrens", "children s",
    "kids", "crime", "fantasy", "graphic novel", "graphic novels", "comic", "comics",
    "historical fiction", "history", "horror", "humor", "humour", "funny", "comedy",
    "memoir", "memoirs", "mystery", "mysteries", "nonfiction", "non fiction",
    "poetry", "poems", "romance", "romantic", "science fiction", "sci fi", "scifi",
    "self help", "thriller", "thrillers", "young adult", "ya",
)
_LANGUAGE_NAMES = (
    "english", "french", "german", "spanish", "italian", "portuguese", "russian",
    "japanese", "chinese", "mandarin", "korean", "arabic", "hindi", "swedish",
    "norwegian", "danish", "dutch", "polish", "greek", "latin", "hebrew", "turkish",
    "persian", "czech", "finnish", "hungarian", "ukrainian", "yiddish",
)
# Nationality words also name a language or a regional language variant
# ("British English"), so they count as questions about both fields.
_NATIONALITIES = (
    "american", "british", "australian", "canadian", "irish", "scottish", "welsh",
    "indian", "mexican", "brazilian", "nigerian", "french", "german", "spanish",
    "italian", "portuguese", "russian", "japanese", "chinese", "korean", "swedish",
    "norwegian", "danish", "dutch", "polish", "greek", "turkish",
)
_COUNTRY_NAMES = (
    "america", "usa", "u s", "united states", "britain", "uk", "united kingdom",
    "england", "scotland", "wales", "ireland", "canada", "australia", "new zealand",
    "france", "germany", "italy", "spain", "portugal", "russia", "japan", "china",
    "india", "mexico", "brazil", "nigeria", "south africa", "sweden", "norway",
    "denmark", "netherlands", "poland", "greece", "korea", "argentina", "chile",
    "colombia", "egypt", "israel", "turkey",
)
_GENERIC_QUESTION_TERMS = (
    "anything else", "any other", "other preference", "other requirement",
    "what matters", "what are you looking for", "tell me more", "more specific",
    "narrow it down", "must have", "deal breaker",
)
_READING_HISTORY = re.compile(r"\b(?:reading|read|your|user|purchase) history\b")
_WHERE_FROM = re.compile(r"\bwhere\b.*\bfrom\b")


@dataclass(frozen=True)
class BookQuestionIntent:
    kind: str
    fields: tuple[str, ...] = ()
    generic: bool = False


def _normalized_question(message: str) -> str:
    return " ".join(re.sub(r"[^\w?]+", " ", message.casefold()).split())


_SUBGENRE_NAMES = (
    "cozy", "cosy", "hardboiled", "hard boiled", "noir", "whodunit", "whodunits",
    "dystopian", "dystopia", "cyberpunk", "space opera", "epic fantasy", "high fantasy",
    "urban fantasy", "dark fantasy", "gothic", "supernatural", "true crime", "espionage",
    "literary fiction", "satire", "lovecraftian",
)
_REGIONAL_LANGUAGE_WORDS = ("american", "british", "australian", "canadian")
# Field words.  Value names (genre, language, country names) are matched
# separately and only inside a choice or preference frame.
_BOOK_QUESTION_TERMS = {
    "genres": (
        "genre", "genres", "subgenre", "subgenres", "sub genre", "kind of book",
        "kind of story", "type of book", "kind of nonfiction", "kind of fiction",
        "kind of novel", "type of nonfiction", "fiction", "fiction or nonfiction",
        "subject area", "subject matter", "theme", "themes", "topic", "topics",
        "flavor", "flavour",
    ),
    "languages": (
        "language", "languages", "original language", "written in", "translated",
        "translation", "translations",
    ),
    "countries": (
        "country", "countries", "country of origin", "from where", "nationality",
        "origin", "region", "part of the world", "author from", "authors from",
        "where the author",
    ),
    "pages": ("page", "pages", "length", "long", "short", "quick read"),
    "rating": (
        "rating", "ratings", "review score", "reviewed", "highly rated", "reader score",
    ),
    "publication_year": (
        "publication", "published", "release year", "year", "years", "recent", "recency", "newer",
        "older", "classic", "classics", "era", "eras", "decade", "decades",
        "time period", "period", "century", "centuries", "date", "dates", "modern",
        "vintage", "how old", "how recent", "new release", "new releases",
    ),
    "formats": (
        "format", "formats", "ebook", "ebooks", "e book", "e books", "kindle",
        "audiobook", "audiobooks", "audio book", "audio books", "paperback",
        "hardcover", "physical", "digital", "print",
    ),
}
# Era words that describe a story's setting rather than its publication.
_SETTING_YEAR_TERMS = (
    "era", "eras", "time period", "period", "century", "centuries", "decade",
    "decades", "date", "dates", "year", "years",
)
# Open-ended requests for requirements the agent does not name.  These alone
# make a field question also ask for one more fact.
_OPEN_REQUEST_TERMS = (
    "anything else", "any other", "what else", "other requirement", "other preference",
    "additional constraint", "additional requirement", "additional detail",
    "something else i should", "what s missing", "what is missing", "what i m missing",
    "i m missing", "still missing", "missing requirement", "requirement i m",
    "still unmet", "isn t being met", "not being met", "didn t work", "didn t fit",
    "doesn t fit", "not working", "isn t working", "not matching", "what specifically",
    "what specific", "which requirement", "specific requirement", "requirement i should",
    "should be matching", "concrete detail", "one detail", "a hint", "any hint",
    "small hint", "specific hint", "what you re looking for", "what you are looking for",
    "what you re after", "what you re in the mood for", "what you re interested in",
    "what you re hoping for", "what you enjoy", "what you d like",
    "full set of",
    "what else the book needs", "tell me the missing", "not fitting", "isn t fitting",
    "one hint", "specific detail", "anything you d like to avoid", "anything to avoid",
    "requirement did", "requirement is failing", "requirements are failing", "what was missing",
    "what s off", "what is off", "remaining requirement", "one more hint", "another hint",
    "what requirement", "isn t matching", "were missing", "not quite right", "about your taste",
    "what are you in the mood for", "missed the mark",
)
# A verdict check ("does this fit what you're looking for?") or a field-specific
# "any other genres" does not invite unnamed requirements.
_NOT_OPEN_REQUEST = re.compile(
    r"\b(?:fit|fits|match|matches|meet|meets|suit|suits|sound like|sounds like|sound more like"
    r"|sounds more like|more like|closer to|close to) what you re (?:looking for|after)\b"
    r"|\bany other (?:genre|genres|language|languages|format|formats|country|countries)\b"
)
# "I'd like to suggest ..." offers something; it does not ask.
_OFFER_PHRASES = re.compile(r"\bi d (?:like|love) to (?:suggest|recommend|propose|offer|share)\b")
# A sentence without "?" asks only when it makes an explicit request.
_REQUEST_PHRASES = (
    "could you", "can you", "would you", "do you", "are you", "will you", "please",
    "tell me", "let me know", "share", "i ll need", "i will need", "i d love", "i d like",
    "i d really like", "i d appreciate", "i d really appreciate", "is there anything",
    "is there something", "any preference", "if you have a preferred", "if you have any",
    "if you d prefer", "if you prefer", "would really help", "help to know", "would be helpful",
)
_ABBREVIATION = re.compile(r"\b(?:vs|e\.g|i\.e|etc|dr|mr|mrs|ms|st|jr|sr|vol|no|u\.s|u\.k)\.(?=\s)", re.IGNORECASE)
_SENTENCE_END = re.compile(r"([.!?;:])([*_\"'”’)\]]*)[ \t]+")
_CHOICE_FRAME = re.compile(
    r"\b(?:prefer|preference|preferred|like|enjoy|interested|mood|looking for|want|lean"
    r"|fancy|drawn|appeal|appeals|appealing|or|vs|versus|either|in mind|favorite|favourite)\b"
)
_SETTING_PHRASE = re.compile(
    r"\b(?:era|eras|period|time period)\b\W+(?:or\W+)?setting\b"
    r"|\bsetting\W+(?:or\W+)?(?:era|eras|period|time period)\b|\bhistorical period\b"
)
# "A particular era or place" asks about a book's subject, not its publication.
_SUBJECT_PHRASE = re.compile(
    r"\b(?:era|eras|period|time period)\W+(?:or\W+)?(?:place|places|location|locations|event|events)\b"
    r"|\b(?:place|places|location|locations|event|events)\W+(?:or\W+)?(?:era|eras|period|time period)\b"
)
# Country words that, beside a setting, describe where a story takes place.
_SETTING_COUNTRY_TERMS = ("region", "part of the world")
_SETTING_EXAMPLES = re.compile(
    r"\b(?:e g|ancient|medieval|napoleonic|victorian|regency|renaissance|wwi|wwii"
    r"|world war|revolution|civil war|antiquity|frontier|\d+(?:st|nd|rd|th) century)\b"
)
_ERA_QUALIFIED_GENRE = re.compile(
    r"\b(classic|modern|recent|older|newer|vintage)\s+(?:"
    + "|".join(sorted(map(re.escape, _GENRE_NAMES + ("whodunit", "whodunits")), key=len, reverse=True))
    + r")\b"
)
_DECADE = re.compile(r"\b(?:(?:1[5-9]|20)?\d0s)\b")
# "A classic whodunit" asks about a kind of book; "classic or modern" asks
# about publication era.
_GENRE_PATTERN = "|".join(
    sorted(map(re.escape, _GENRE_NAMES + _SUBGENRE_NAMES + ("whodunit", "whodunits")), key=len, reverse=True)
)
_CLASSIC_GENRE = re.compile(
    rf"\bclassic\s+(?:\w+\s+){{0,2}}?(?:{_GENRE_PATTERN})\b|\b(?:{_GENRE_PATTERN})(?: s)?\s+classics?\b"
)
_OLDER_WORDS = (" classic ", " classics ", " older ", " vintage ")
_NEWER_WORDS = (" modern ", " recent ", " newer ", " contemporary ")


def _sentences(message: str) -> list[str]:
    """Split on sentence ends (even before markdown), newlines, not abbreviations."""
    text = _ABBREVIATION.sub(lambda match: match.group(0).replace(".", "․"), message)
    text = _SENTENCE_END.sub(r"\1\2\n", text)
    return [piece for piece in re.split(r"\n+", text) if piece.strip()]


def _question_sentences(message: str) -> list[str]:
    sentences = (_normalized_question(raw) for raw in _sentences(message))
    return [
        sentence for sentence in sentences
        if sentence and (
            "?" in sentence
            or any(p in _OFFER_PHRASES.sub(" ", sentence) for p in _REQUEST_PHRASES)
        )
    ]


def classify_book_question(message: str) -> BookQuestionIntent:
    """Map a user-facing turn to a frozen disclosure intent.

    Deliberately lexical and auditable.  Only sentences that ask something
    ("?" or an explicit request) count, so field words in a search report or a
    candidate's fact sheet are ignored.  Value names (fantasy, French, Japan)
    count only inside a choice or preference frame; era words next to "setting"
    with setting examples describe the story, not its publication.  A field
    question also asks for one more fact only when it explicitly invites
    unnamed requirements ("anything else?", "what's missing?").  Unknown
    paraphrases fall back to no disclosure; no model or fuzzy semantic judge can
    silently expose a hidden fact.
    """
    found: set[str] = set()
    any_generic = extra_generic = False
    for sentence in _question_sentences(message):
        padded = f" {_READING_HISTORY.sub(' ', sentence.replace('?', ' '))} "
        padded = padded.replace(" country of publication ", " country of origin ")
        setting = bool(_SUBJECT_PHRASE.search(padded)) or (
            bool(_SETTING_PHRASE.search(padded))
            and (" historical period " in padded or bool(_SETTING_EXAMPLES.search(padded)))
        )
        contrast = any(w in padded for w in _OLDER_WORDS) and any(w in padded for w in _NEWER_WORDS)
        style = bool(_CLASSIC_GENRE.search(padded)) and not contrast
        year_text = (
            _CLASSIC_GENRE.sub(lambda match: re.sub(r"\bclassics?\b", " ", match.group(0)), padded)
            if style else padded
        )
        for field, terms in _BOOK_QUESTION_TERMS.items():
            if setting and field == "publication_year":
                terms = tuple(term for term in terms if term not in _SETTING_YEAR_TERMS)
            if setting and field == "countries":
                terms = tuple(term for term in terms if term not in _SETTING_COUNTRY_TERMS)
            text = year_text if field == "publication_year" else padded
            if any(f" {term} " in text for term in terms):
                found.add(field)
        names = padded if style else _ERA_QUALIFIED_GENRE.sub(lambda match: match.group(1), padded)
        genre_names = _GENRE_NAMES + _SUBGENRE_NAMES
        framed = bool(_CHOICE_FRAME.search(padded)) or (
            sum(f" {name} " in names for name in genre_names) >= 2
        )
        if framed and any(f" {name} " in names for name in genre_names):
            found.add("genres")
        if framed and not setting:
            if any(f" {name} " in padded for name in _LANGUAGE_NAMES + _REGIONAL_LANGUAGE_WORDS):
                found.add("languages")
            language_listed = any(
                f" {term} " in padded for term in _BOOK_QUESTION_TERMS["languages"]
            )
            if not language_listed and any(
                f" {name} " in padded for name in _NATIONALITIES + _COUNTRY_NAMES
            ):
                found.add("countries")
        if not setting and _DECADE.search(padded):
            found.add("publication_year")
        if _WHERE_FROM.search(padded):
            found.add("countries")
        request = _NOT_OPEN_REQUEST.sub(" ", sentence)
        any_generic = any_generic or any(
            term in request for term in _GENERIC_QUESTION_TERMS + _OPEN_REQUEST_TERMS
        )
        extra_generic = extra_generic or any(term in request for term in _OPEN_REQUEST_TERMS)
    fields = tuple(field for field in _BOOK_QUESTION_TERMS if field in found)
    if fields:
        return BookQuestionIntent("field", fields, extra_generic)
    return BookQuestionIntent("generic" if any_generic else "none", (), any_generic)


def _book_requirement(tagged) -> str:
    constraint = tagged.constraint
    field, op, value = constraint.field, constraint.op, constraint.value
    if field == "genres" and op == "contains":
        return f"it should include {value}"
    if field == "languages" and op == "contains":
        return f"its language should include {value}"
    if field == "countries" and op == "contains":
        return f"its country of origin should include {value}"
    if field == "formats" and op == "contains":
        return f"it should be available as {value}"
    if field == "pages":
        relation = "at most" if op == "<=" else "at least"
        return f"it should be {relation} {value} pages"
    if field == "rating":
        relation = "at least" if op == ">=" else "at most"
        return f"its average rating should be {relation} {value}"
    if field == "publication_year":
        relation = "no earlier than" if op == ">=" else "no later than"
        return f"it should have been published {relation} {value}"
    return f"{field} must be {op} {value}"


_FIELD_NAMES = {
    "genres": "the genre", "languages": "the language", "countries": "the country of origin",
    "pages": "the length", "rating": "the rating", "publication_year": "the publication year",
    "formats": "the format",
}


class BookDisclosureController:
    """Deterministic fact ownership and reveal-state transitions for books."""

    def __init__(self, task: Task, decisions: list[dict] | None = None) -> None:
        self.task = task
        self.disclosed: set[int] = set()
        self.format_preference_disclosed = False
        self.opened = False
        # A UserSimulator shares its log so candidate verdicts interleave in order.
        self.decisions: list[dict] = decisions if decisions is not None else []

    def _record(
        self,
        action: str,
        *,
        intent: BookQuestionIntent,
        indices: list[int],
        policy_facts: list[str] | None = None,
        **details,
    ) -> None:
        self.decisions.append({
            "turn": len(self.decisions),
            "controller_version": BOOKS_CONTROLLER_VERSION,
            "action": action,
            "intent": intent.kind,
            "requested_fields": list(intent.fields),
            "disclosed_constraint_indices": indices,
            "disclosed_fields": [
                self.task.constraints[index].constraint.field for index in indices
            ],
            "disclosed_policy_facts": policy_facts or [],
            **details,
        })

    def _requirements(self, indices: list[int]) -> str:
        return "; and ".join(_book_requirement(self.task.constraints[index]) for index in indices)

    def _formats_sentence(self) -> str:
        return f"I can use any of these formats: {', '.join(self.task.user_formats)}."

    def respond(self, message: str) -> Message:
        if not self.opened:
            self.opened = True
            indices = [
                index for index, tagged in enumerate(self.task.constraints)
                if tagged.reveal == RevealTag.VOLUNTEER
            ]
            self.disclosed.update(indices)
            policy_facts = []
            requirements = self._requirements(indices)
            content = (
                f"I'm looking for one book. {requirements[:1].upper()}{requirements[1:]}."
                if requirements else "I'm looking for one book recommendation."
            )
            if self.task.reveal_difficulty == "volunteer" and self.task.user_formats:
                content += f" {self._formats_sentence()}"
                self.format_preference_disclosed = True
                policy_facts.append("accepted_formats")
            if "history" in self.task.policy_flags:
                content += " I want something I haven't already read."
                policy_facts.append("history_requirement")
            self._record(
                "opening",
                intent=BookQuestionIntent("opening"),
                indices=indices,
                policy_facts=policy_facts,
            )
            return Message(role=Role.USER, content=content)

        intent = classify_book_question(message)
        content = self._answer(intent)
        if content is None:
            self._record("no_disclosure", intent=intent, indices=[])
            content = "Please ask about anything you need to narrow down one book."
        return Message(role=Role.USER, content=content)

    def answer_embedded(self, message: str) -> str | None:
        """Answer questions asked alongside a candidate; ``None`` if nothing was asked."""
        if not self.opened:
            return None
        return self._answer(classify_book_question(message))

    def _answer(self, intent: BookQuestionIntent) -> str | None:
        """Answer a turn's questions field by field; ``None`` if it asked nothing.

        A field's stated facts are given, newly or again; a field holding a
        hidden fact gets "I'd rather not say" (more, if part of it was stated);
        a field with no requirement gets "No preference". A broad request also
        advances one undisclosed askable fact.
        """
        if intent.kind == "none":
            return None
        constraints = self.task.constraints
        stated: list[int] = []
        withheld: list[str] = []
        unset: list[str] = []
        formats = False
        for field in intent.fields:
            if field == "formats":
                if self.task.user_formats:
                    formats = True
                else:
                    unset.append(field)
                continue
            indices = [i for i, tagged in enumerate(constraints) if tagged.constraint.field == field]
            if not indices:
                unset.append(field)
                continue
            stated += [i for i in indices if constraints[i].reveal != RevealTag.HIDDEN]
            if any(constraints[i].reveal == RevealTag.HIDDEN for i in indices):
                withheld.append(field)
        if intent.generic:
            # A broad question advances one canonical fact, rather than
            # collapsing a multi-turn elicitation task into a single prompt.
            stated += [
                i for i, tagged in enumerate(constraints)
                if tagged.reveal == RevealTag.ON_ASK and i not in self.disclosed and i not in stated
            ][:1]

        sentences = []
        if stated:
            sentences.append(f"To answer your question: {self._requirements(stated)}.")
        if formats:
            sentences.append(self._formats_sentence())
        stated_fields = {constraints[i].constraint.field for i in stated}
        for field in withheld:
            more = " more" if field in stated_fields else ""
            sentences.append(f"I'd rather not say{more} about {_FIELD_NAMES[field]}.")
        if unset:
            sentences.append(f"No preference on {' or '.join(_FIELD_NAMES[field] for field in unset)}.")
        if not sentences:
            sentences.append("That's everything I can share.")
        elif len(sentences) == 1 and len(intent.fields) == 1 and not stated and not formats:
            sentences = ["I'd rather not say." if withheld else "No preference on that."]

        new = [i for i in stated if i not in self.disclosed]
        new_formats = formats and not self.format_preference_disclosed
        self._record(
            "disclose" if new or new_formats else "answer",
            intent=intent,
            indices=new,
            policy_facts=["accepted_formats"] if new_formats else [],
            restated_constraint_indices=[i for i in stated if i in self.disclosed],
            withheld_fields=withheld,
            no_preference_fields=unset,
        )
        self.disclosed.update(new)
        self.format_preference_disclosed = self.format_preference_disclosed or formats
        return " ".join(sentences)


def books_user_versions(interaction_mode: str) -> dict:
    """Version stamps of the books user a run is scored under."""
    versions = {"proposal_grounding": BOOKS_GROUNDING_VERSION}
    if interaction_mode == "symbolic":
        versions["disclosure_controller"] = BOOKS_CONTROLLER_VERSION
    else:
        versions["user_prompt"] = BOOKS_LLM_USER_VERSION
        versions["embedded_answers"] = BOOKS_LLM_EMBEDDED_ANSWERS_VERSION
    return versions


def request_parameter_manifest(
    spec: DomainSpec = MOVIES,
    mode: str = "tools",
    interaction_mode: str = "llm",
    provider_order: list[str] | None = None,
) -> dict:
    if interaction_mode == "symbolic":
        return {
            "temperature": "not applicable",
            "top_p": "not applicable",
            "seed": "not applicable",
            "timeout_seconds": "not applicable",
            "proposal_grounding": BOOKS_GROUNDING_VERSION,
            "disclosure_controller": BOOKS_CONTROLLER_VERSION,
            "rejection_feedback": "generic (no failed-field hint)",
        }
    manifest = {
        "temperature": 1.0,
        "top_p": "omitted (provider default)",
        "seed": "omitted (provider default)",
        "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    }
    if spec.id == "books" and mode == "tools":
        manifest["proposal_grounding"] = BOOKS_GROUNDING_VERSION
    from tau_rec.agents.litellm_agent import provider_routing

    routing = provider_routing(provider_order)
    if routing:
        manifest["provider_routing"] = routing
    return manifest

# {item_noun} is filled from a DomainSpec; the MOVIES rendering is pinned by
# tests/test_goldens.py.
BASE_GUIDELINES_SATISFIABLE = """\
You are simulating a real person looking for a {item_noun} recommendation.
Behave naturally. Do not be adversarial. Respond conversationally.

A recommendation means the assistant has committed to a specific {item_noun} — naming it
as THE pick for you, not offering multiple candidates or discussing possibilities.

When the assistant makes a recommendation:
- If it fits your preferences and constraints, respond with ###ACCEPTED###
- If it does not, respond with ###REJECTED### and briefly explain why.

If the assistant stalls, keeps asking clarifying questions without committing, or says
they cannot find a match, DO NOT accept. Push back: tell them to just pick their best
current option and commit to a single recommendation. Stay in the conversation — do
not emit ###ACCEPTED### until they actually name a specific {item_noun} as their pick.

If the assistant has named a specific title but hedges (e.g. "the closest match is X,
but I can't verify availability"), treat that as a concrete recommendation of X and
judge it on your constraints.

Follow the reveal rules below strictly.
"""

BASE_GUIDELINES_NO_VALID = """\
You are simulating a real person looking for a {item_noun} recommendation, but your
requirements are intentionally impossible — nothing in the catalog will satisfy
them. Behave naturally. Do not be adversarial.

When the assistant tells you no {item_noun} matches your criteria (either in plain words
or by declining to recommend anything), respond with ###ACCEPTED###.

If the assistant attempts to recommend a specific {item_noun} anyway, respond with
###REJECTED### and explain which requirement it fails.

Follow the reveal rules below strictly.
"""

BOOKS_USER_GUIDELINES = """\
You are simulating a real person looking for a book recommendation.
Behave naturally and keep your replies brief. Do not be adversarial.

The assistant may ask clarifying questions; answer them under the reveal rules
below. You do not judge specific books yourself: when the assistant proposes a
catalog candidate, the verdict is given separately.
Never write ###ACCEPTED### or ###REJECTED###.
"""

BOOKS_USER_FIDELITY = """\

## Constraint Fidelity
The persona is background context only. It does not create extra preferences.
Only the requirements in the Reveal Rules and the explicitly listed Soft Preferences,
if any, may affect what you ask for.

- Do not invent requirements about genre, author, language, country, page count, rating,
  publication year, format, series status, tone, themes, content, popularity, or anything else.
- Apply every operator literally. "At least" is a floor, not a ceiling; "at most" is a
  ceiling, not a floor; and "includes" does not prohibit additional values.
- Treat catalog metadata stated by the assistant as authoritative. Do not replace it
  with outside knowledge or assumptions about the title.
"""


def _build_books_simulator_prompt(task: Task) -> str:
    """The books LLM user (books-llm-user-v3).

    The same contract on every task, satisfiable or not, so the model cannot
    tell them apart; verdicts come from the harness. It states only its
    volunteer facts (and, in the volunteer variant, its formats) first, and
    answers like the symbolic user: askable facts when asked, "I'd rather not
    say" for hidden ones, "no preference" for anything else.
    """
    from tau_rec.domain import BOOKS

    sections = [BOOKS_USER_GUIDELINES, BOOKS_USER_FIDELITY, f"\n## Your Persona\n{task.persona}"]
    if task.soft_preferences:
        prefs = "\n".join(f"- {p}" for p in task.soft_preferences)
        sections.append(f"\n## Your Soft Preferences\n{prefs}")
    accepted = task.availability_values(BOOKS)
    if accepted:
        when = (
            "State them in your first message."
            if task.reveal_difficulty == "volunteer"
            else "Share them only if the assistant asks which formats you read."
        )
        formats = BOOKS.sim_availability_have.format(services=", ".join(accepted))
        sections.append(f"{formats} {when}")
    else:
        sections.append(
            f"{BOOKS.sim_availability_none} If the assistant asks which formats you read, "
            "say you have no preference on format."
        )

    groups: dict[RevealTag, list[str]] = {tag: [] for tag in RevealTag}
    for tc in task.constraints:
        groups[tc.reveal].append(_describe_constraint(tc, BOOKS))
    if "history" in task.policy_flags:
        groups[RevealTag.VOLUNTEER].append("You want a book you have not already read.")
    rules = ["\n## Reveal Rules"]
    headings = {
        RevealTag.VOLUNTEER: "**State in your first message** (your requirements for this conversation):",
        RevealTag.ON_ASK: (
            "**Share only if asked** (reveal one of these only when the assistant asks about its "
            "topic; never volunteer it, including in your first message):"
        ),
        RevealTag.HIDDEN: (
            "**Never share** (never mention, hint at, or confirm these, even if asked directly; if "
            "the assistant asks about their topic, say you'd rather not say):"
        ),
    }
    for tag in (RevealTag.VOLUNTEER, RevealTag.ON_ASK, RevealTag.HIDDEN):
        if groups[tag]:
            items = "\n".join(f"- {d}" for d in groups[tag])
            rules.append(f"\n{headings[tag]}\n{items}")
    rules.append(
        "\nYour first message states only the items under \"State in your first message\" (and your "
        "formats, if you are told to state them) with natural context from your persona: do not mention "
        "any other genre, language, country, publication year or format, and do not say that you have "
        "other requirements."
    )
    rules.append(
        "\nIf the assistant asks about a topic none of your requirements covers, say you have no "
        "preference on it. If a topic has requirements you may share and others you must not, "
        "share the first and say you'd rather not say more. If it asks again about something you "
        "already said, repeat it."
    )
    sections.append("\n".join(rules))
    return "\n".join(sections)


def _describe_constraint(tc, spec: DomainSpec = MOVIES) -> str:
    c = tc.constraint
    field = c.field
    if spec.id == "books":
        field = {
            "genres": "genre list",
            "languages": "language-of-work list",
            "countries": "country-of-origin list",
            "formats": "format list",
            "pages": "page count",
            "publication_year": "earliest catalog-edition year",
            "rating": "average rating",
        }.get(field, field.replace("_", " "))
        value = f'"{c.value}"' if isinstance(c.value, str) else c.value
        if c.op == ">=":
            direction = "later" if c.field == "publication_year" else "higher"
            return (
                f"{field} must be at least {value}; {value} or any {direction} "
                "value satisfies this requirement"
            )
        if c.op == "<=":
            direction = "earlier" if c.field == "publication_year" else "lower"
            return (
                f"{field} must be at most {value}; {value} or any {direction} "
                "value satisfies this requirement"
            )
        if c.op == "contains":
            return f"{field} must include {value}; additional values are allowed"
        if c.op == "contains_any":
            values = ", ".join(f'"{v}"' for v in c.value)
            return f"{field} must include at least one of [{values}]; additional values are allowed"
        if c.op == "not_contains":
            return f"{field} must not include {value}"
        if c.op == "in":
            values = ", ".join(f'"{v}"' for v in c.value)
            return f"{field} must equal one of [{values}]"
        if c.op == "==":
            return f"{field} must equal exactly {value}"
        if c.op == "!=":
            return f"{field} must not equal {value}"
    return f"{field} {c.op} {c.value}"


def build_simulator_prompt(task: Task, spec: DomainSpec = MOVIES) -> str:
    if spec.id == "books":
        return _build_books_simulator_prompt(task)
    base = BASE_GUIDELINES_NO_VALID if task.no_valid_recommendation else BASE_GUIDELINES_SATISFIABLE
    sections = [base.format(item_noun=spec.item_noun)]

    # Persona
    sections.append(f"\n## Your Persona\n{task.persona}")

    # Soft preferences
    if task.soft_preferences:
        prefs = "\n".join(f"- {p}" for p in task.soft_preferences)
        sections.append(f"\n## Your Soft Preferences\nUse these to judge recommendations:\n{prefs}")

    # Availability (share when asked). New domain-native tracks may use a
    # domain-specific task key; old tracks continue to load user_services.
    availability = task.availability_values(spec)
    if availability:
        svc_list = ", ".join(availability)
        sections.append(
            spec.sim_availability_have.format(services=svc_list, item_noun=spec.item_noun)
        )
    else:
        sections.append(spec.sim_availability_none)

    # Reveal rules
    volunteer, on_ask, hidden = [], [], []
    for tc in task.constraints:
        desc = _describe_constraint(tc, spec)
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
    def __init__(
        self,
        model: str,
        task: Task,
        spec: DomainSpec = MOVIES,
        catalog: Catalog | None = None,
        interaction_mode: str = "llm",
        provider_order: list[str] | None = None,
        harness: HarnessProfile = LEGACY,
    ) -> None:
        if interaction_mode not in SIMULATOR_INTERACTION_MODES:
            raise ValueError(f"unknown simulator interaction mode {interaction_mode!r}")
        if interaction_mode == "symbolic" and (spec.id != "books" or catalog is None):
            raise ValueError("symbolic simulator mode requires a books catalog")
        self.model = model
        self._harness = harness
        self._provider_order = list(provider_order) if provider_order else None
        self.task = task
        self.spec = spec
        self.interaction_mode = interaction_mode
        self._catalog = catalog
        self._call_log: list[dict] = []
        # Candidate verdicts in both modes, and the controller's turns in symbolic mode.
        self._decisions: list[dict] = []
        self.catalog_grounding_enabled = spec.id == "books" and catalog is not None
        self._controller = (
            BookDisclosureController(task, decisions=self._decisions)
            if interaction_mode == "symbolic" else None
        )
        self.system_prompt = build_simulator_prompt(task, spec)

    @property
    def decision_log(self) -> list[dict]:
        return [dict(decision) for decision in self._decisions]

    @property
    def call_log(self) -> list[dict]:
        return [dict(row) for row in self._call_log]

    def _grounded_candidate(self, message: str):
        """The one catalog candidate a turn names by marker, and how it was named."""
        markers = list(dict.fromkeys(_CANDIDATE_MARKER.findall(message)))
        truncated = len(_MARKER_START.findall(message)) > len(_CANDIDATE_MARKER.findall(message))
        if len(markers) > 1 or (markers and truncated):
            return None, "multiple"
        if truncated:
            return None, "invalid"
        if not markers:
            return None, "none"
        if markers[0].casefold() in {"none", "null"}:
            return None, "abstain"
        item = self._catalog.get(markers[0])
        return (item, "resolved") if item is not None else (None, "invalid")

    def _failed_checks(self, item, conversation: list[Message]) -> list[str]:
        """Why a candidate is not acceptable: its facts, the history, the formats."""
        failed = []
        if self.task.no_valid_recommendation or not all(
            tagged.constraint.evaluate(item) for tagged in self.task.constraints
        ):
            failed.append("constraints")
        if "history" in self.task.policy_flags:
            history = (self.task.user_history or {}).get(self.task.user_id or "", {})
            if item.id in history.get(self.spec.history_key, []):
                failed.append("already_read")
        formats = self.task.user_formats
        named = any(
            message.role == Role.USER
            and any(self.spec.mentions_service(message.content, value) for value in formats)
            for message in conversation
        )
        if named and not set(formats) & set(item.formats):
            failed.append("format")
        return failed

    def _grounded_response(self, conversation: list[Message]) -> Message | None:
        if not self.catalog_grounding_enabled:
            return None
        item, status = self._grounded_candidate(conversation[-1].content)
        if status == "none":
            return None
        failed: list[str] = []
        if status == "multiple":
            content = (
                "Please choose one catalog title as your current candidate and include "
                "only its candidate_id marker so I can respond to that single pick."
            )
        elif status == "invalid":
            content = "I couldn't identify that catalog candidate. Please check its item ID."
        elif status == "abstain":
            content = (
                "###REJECTED###\nI cannot verify whether the catalog has no match. "
                "If your search is exhaustive, follow the written policy and finalize; "
                "otherwise keep looking."
            )
        else:
            failed = self._failed_checks(item, conversation[:-1])
            # Feedback specificity is held fixed across symbolic and LLM modes.
            # Naming the failed check would alter disclosure as well as the
            # utterance generator and confound simulator comparisons.
            content = (
                "###ACCEPTED###\nThat book fits my requirements. Please make it your "
                "single final recommendation."
                if not failed else
                "###REJECTED###\nThat choice doesn't satisfy all of my requirements. "
                "Please try another single book."
            )
        verdict = (
            "accepted" if "###ACCEPTED###" in content
            else "rejected" if "###REJECTED###" in content else "clarification"
        )
        self._decisions.append({
            "turn": len(self._decisions),
            "controller_version": BOOKS_CONTROLLER_VERSION if self._controller else None,
            "grounding_version": BOOKS_GROUNDING_VERSION,
            "action": "candidate_verdict",
            "candidate_status": status,
            "candidate_id": item.id if item is not None else None,
            "verdict": verdict,
            "failed_checks": failed,
            "disclosed_constraint_indices": [],
            "disclosed_fields": [],
            "disclosed_policy_facts": [],
        })
        return Message(role=Role.USER, content=content)

    @staticmethod
    def _without_candidate_sentences(message: str) -> str:
        """Drop only the sentences that carry a candidate marker, keeping questions."""
        return "\n".join(
            sentence for sentence in _sentences(message) if not _MARKER_START.search(sentence)
        )

    async def respond(self, conversation: list[Message]) -> Message:
        if conversation and conversation[-1].role == Role.AGENT:
            grounded = self._grounded_response(conversation)
            if grounded is not None:
                question = self._without_candidate_sentences(conversation[-1].content)
                if self._controller is not None:
                    answer = self._controller.answer_embedded(question)
                else:
                    verdict = re.sub(r"###(?:ACCEPTED|REJECTED)###\s*", "", grounded.content)
                    answer = await self._embedded_llm_answer(
                        conversation[:-1], conversation[-1].content, question, verdict
                    )
                if answer is not None:
                    grounded = Message(
                        role=Role.USER, content=f"{grounded.content}\n{answer}"
                    )
                return grounded

        if self._controller is not None:
            message = conversation[-1].content if conversation else ""
            return self._controller.respond(message)

        content = await self._llm_reply(conversation)
        if self.catalog_grounding_enabled and (
            "###ACCEPTED###" in content or "###REJECTED###" in content
        ):
            # No grounded response was found above, so an acceptance/rejection
            # token here would be an LLM guess about catalog state. Never let a
            # guess steer the agent. Ask it to identify one candidate instead;
            # the next turn will be evaluated against typed predicates.
            content = (
                "Please identify one concrete catalog candidate and include its exact "
                "[[candidate_id:ID]] marker so I can assess that single book."
            )
        return Message(role=Role.USER, content=content)

    async def _embedded_llm_answer(
        self, history: list[Message], turn: str, question: str, verdict: str,
    ) -> str | None:
        """Let the LLM user answer a question asked alongside a candidate.

        The symbolic user answers such questions too, so both kinds of user see
        the same agent turn. The verdict stays deterministic: the model sees the
        whole turn and the verdict already given, and a guessed verdict is
        dropped.
        """
        if not _question_sentences(question):
            return None
        system_prompt = (
            f"{self.system_prompt}\n\n## Your Reply So Far\n"
            f"You have already answered the candidate in the assistant's last message: \"{verdict}\" "
            "Now reply only to the assistant's other questions, consistent with that answer and the "
            "reveal rules. Do not comment on the candidate again."
        )
        answer = await self._llm_reply(
            [*history, Message(role=Role.AGENT, content=turn)], system_prompt=system_prompt
        )
        self._call_log[-1]["purpose"] = BOOKS_LLM_EMBEDDED_ANSWERS_VERSION
        if "###ACCEPTED###" in answer or "###REJECTED###" in answer:
            return None
        return answer

    async def _llm_reply(self, conversation: list[Message], system_prompt: str | None = None) -> str:
        import litellm

        messages = [{"role": "system", "content": system_prompt or self.system_prompt}]
        for msg in conversation:
            role = "assistant" if msg.role == Role.USER else "user"
            messages.append({"role": role, "content": msg.content})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 1.0,
            "timeout": REQUEST_TIMEOUT_SECONDS,
        }
        if "deepseek" in self.model.lower():
            # litellm refuses a top-level `thinking` for unmapped OpenRouter models.
            if not self.model.startswith("openrouter/"):
                kwargs["thinking"] = {"type": "disabled"}
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        from tau_rec.agents.litellm_agent import provider_routing

        routing = provider_routing(self._provider_order)
        if routing and self.model.startswith("openrouter/"):
            kwargs.setdefault("extra_body", {})["provider"] = routing
        for attempt in range(MAX_RETRIES):
            started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
            call_start = time.perf_counter()
            try:
                from tau_rec.cost import check_provider_budget
                check_provider_budget()
                response = await timed_call(
                    litellm.acompletion(**kwargs), self._harness.call_timeout_seconds
                )
                from tau_rec.cost import call_cost, record_usage, response_usage
                usage = response_usage(response)
                content = response.choices[0].message.content
                empty = not (content and content.strip())
                # A token-free empty reply consumed nothing and often carries no
                # provider cost; recording it would trip the fail-closed ceiling.
                if not (empty and not any(usage.values())):
                    record_usage("sim", response)
                hidden_value = getattr(response, "_hidden_params", {}) or {}
                hidden = hidden_value if isinstance(hidden_value, dict) else {}
                self._call_log.append({
                    "attempt": attempt + 1,
                    "status": "ok",
                    "started_at": started_at,
                    "latency_seconds": time.perf_counter() - call_start,
                    "requested_model": self.model,
                    "response_model": getattr(response, "model", None),
                    "response_id": getattr(response, "id", None),
                    "system_fingerprint": (
                        getattr(response, "system_fingerprint", None)
                        or hidden.get("system_fingerprint")
                    ),
                    "provider": hidden.get("custom_llm_provider"),
                    "provider_model_id": hidden.get("model_id"),
                    "provider_order": self._provider_order,
                    "served_provider": getattr(response, "provider", None),
                    "request_parameters": {
                        "extra_body": kwargs.get("extra_body"),
                        "temperature": kwargs["temperature"],
                        "top_p": kwargs.get("top_p", "omitted"),
                        "seed": kwargs.get("seed", "omitted"),
                        "thinking": kwargs.get("thinking"),
                        "timeout_seconds": kwargs["timeout"],
                    },
                    "tokens": usage,
                    "provider_reported_cost_usd": hidden.get("response_cost"),
                    "estimated_cost_usd": call_cost(self.model, usage),
                })
                if not empty:
                    return content
                # Providers occasionally return no text (zero tokens, or reasoning
                # only). Resample the reply instead of failing the whole trial.
                self._call_log[-1]["status"] = "empty"
                await asyncio.sleep(2 * (attempt + 1))
                continue
            except CallTimeout as error:
                retrying = attempt < MAX_RETRIES - 1
                self._call_log.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "started_at": started_at,
                    "latency_seconds": time.perf_counter() - call_start,
                    "requested_model": self.model,
                    "error_type": "TimeoutError",
                    "error": str(error),
                    "retrying": retrying,
                })
                if not retrying:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
            except Exception as e:
                message = str(e).lower()
                response_decode = "unable to get json response" in message
                retryable = (
                    "503" in message or "429" in message
                    or "rate_limit" in message
                    or "overloaded" in message
                    or "unavailable" in message
                    or "connection" in message
                    or response_decode
                )
                retrying = attempt < MAX_RETRIES - 1 and retryable
                self._call_log.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "started_at": started_at,
                    "latency_seconds": time.perf_counter() - call_start,
                    "requested_model": self.model,
                    "error_type": type(e).__name__,
                    "error": str(e)[:300],
                    "retrying": retrying,
                })
                if retrying:
                    await asyncio.sleep(
                        2 * (attempt + 1)
                        if response_decode else RETRY_DELAY * (attempt + 1)
                    )
                else:
                    raise
        raise RuntimeError(f"user simulator returned no text in {MAX_RETRIES} attempts")
