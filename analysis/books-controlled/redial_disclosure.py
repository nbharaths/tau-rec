#!/usr/bin/env python3
"""When human seekers in ReDial first state attribute requirements (post hoc, external corpus).

ReDial (Li et al., 2018; CC BY 4.0) holds crowdsourced movie-recommendation
dialogues between a seeker (the initiator) and a recommender. A conservative
lexicon detects three requirement fields analogous to the books track's:
genre, era (decade, year, old/classic/recent), and origin (language, country,
subtitles). For each field a seeker mentions, its first mention is classified
by the recommender's last message before it: none yet or neither of the
below (unprompted), a question other than a greeting (answered_question), or a
message suggesting a movie (after_suggestion). The corpus is not redistributed; pass its path.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

LEXICON = {
    "genre": r"\b(comed(y|ies)|horror|action|drama|romance|romantic|rom-?com|thriller|sci-?fi|science fiction|"
             r"animated|animation|cartoon|documentar(y|ies)|musical|mystery|fantasy|western|crime|superhero|"
             r"adventure|suspense|scary|war movie|family movie|kids movie|disney|chick flick|biopic|anime)\b",
    "era": r"\b((19|20)\d\d|\d0'?s|older|oldies|old movies?|classics?|classic movies?|recent|newer|latest|"
           r"modern|decade|this year|last year)\b",
    "origin": r"\b(foreign|subtitles?|subtitled|french|spanish|korean|japanese|chinese|indian|bollywood|"
              r"british|german|italian|swedish|mexican|hindi|language)\b",
}
PATTERNS = {field: re.compile(pattern, re.IGNORECASE) for field, pattern in LEXICON.items()}
MOVIE = re.compile(r"@\d+")
GREETING = re.compile(r"(how are you|how's it going|how are things|how is your day|how's your day|how you doing)"
                      r"[^?.!]*\?*", re.IGNORECASE)
QUESTION_WORDS = re.compile(r"\b(what|which|do you|are you|would you|any (particular|preference|genre)|tell me|"
                            r"looking for|in the mood)\b", re.IGNORECASE)


def is_question(text: str) -> bool:
    """A recommender turn that asks the seeker something, other than a greeting."""
    rest = GREETING.sub(" ", text)
    return "?" in rest or bool(QUESTION_WORDS.search(rest))


def fields(text: str) -> set[str]:
    return {field for field, pattern in PATTERNS.items() if pattern.search(MOVIE.sub(" ", text))}


def first_mentions(dialogue: dict) -> dict[str, str]:
    seeker = dialogue["initiatorWorkerId"]
    last_recommender, out = None, {}
    for message in dialogue["messages"]:
        text = message.get("text") or ""
        if message["senderWorkerId"] != seeker:
            last_recommender = text
            continue
        for field in sorted(fields(text) - set(out)):
            if last_recommender is not None and is_question(last_recommender):
                out[field] = "answered_question"
            elif last_recommender is not None and MOVIE.search(last_recommender):
                out[field] = "after_suggestion"
            else:
                out[field] = "unprompted"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redial", type=Path, required=True, help="directory with train_data.jsonl and test_data.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dialogues = [json.loads(line) for name in ("train_data.jsonl", "test_data.jsonl")
                 for line in (args.redial / name).read_text().splitlines() if line.strip()]
    per_dialogue = [first_mentions(d) for d in dialogues]
    with_requirement = [m for m in per_dialogue if m]
    kinds = Counter(kind for m in with_requirement for kind in m.values())
    total = sum(kinds.values())
    by_field = {field: dict(sorted(Counter(m[field] for m in with_requirement if field in m).items()))
                for field in LEXICON}
    out = {
        "status": "post hoc; external corpus; lexicon-based",
        "dialogues": len(dialogues),
        "dialogues_with_a_requirement": len(with_requirement),
        "first_mentions": total,
        "share_of_first_mentions": {kind: round(n / total, 4) for kind, n in sorted(kinds.items())},
        "dialogues_with_a_requirement_after_opening_share": round(
            sum(any(k != "unprompted" for k in m.values()) for m in with_requirement) / len(with_requirement), 4),
        "dialogues_with_a_requirement_only_after_a_suggestion_share": round(
            sum("after_suggestion" in m.values() for m in with_requirement) / len(with_requirement), 4),
        "by_field": by_field,
        "lexicon": LEXICON,
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "lexicon"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
