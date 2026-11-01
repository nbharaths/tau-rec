"""When human seekers in ReDial first state attribute requirements (redial_disclosure.py)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "analysis/books-controlled/redial_disclosure.py"


def _load():
    spec = importlib.util.spec_from_file_location("redial_disclosure", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dialogue(messages):
    return {"initiatorWorkerId": 0, "respondentWorkerId": 1,
            "messages": [{"senderWorkerId": sender, "text": text} for sender, text in messages]}


def test_fields_are_detected_conservatively():
    fields = _load().fields
    assert fields("I love a good horror movie from the 80s") == {"genre", "era"}
    assert fields("anything with subtitles is fine, maybe Korean") == {"origin"}
    assert fields("something new to me") == set()


def test_first_mentions_are_classified_by_what_preceded_them():
    first_mentions = _load().first_mentions
    dialogue = _dialogue([
        (0, "Hi! I want a comedy"),                     # genre: unprompted
        (1, "Sure. Older or more recent?"),
        (0, "recent please"),                           # era: answered a question
        (1, "Try @123, it is great."),
        (0, "Hmm, I'd rather something Korean."),       # origin: after a suggestion
        (0, "and maybe a bit of comedy too"),           # genre again: not a first mention
    ])
    assert first_mentions(dialogue) == {"genre": "unprompted", "era": "answered_question",
                                        "origin": "after_suggestion"}


def test_greetings_are_not_questions_and_question_words_count_without_a_mark():
    first_mentions = _load().first_mentions
    greeting = _dialogue([(1, "Hi there, how are you?"), (0, "Good! I want a comedy")])
    assert first_mentions(greeting) == {"genre": "unprompted"}
    unmarked = _dialogue([(1, "what kinds of movies do you enjoy"), (0, "mostly horror")])
    assert first_mentions(unmarked) == {"genre": "answered_question"}
