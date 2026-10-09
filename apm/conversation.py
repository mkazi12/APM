"""Local turn-taking hints, without another inference or an open-ended listen.

This conservative English heuristic identifies invitations to answer, not every
possible question or rhetorical expression. Tool results take precedence over
catalog prose, whose song titles and device names may themselves be questions.
"""
from collections.abc import Mapping
import re


class AssistantReply(str):
    """A normal spoken/displayed string with an explicit turn-taking hint."""

    def __new__(cls, text, *, expects_reply=False):
        if not isinstance(text, str) or not isinstance(expects_reply, bool):
            raise TypeError("AssistantReply requires text and a boolean expects_reply")
        reply = super().__new__(cls, text)
        reply.expects_reply = expects_reply
        return reply


_MUSIC_TOOLS = frozenset({"resolve_music", "play_music", "play_music_selection"})
_QUOTED = re.compile(
    r'```[\s\S]*?```|`[^`\n]*`|"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|'
    r"(?<!\w)'(?:[^'\n]|(?<=\w)'(?=\w))*'(?!\w)"
)
_LEADING = re.compile(r"^\s*(?:[-*#]+\s*|\d+[.)]\s*)")
_COURTESY = re.compile(r"^(?:(?:okay|ok|sure|alright|certainly|sorry|hi|hello)[,!:]?\s+)+")
_WH_QUESTION = re.compile(
    r"^(?:which|whose|whom)\s+\w+|"
    r"^(?:where|when|why)\s+(?:do|does|did|is|are|was|were|should|would|will|"
    r"shall|can|could|have|has|to|not|exactly)\b|"
    r"^who\s+(?:is|are|was|were|do|does|did|would|should|can|could|will|shall)\b|"
    r"^what\s+(?!(?:a|an|i|we|you|he|she|it|they)\b)\w+|"
    r"^how\s+(?:can|could|may|do|does|did|are|is|was|were|would|should|will|"
    r"have|has|long|many|much|often|soon|about)\b"
)
_YES_NO_QUESTION = re.compile(
    r"^(?:can|could|would|will|shall|should|do|does|did|are|is|was|were|have|has|may)\s+"
    r"(?:you|i|we|it|that|this|there|the|your|these|those)\b"
)
_ELICITATION = re.compile(
    r"^(?:please\s+)?(?:choose|select|specify|clarify|confirm|repeat|provide)\b|"
    r"^(?:please\s+)?tell me\b|"
    r"^(?:please\s+)?let me know\s+(?:what|which|when|where|who|how|whether|if)\b|"
    r"^(?:i need|i'd need|i would need)\s+(?:you to|more (?:details|information)|"
    r"a (?:time|date|name|location)|the (?:time|date|name|location))\b|"
    r"^(?:go ahead|i'm listening|i am listening|anything else|what else)\b"
)
_STATEMENT = re.compile(
    r"^(?:playing|now playing|found|playback|song|track|title|artist|album|"
    r"the (?:song|track|title|artist|album)|i (?:played|found|said|asked|heard)|"
    r"you (?:said|asked)|for example|example)\b"
)
_CLOSING = re.compile(r"^(?:please\s+)?let me know if you (?:need|want|have)\b")


def reply_needs_answer(text, results=None):
    """Whether this delivered reply warrants one bounded follow-up window.

    Pass the final spoken text and the executed tool result list. A nonempty
    result list is authoritative: currently only successful music ambiguity
    asks for a choice. Other tool outcomes do not request a spoken answer.
    Without tool results, recognize questions and explicit elicitation in the
    model's prose, ignoring quoted titles and code. Missing '?' is supported.
    """
    if results:
        return any(isinstance(result, Mapping)
                   and result.get("ok") is True
                   and result.get("tool") in _MUSIC_TOOLS
                   and isinstance(result.get("data"), Mapping)
                   and result["data"].get("status") == "ambiguous"
                   for result in results)
    if not isinstance(text, str) or not text.strip():
        return False
    visible = _QUOTED.sub(" ", text).replace("’", "'").casefold()
    visible = re.sub(r"\b(what|where|when|who|why|how)'s\b", r"\1 is", visible)
    for match in re.finditer(r"[^.!?\n]+[.!?]?", visible):
        sentence = _LEADING.sub("", match.group()).strip()
        sentence = _COURTESY.sub("", sentence)
        if not sentence or _STATEMENT.match(sentence) or _CLOSING.match(sentence):
            continue
        if (_WH_QUESTION.match(sentence) or _YES_NO_QUESTION.match(sentence)
                or _ELICITATION.match(sentence)):
            return True
        # Short elliptical questions such as "Tomorrow morning?" or
        # "Kitchen or bedroom?" need no auxiliary verb. Require a final question
        # so punctuation inside an unquoted title followed by metadata is inert.
        if sentence.endswith("?") and not visible[match.end():].strip() and len(sentence.split()) <= 8:
            return True
    return False
