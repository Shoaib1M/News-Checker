"""
FILE PURPOSE:
Turn what a user actually pastes into the proposition they are asking about,
before any other stage looks at it.

WHY THIS EXISTS:
Every stage downstream — triage, entity extraction, query generation, and NLI,
which uses the claim as its hypothesis — was reading the raw submission. People
do not submit propositions. They submit what they saw, with the framing they
saw it in:

    is it true that the prime minister of india resigned?
    BREAKING: The United States has banned Google across all its cities!!!
    https://twitter.com/x/status/123 The prime minister of India resigned
    "Google banned in US" - Reuters, March 2024
    🚨🚨 GOOGLE BANNED IN ALL US CITIES 🚨🚨 #breaking #news
    so i heard that google got banned in the us??? can someone confirm

Each of those broke something measurable:

  - The FIRST one was triaged `not_a_claim` and never searched at all. It is
    the most natural way there is to ask a fact-checker a question, and the
    system answered "no verifiable claim found".
  - The URL made "Twitter" a primary entity, and a dispatched query became
    "India Twitter resigned".
  - The attribution made "Reuters" and "March" entities, and searched for
    "Google Reuters banned".
  - The first dispatched query is the submission itself, so the emoji, the
    hashtags and the URL were sent verbatim to a news index.

WHAT IS DELIBERATELY NOT DONE HERE:
No rewriting, no paraphrase, no spelling correction, no case normalisation of
the claim's own words. This removes framing and packaging; it must never change
what is being asserted. Every transformation below is reversible in the sense
that matters — the user's original text is what the UI shows, and what history
stores. The normalised form exists for the machinery.

Negation, hedging and quantifiers are left completely alone: "not", "all",
"only", "may" change the proposition, and a normaliser that touched them would
be answering a different question from the one asked.
"""

from __future__ import annotations

import re
import unicodedata

# Anything with a scheme, plus bare domains people paste from an address bar.
_URL = re.compile(r"""(?xi)
    \b(?:https?://|www\.)\S+
    | \b[a-z0-9-]+\.(?:com|org|net|co\.uk|io|gov|edu|in)/\S*
""")

# Wire-service and social attention markers, only at the very start.
_LEADING_LABEL = re.compile(r"""(?xi)^\s*(?:
    breaking(?:\s+news)? | just\s+in | urgent | developing | exclusive
    | update | alert | live | watch | video | opinion | analysis
)\s*[:\-–—]\s*""")

# Conversational framing. The proposition is what follows.
#
# These are matched only at the START, and each requires the filler word that
# makes it framing rather than content: "is it true that X" is a question about
# X, while "it is true that X" is an assertion of X and is left alone.
_LEADING_FRAME = re.compile(r"""(?xi)^\s*(?:
      (?:is|was)\s+it\s+true\s+that
    | (?:did|does|has|have|is|are|was|were)\s+(?:you\s+)?(?:hear|know)\s+(?:that|if|whether)
    | (?:i|we)\s+(?:just\s+)?(?:heard|read|saw)\s+(?:that|somewhere\s+that)
    | so\s+(?:i|we)\s+(?:just\s+)?(?:heard|read|saw)\s+(?:that)?
    | (?:someone|somebody)\s+(?:told\s+me|said)\s+(?:that)?
    | (?:apparently|allegedly|reportedly|supposedly)
    | (?:can|could)\s+(?:someone|somebody|you)\s+(?:please\s+)?
      (?:confirm|verify|check|fact[\s-]?check)\s+(?:that|if|whether)
    | (?:please\s+)?(?:fact[\s-]?check|verify|check)\s*(?:that|this|if|whether)?
    | true\s+or\s+false\s*[:?]?
)\s*[:,]?\s+""")

# The same wrappers when they trail the claim. The optional separator matters:
# "google got banned in the us, can someone confirm" left a dangling comma
# without it, and that comma then travels into every dispatched query.
_TRAILING_FRAME = re.compile(r"""(?xi)\s*[,;:–—-]?\s*(?:
      can\s+(?:someone|somebody|anyone|you)\s+(?:please\s+)?
      (?:confirm|verify|check|fact[\s-]?check)(?:\s+this|\s+that)?
    | is\s+(?:this|that)\s+(?:true|real|fake|correct)
    | true\s+or\s+false
    | any(?:one)?\s+know
    | thoughts
)\s*[?.!]*\s*$""")

# A source credit appended to a quoted headline: "…" - Reuters, March 2024
_TRAILING_ATTRIBUTION = re.compile(
    r"""\s*[-–—]\s*[A-Z][\w.&' ]{1,40}"""          # - Reuters
    r"""(?:\s*,\s*[A-Z][a-z]+\.?(?:\s+\d{1,2})?(?:\s*,?\s*\d{4})?)?"""  # , March 2024
    r"""\s*$"""
)

_HASHTAG = re.compile(r"(?:^|\s)#\w+")
_HANDLE = re.compile(r"(?:^|\s)@\w+")
_VIA = re.compile(r"(?i)\s*\bvia\s+@?\w+\s*$")

# Repeated terminal punctuation: "resigned???" and "cities!!!" are emphasis.
_REPEATED_PUNCT = re.compile(r"([!?.])\1+")
_MULTISPACE = re.compile(r"\s+")

# Wrapping quotes around the whole submission, of the several kinds keyboards
# and copy-paste produce.
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"),
                ("‘", "’"), ("«", "»"))


def _strip_emoji(text: str) -> str:
    """Drop symbol and pictograph characters, keeping all letters and marks.

    Unicode categories rather than a range list: So/Sk/Cs covers emoji,
    dingbats and flags without touching accented letters or non-Latin scripts,
    which a hand-written range list reliably gets wrong.
    """
    return "".join(
        char for char in text
        if unicodedata.category(char) not in {"So", "Sk", "Cs", "Cf"}
    )


def _strip_wrapping_quotes(text: str) -> str:
    for opening, closing in _QUOTE_PAIRS:
        if len(text) > 2 and text.startswith(opening) and text.endswith(closing):
            return text[1:-1].strip()
    return text


def normalize_claim(raw: str) -> str:
    """The proposition inside a submission, or the submission unchanged.

    Never returns empty: if stripping the packaging would leave nothing, the
    packaging was the submission, and the caller must see it as it was so
    triage can say so.
    """
    if not raw:
        return raw

    text = unicodedata.normalize("NFKC", raw).strip()
    text = _strip_emoji(text)
    text = _URL.sub(" ", text)
    text = _HASHTAG.sub(" ", text)
    text = _HANDLE.sub(" ", text)
    text = _VIA.sub("", text)
    text = _MULTISPACE.sub(" ", text).strip()

    # Framing can nest — "so i heard that apparently X" — but only a couple of
    # layers deep in real submissions, and an unbounded loop on a regex that
    # can match empty-ish input is how a normaliser hangs.
    for _ in range(3):
        before = text
        text = _LEADING_LABEL.sub("", text)
        text = _LEADING_FRAME.sub("", text)
        text = _TRAILING_FRAME.sub("", text)
        text = text.strip()
        if text == before:
            break

    text = _strip_wrapping_quotes(text)
    text = _TRAILING_ATTRIBUTION.sub("", text).strip()
    text = _strip_wrapping_quotes(text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    text = _MULTISPACE.sub(" ", text).strip()
    # Punctuation left dangling by a clause that was removed from either end.
    text = text.strip(" ,;:-–—")

    # A question mark left on a proposition ("the PM resigned?") is the user's
    # uncertainty, not part of the claim. A question WORD at the front means
    # this is a genuine question and triage must still see it as one.
    if text.endswith("?") and not re.match(
        r"(?i)^(?:who|what|when|where|why|how|which|do|does|did|is|are|was|"
        r"were|can|could|should|will|would|has|have|had)\b", text
    ):
        text = text[:-1].strip()

    return text or raw.strip()
