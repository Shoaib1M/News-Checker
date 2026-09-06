"""
FILE PURPOSE:
Decide *what kind of question a claim actually poses* before spending any
time searching for evidence.

WHY THIS EXISTS:
The pipeline used to treat every submission identically: search, classify with
NLI, aggregate. That works for a well-formed past-tense factual claim — the
kind the LIAR dataset is full of — and produces a misleading result for
everything else:

  - "The US is going to ban Google across all its cities"
        A future action. No source on earth can make this true or false yet.
        The old pipeline searched, found nothing entailing it, and reported
        "insufficient evidence" — indistinguishable from "our search broke".

  - "asdkjh qwe zxc"
        Not a claim at all. The old pipeline still ran four search queries
        against it and reported "insufficient evidence".

  - "Elon Musk bought the Eiffel Tower"
        A fabricated but *highly newsworthy* claim. If it were true, every
        major outlet would have covered it. Silence is genuinely informative
        here — but only for claims of that kind, and only when the search
        actually worked.

This module produces the two facts the rest of the pipeline needs to say
something honest about those cases:

  kind      — what sort of proposition this is (checkable / prospective /
              opinion / not_a_claim)
  salience  — whether a true version of this claim would necessarily have
              been reported. Only "high" salience licenses treating an
              absence of coverage as meaningful (see evidence_aggregator.
              assess_coverage).

USED BY:
- main.py, to route the request and to label the result.
- evidence_aggregator.assess_coverage, which needs `salience`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ── Result ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ClaimTriage:
    """What kind of question this claim poses, decided before retrieval."""

    kind: str          # checkable | prospective | opinion | not_a_claim
    claim_type: str    # short human-readable label shown in the UI
    search_worthwhile: bool  # is there anything external evidence could settle?
    salience: str      # high | normal — see module docstring
    reason: str        # plain-English explanation shown to the user
    # True when the claim asserts that something did NOT happen. Absence of
    # coverage cannot count against such a claim — nobody reporting that the
    # US banned Google is consistent with "the US did not ban Google", not
    # evidence against it. See evidence_aggregator.assess_coverage.
    negated: bool = False

    @property
    def is_prospective(self) -> bool:
        return self.kind == "prospective"


# ── Vocabulary ───────────────────────────────────────────────────────
# Words that mark the claim as being about something that has not happened
# yet. A claim about the future is not false — it is not yet decidable.
_FUTURE_MARKERS = re.compile(
    r"\b(?:will\s+(?:soon\s+)?\w+|going\s+to|gonna|about\s+to|set\s+to|"
    r"due\s+to\s+\w+|plans?\s+to|planning\s+to|intends?\s+to|expected\s+to|"
    r"scheduled\s+to|by\s+(?:20[2-9]\d|next\s+(?:year|month|week))|"
    r"in\s+the\s+(?:coming|next)\s+(?:years?|months?|weeks?|days?))\b",
    re.IGNORECASE,
)

# When a future action is *attributed* to someone ("Apple announced it will
# …", "officials say they plan to …"), the claim is no longer purely about
# the future: whether the announcement happened is checkable today.
_ANNOUNCEMENT_MARKERS = re.compile(
    r"\b(?:announced|announces|said|says|stated|confirmed|confirms|"
    r"reportedly|according\s+to|unveiled|declared|pledged|promised|"
    r"proposed|proposes)\b",
    re.IGNORECASE,
)

# Superlatives and aesthetic words only signal an opinion in predicative
# position — "X is the best" is a value judgment, "the best-selling car in
# 2024 was the Model Y" is a checkable fact about sales figures. Matching the
# bare phrase flagged the second one as unverifiable.
_OPINION_MARKERS = re.compile(
    r"\b(?:i\s+(?:think|believe|feel)|in\s+my\s+opinion|arguably|"
    r"(?:is|are|was|were)\s+the\s+(?:best|worst|greatest)\b(?!-|\s*(?:selling|"
    r"known|paid|performing|rated|documented|recorded|attended))|"
    r"(?:is|are|was|were)\s+(?:so\s+|very\s+|really\s+)?"
    r"(?:beautiful|delicious|overrated|underrated|boring|amazing|awful)|"
    r"should\s+(?:be\s+)?(?:banned|allowed|stopped))\b",
    re.IGNORECASE,
)

# Verbs and scope words whose true occurrence would be covered by every major
# newsroom. This is the whole basis for absence-of-evidence reasoning, so it
# is deliberately a short list of unmistakably major events rather than a
# broad one — a false positive here would turn "we didn't find it" into an
# unearned "nobody reported it".
_HIGH_IMPACT_EVENTS = re.compile(
    r"\b(?:ban|bans|banned|banning|outlaw(?:s|ed|ing)?|"
    r"invad(?:e|es|ed|ing)|declare[sd]?\s+war|nuclear\s+(?:strike|attack|war)|"
    r"resign(?:s|ed|ing)?|step(?:s|ped)\s+down|impeach(?:es|ed|ment)?|"
    r"assassinat(?:e|es|ed|ion)|arrest(?:s|ed)?|indict(?:s|ed|ment)?|"
    r"die[sd]?|dead|killed|bankrupt(?:cy)?|collaps(?:e|es|ed)|"
    r"acquir(?:e|es|ed)|buy(?:s)?|bought|merge[sd]?|"
    r"legaliz(?:e|es|ed)|shut\s+down|dissolv(?:e|es|ed)|"
    r"secede[sd]?|renam(?:e|es|ed))\b",
    re.IGNORECASE,
)

# Absolute scope. "Google was fined in one state" is ordinary news; "Google is
# banned across all cities" could not happen quietly.
_ABSOLUTE_SCOPE = re.compile(
    r"\b(?:all|every|entire|nationwide|worldwide|globally|countrywide|"
    r"across\s+the\s+(?:country|world)|completely|totally|permanently)\b",
    re.IGNORECASE,
)

# Copulas and auxiliaries. A string with none of these and no other verb is
# a noun phrase, not an assertion.
_NEGATION = re.compile(
    r"\b(?:not|never|no|didn't|did\s+not|doesn't|does\s+not|isn't|is\s+not|"
    r"wasn't|was\s+not|weren't|were\s+not|won't|will\s+not|hasn't|has\s+not|"
    r"haven't|have\s+not|cannot|can't|denied|denies|refuted|false)\b",
    re.IGNORECASE,
)

_COPULA = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|do|does|did|"
    r"will|would|can|could|may|might|must|should)\b",
    re.IGNORECASE,
)

# ── Language scope ───────────────────────────────────────────────────
# Every stage of this pipeline is English-only: the NLI model, the event
# vocabulary, the abbreviation and demonym tables, and the providers, which
# are queried with lang=en. A Hindi or Spanish claim therefore cannot be
# checked — but it used to be reported as "no verifiable claim found", which
# tells the user their claim was unintelligible rather than out of scope.
# Those are very different messages to receive.
_NON_LATIN_LETTER = re.compile(
    r"[\u0370-\u03ff\u0400-\u04ff\u0530-\u058f\u0590-\u05ff"
    r"\u0600-\u06ff\u0700-\u074f\u0900-\u097f\u0980-\u09ff"
    r"\u0a00-\u0d7f\u0e00-\u0e7f\u1000-\u109f\u3040-\u30ff"
    r"\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]"
)

# Function words that are distinctly English. Words that double as function
# words elsewhere are deliberately absent — "a" is a French verb form, "in"
# and "on" are German and Dutch, "no" is Spanish — because a false English
# marker is what lets a foreign claim through to be reported as gibberish.
_ENGLISH_MARKERS = frozenset({
    "the", "is", "are", "was", "were", "of", "to", "and", "for", "with",
    "that", "has", "have", "had", "will", "from", "this", "these", "been",
    "after", "before", "about", "than", "which", "would", "could", "should",
})

# Function words common in other Latin-script languages and rare in English.
# Short words that collide with English are deliberately absent: Portuguese
# "as" and "os", Spanish "no" and "se", Italian "lo". Leaving "as" in flagged
# "Angela Merkel resigned as chancellor" as foreign.
_FOREIGN_MARKERS = frozenset({
    "el", "la", "los", "las", "una", "del", "que", "por", "para", "con",
    "esta", "este", "muy", "pero", "sus",
    "le", "les", "des", "du", "et", "est", "sont", "dans", "pour", "sur",
    "ce", "cette", "qui", "avec", "mais", "leur", "aux", "nous", "vous",
    "ils", "elle", "cela",
    "der", "die", "das", "den", "dem", "und", "ist", "sind", "eine",
    "nicht", "auch", "wird", "heute", "sich",
    "gli", "che", "non", "sono", "nel", "alla",
    "uma", "nao", "com", "pelo", "seu",
})

# Two independent foreign markers, not one. A single hit is too easy to score
# on a proper noun or a borrowed word.
_MIN_FOREIGN_MARKERS = 2


def _language_looks_english(text: str, tokens: list[str]) -> bool:
    """Heuristic scope check — is this English enough to attempt?

    Deliberately permissive: it reports "not English" only when the evidence
    is clear, because refusing a real English claim is worse than attempting a
    borderline one.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if letters:
        non_latin = sum(1 for ch in letters if _NON_LATIN_LETTER.match(ch))
        if non_latin / len(letters) > 0.3:
            return False

    lowered = {token.lower() for token in tokens}
    if lowered & _ENGLISH_MARKERS:
        return True
    return len(lowered & _FOREIGN_MARKERS) < _MIN_FOREIGN_MARKERS


_VERB_SHAPED = re.compile(r"\w+(?:s|ed|ing)$", re.IGNORECASE)

# The -s/-ed/-ing shape test misses every irregular past tense, so "The
# greatest earthquake in history hit Chile in 1960" was rejected as "not an
# assertion" — it contains no copula and not one of its words ends in -ed.
# Erring toward admitting a claim is the safe direction here: a false
# positive gets searched, a false negative refuses to check a real claim.
_IRREGULAR_VERBS = frozenset({
    "hit", "won", "lost", "met", "went", "gave", "got", "saw", "came", "took",
    "made", "held", "told", "left", "sold", "put", "cut", "set", "ran", "began",
    "broke", "brought", "bought", "built", "chose", "drove", "fell", "felt",
    "found", "flew", "forgot", "grew", "kept", "knew", "led", "paid", "read",
    "rose", "sent", "shot", "shut", "spent", "stood", "struck", "swore",
    "thought", "threw", "wore", "wrote", "became", "began", "beat", "bore",
    "dealt", "drew", "fought", "hid", "hung", "quit", "rode", "sank", "sat",
    "slept", "spoke", "spread", "stole", "swept", "tore", "withdrew",
})


def _wordlike(token: str) -> bool:
    """True for tokens that plausibly belong to a natural language.

    Keystroke mash ("asdkjh", "qwerty", "zxcv") has no vowels or improbable
    consonant runs. This is a shape test, not a dictionary — real words the
    dictionary would miss (names, jargon, transliterations) still pass.
    """
    lowered = token.lower()
    if not lowered.isalpha():
        return True  # numbers and punctuation are not evidence of gibberish
    if len(lowered) <= 2:
        return True
    if not re.search(r"[aeiouy]", lowered):
        return False
    return not re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", lowered)


def triage_claim(statement: str) -> ClaimTriage:
    """Classify a submission before any search work is done.

    The order matters: a string that is not a claim at all should never be
    scored for salience, and a subjective statement should never be sent to
    a retrieval pipeline that can only find articles, not settle taste.
    """
    text = statement.strip()

    # A pasted link carries no proposition of its own. Tokenising it would
    # otherwise yield "https example com article" — four word-shaped tokens
    # ending in a plural-looking "s", enough to pass every check below.
    if re.fullmatch(r"\s*https?://\S+\s*", text):
        return ClaimTriage(
            kind="not_a_claim",
            claim_type="link, not a claim",
            search_worthwhile=False,
            salience="normal",
            reason=(
                "That is a link, not a claim. Paste the specific sentence you "
                "want checked and it can be compared against sources."
            ),
        )

    tokens = re.findall(r"[A-Za-z0-9']+", text)

    # ── 1. Is this a claim at all? ───────────────────────────────────
    if text.endswith("?"):
        return ClaimTriage(
            kind="not_a_claim",
            claim_type="question",
            search_worthwhile=False,
            salience="normal",
            reason=(
                "This is a question, not a claim. Enter it as a statement "
                "(\"X did Y\") and it can be checked against evidence."
            ),
        )

    # Checked before the length test: a Devanagari or Arabic claim produces
    # zero ASCII tokens, and would otherwise be rejected as "too short".
    if not _language_looks_english(text, tokens):
        return ClaimTriage(
            kind="not_a_claim",
            claim_type="unsupported language",
            search_worthwhile=False,
            salience="normal",
            reason=(
                "This checker only handles English. The claim itself may well "
                "be checkable — the language is the limitation here, not the "
                "claim. Try submitting it in English."
            ),
        )

    if len(tokens) < 4:
        return ClaimTriage(
            kind="not_a_claim",
            claim_type="incomplete claim",
            search_worthwhile=False,
            salience="normal",
            reason=(
                "This is too short to contain a checkable assertion — there is "
                "no subject and action to verify."
            ),
        )

    alpha_tokens = [t for t in tokens if t.isalpha()]
    if alpha_tokens and sum(_wordlike(t) for t in alpha_tokens) < len(alpha_tokens) * 0.6:
        return ClaimTriage(
            kind="not_a_claim",
            claim_type="unrecognized text",
            search_worthwhile=False,
            salience="normal",
            reason=(
                "This does not parse as a sentence, so there is no proposition "
                "to check. Nothing was searched."
            ),
        )

    lowered_tokens = {t.lower() for t in tokens}
    has_verb = (
        bool(_COPULA.search(text))
        or bool(lowered_tokens & _IRREGULAR_VERBS)
        or any(_VERB_SHAPED.match(t) and len(t) > 3 for t in tokens)
    )
    if not has_verb:
        return ClaimTriage(
            kind="not_a_claim",
            claim_type="not an assertion",
            search_worthwhile=False,
            salience="normal",
            reason=(
                "No assertion was found — this reads as a topic or phrase "
                "rather than a statement that can be true or false."
            ),
        )

    # ── 2. Is it a matter of opinion? ────────────────────────────────
    if _OPINION_MARKERS.search(text):
        return ClaimTriage(
            kind="opinion",
            claim_type="subjective",
            search_worthwhile=False,
            salience="normal",
            reason=(
                "This is a value judgment. Evidence can inform it but cannot "
                "make it true or false, so no verdict is given."
            ),
        )

    # ── 3. Salience: would a true version of this have been reported? ─
    negated = bool(_NEGATION.search(text))
    salience = "normal"
    if _HIGH_IMPACT_EVENTS.search(text) and (
        _ABSOLUTE_SCOPE.search(text) or len(tokens) <= 20
    ):
        salience = "high"

    # ── 4. Future vs. already-happened ───────────────────────────────
    if _FUTURE_MARKERS.search(text) and not _ANNOUNCEMENT_MARKERS.search(text):
        return ClaimTriage(
            kind="prospective",
            claim_type="claim about a future event",
            # Still worth searching: the *plan* may have been announced even
            # though the event itself has not happened.
            search_worthwhile=True,
            salience=salience,
            negated=negated,
            reason=(
                "This describes something that has not happened yet, so it "
                "cannot be true or false today. What can be checked is whether "
                "any credible source reports it as planned or announced."
            ),
        )

    return ClaimTriage(
        kind="checkable",
        claim_type="factual claim",
        search_worthwhile=True,
        salience=salience,
        negated=negated,
        reason="",
    )


# ── Manual check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "The United States is Going to ban Google across all its cities",
        "Elon Musk bought the Eiffel Tower for 3 trillion dollars",
        "Apple announced it will move all manufacturing to India by 2027",
        "The prime minister of India resigned this morning",
        "Water freezes at 0°C at sea level",
        "asdkjh asdkjh asdkjh qwe",
        "Is the earth flat?",
        "Pizza is the best food",
        "unemployment rate",
    ]
    for sample in samples:
        result = triage_claim(sample)
        print(f"{result.kind:<13} {result.salience:<7} {sample}")
