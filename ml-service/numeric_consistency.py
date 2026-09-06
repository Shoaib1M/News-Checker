"""
FILE PURPOSE:
Decide whether a document actually states the quantity a claim asserts, or a
different one — independently of the NLI model.

WHY THIS EXISTS:
"The vaccine is 95% effective" and "the vaccine is 62% effective" are the same
sentence apart from two digits. Every stage before this one is built on word
overlap: the entities match, the action matches, the phrasing matches, and the
sentence is a near-perfect lexical neighbour of the claim. Textual entailment
models are trained on exactly that resemblance and are known to be unreliable
on the numbers inside it, so nothing in the pipeline was in a position to
notice that the document says something different from the claim.

The result is the worst failure this system can produce: a real article, from a
real publisher, at full source weight, recorded as CONFIRMING a figure it
contradicts.

WHAT THIS DOES *NOT* DO:
It never turns a document into evidence AGAINST a claim. Two figures can differ
because they measure different things — "62% against severe disease" is not a
refutation of "95% effective overall" — and this module cannot tell those apart.
So a conflict here only withdraws support: the honest reading is "this document
is about the claim but does not state the figure it asserts".

HOW A CONFLICT IS ESTABLISHED (all three must hold):
  1. The claim asserts a quantity.
  2. No passage from the document states that quantity.
  3. Some passage states a DIFFERENT quantity, of the same kind, describing
     the same attribute.

Condition 3 is what keeps the guard off claims where the number is incidental.
"Musk bought the Eiffel Tower for 3 trillion dollars" against an article
mentioning a 400 billion dollar net worth is not a conflict: both are currency
amounts, but "net worth" is not the attribute the claim's figure describes, and
an article confirming the purchase must not be discarded over the price.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from article_extractor import _PASSAGE_STOPWORDS

# How far either side of a number to look for the attribute it describes.
# Four tokens covers "the vaccine is 95% effective" and "unemployment rose to
# 4.2% in March" without reaching into the next clause.
ATTRIBUTE_WINDOW = 4

# Two figures count as the same measurement within this relative distance, so
# that "nearly 95%" reported as 94.8% is not a conflict.
#
# A consequence worth stating: four-digit years are within 2% of each other for
# any pair in the same era (2035 and 2050 differ by 0.7%), so years never
# register as conflicting. That is deliberate. A date needs to be compared
# against the article's own timeline, not string-matched, and treating "by
# 2050" against "by 2035" as a conflict here would withdraw support from
# articles on the strength of any year they happened to mention.
RELATIVE_TOLERANCE = 0.02

_MAGNITUDES = {
    "hundred": 1e2, "thousand": 1e3, "million": 1e6, "billion": 1e9,
    "trillion": 1e12, "bn": 1e9, "tn": 1e12,
}

# Deliberately no bare "m" or "k": "3 m" is three metres far more often than
# three million, and a wrong unit here produces a wrong conflict.
_QUANTITY = re.compile(
    r"(?P<symbol>[$£€])?\s*"
    r"(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<magnitude>trillion|billion|million|thousand|hundred|bn|tn)?\s*"
    r"(?P<unit>%|percent|percentage points?|dollars?|euros?|pounds?|usd|eur|gbp)?",
    re.IGNORECASE,
)

_CURRENCY_WORDS = {"dollar", "dollars", "euro", "euros", "pound", "pounds",
                   "usd", "eur", "gbp"}
_PERCENT_WORDS = {"%", "percent", "percentage point", "percentage points"}

# Words that describe the number's shape rather than what it measures.
_NON_ATTRIBUTE = _PASSAGE_STOPWORDS | set(_MAGNITUDES) | _CURRENCY_WORDS | {
    "percent", "percentage", "point", "points", "roughly", "about",
    "approximately", "nearly", "almost", "around", "least", "most", "some",
    "up", "down", "just", "only", "estimated", "reported",
}


@dataclass(frozen=True)
class Quantity:
    """One figure written in a piece of text, and what it measures."""
    value: float
    kind: str                    # percent | currency | count
    attribute: frozenset[str]    # content words describing what it measures
    text: str                    # as written, for the explanation

    def same_kind_as(self, other: "Quantity") -> bool:
        return self.kind == other.kind

    def matches_value(self, other: "Quantity") -> bool:
        if not self.same_kind_as(other):
            return False
        if self.value == other.value:
            return True
        scale = max(abs(self.value), abs(other.value))
        return abs(self.value - other.value) <= RELATIVE_TOLERANCE * scale

    def describes_same_attribute(self, other: "Quantity") -> bool:
        return bool(self.attribute & other.attribute)


def _classify(symbol: str | None, unit: str | None) -> str:
    unit_lower = (unit or "").lower()
    if unit_lower in _PERCENT_WORDS or unit_lower.startswith("percentage point"):
        return "percent"
    if symbol or unit_lower in _CURRENCY_WORDS:
        return "currency"
    return "count"


def _attribute_words(tokens: list[str], centre: int) -> frozenset[str]:
    """Content words within ATTRIBUTE_WINDOW tokens of the number."""
    start = max(0, centre - ATTRIBUTE_WINDOW)
    end = min(len(tokens), centre + ATTRIBUTE_WINDOW + 1)
    return frozenset(
        token for token in tokens[start:end]
        if len(token) > 2 and token not in _NON_ATTRIBUTE
        and not any(char.isdigit() for char in token)
    )


def quantities_in(text: str) -> list[Quantity]:
    """Every figure written in ``text``, with the words describing it."""
    if not text:
        return []
    lowered = text.lower()
    # Token positions let the attribute window be measured in words rather
    # than characters, so a long number does not shrink its own context.
    tokens = re.findall(r"[a-z0-9$£€%.,]+", lowered)
    token_starts: list[int] = []
    cursor = 0
    for token in tokens:
        index = lowered.find(token, cursor)
        token_starts.append(index)
        cursor = index + len(token)

    found: list[Quantity] = []
    for match in _QUANTITY.finditer(lowered):
        raw_number = match.group("number").replace(",", "")
        try:
            value = float(raw_number)
        except ValueError:
            continue
        magnitude = (match.group("magnitude") or "").lower()
        if magnitude:
            value *= _MAGNITUDES[magnitude]

        kind = _classify(match.group("symbol"), match.group("unit"))
        # A bare number with no unit and no magnitude is usually an
        # identifier, an ordinal or a date fragment rather than a measurement.
        # It is still recorded as a count: the attribute test decides whether
        # it is comparable to anything in the claim.
        centre = 0
        for position, start in enumerate(token_starts):
            if start > match.start():
                break
            centre = position
        found.append(Quantity(
            value=value,
            kind=kind,
            attribute=_attribute_words(tokens, centre),
            text=match.group(0).strip(),
        ))
    return found


def conflicting_quantity(
    claim: str, passages: list[str]
) -> tuple[Quantity, Quantity] | None:
    """The claim's figure and the different one the document states, or None.

    Returns None whenever the document is silent about the claim's figure
    without stating a rival one — silence is not a conflict, it is simply not
    the confirmation the entailment score implied.
    """
    claim_quantities = quantities_in(claim)
    if not claim_quantities:
        return None

    document_quantities = [q for passage in passages for q in quantities_in(passage)]
    if not document_quantities:
        return None

    for claimed in claim_quantities:
        if not claimed.attribute:
            continue  # No idea what it measures; nothing safe to compare.
        comparable = [
            q for q in document_quantities
            if q.same_kind_as(claimed) and q.describes_same_attribute(claimed)
        ]
        if not comparable:
            continue
        if any(q.matches_value(claimed) for q in comparable):
            continue  # The document states the claim's figure somewhere.
        return claimed, comparable[0]
    return None
