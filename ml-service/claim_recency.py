"""
FILE PURPOSE:
Work out when a claim says something happened, and whether a document is old
enough that it cannot be reporting it.

WHY THIS EXISTS:
Entailment has no clock. "India's prime minister resigned on Tuesday" entails
"the prime minister of India resigned this morning" perfectly — and if that
article is from 2014, the entailment is a coincidence, not a confirmation.
Nothing in the pipeline compared an article's date against the claim's, because
until now no article carried a date at all.

This is the single largest weakness on TODAY's news specifically. On a claim
with no time anchor ("the Eiffel Tower is in Paris") an old source is a fine
source. On "the PM resigned this morning" an old source is the wrong event.

WHAT IT DELIBERATELY DOES NOT DO:
It never marks a document as CONTRADICTING the claim. An old article is not
evidence that today's event didn't happen; it is simply not evidence that it
did. Support is withdrawn, nothing is asserted.

It also never fires when the document has no date. Wikipedia and DuckDuckGo
supply none, and treating "unknown" as "old" would delete real evidence on the
strength of a missing field.

THE WINDOW IS DELIBERATELY GENEROUS:
STALE_AFTER_DAYS is 14, not 1, for a claim that says "today". Feed timestamps
are unreliable — republished articles carry an updated date, some feeds emit
the crawl time — and people say "today" about something they read last week.
Fourteen days absorbs all of that while still excluding the article from 2014,
which is the failure this exists to prevent. Rejecting evidence is destructive:
enough wrong rejections become "no credible source reports this", a statement
about the world.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# The boundary between the two modes, in days. A claim in RECENT mode is about
# something inside this window, and retrieval asks every provider that supports
# a date filter for exactly this window.
RECENT_WINDOW_DAYS = 30

# How far back a document may be published and still count as reporting a
# RECENT claim's event. Wider than the retrieval window on purpose: retrieval
# should ASK for the last 30 days, but evidence that arrives from just outside
# it — a feed with a sloppy timestamp, an article republished with a new date —
# should not be thrown away at the boundary. See the docstring on why rejecting
# evidence is the destructive direction.
STALE_AFTER_DAYS = 45

RECENT = "recent"
HISTORICAL = "historical"
AUTO = "auto"
MODES = (AUTO, RECENT, HISTORICAL)

# Phrases that anchor a claim to roughly now. Multi-word forms are matched
# whole: a bare "today" is unambiguous, but "just" alone is a filler word in
# half the sentences it appears in.
_RECENT_MARKERS = (
    "today", "this morning", "this afternoon", "this evening", "tonight",
    "yesterday", "last night", "this week", "this weekend", "right now",
    "just now", "just announced", "just resigned", "just died",
    "moments ago", "hours ago", "minutes ago", "earlier today",
    "breaking", "developing", "currently", "as of today", "so far today",
)

# A four-digit year, or a month name, anchors the claim to a specific past
# time instead of to now. Those claims are checked against coverage from then,
# which this module has no opinion about.
_EXPLICIT_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_MONTH_NAME = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class ClaimTime:
    """Which mode this check runs in, and why."""
    anchor: str          # "recent" | "dated" | "none"
    marker: str = ""     # the phrase that decided it, for the explanation
    requested: str = AUTO  # what the caller asked for

    @property
    def is_recent(self) -> bool:
        return self.anchor == "recent"

    @property
    def mode(self) -> str:
        return RECENT if self.is_recent else HISTORICAL

    @property
    def window_days(self) -> int | None:
        """Days of coverage to ask providers for, or None for no limit."""
        return RECENT_WINDOW_DAYS if self.is_recent else None

    @property
    def reason(self) -> str:
        if self.requested != AUTO:
            return f"you selected {self.mode} coverage"
        if self.marker:
            return f'the claim says "{self.marker}"'
        if self.anchor == "dated":
            return "the claim names a specific date"
        return "the claim is not anchored to a particular time"


def detect_anchor(claim: str) -> ClaimTime:
    """Read the claim's own time anchor, ignoring any requested mode."""
    lowered = f" {(claim or '').lower()} "
    for marker in _RECENT_MARKERS:
        if re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", lowered):
            return ClaimTime(anchor="recent", marker=marker)
    if _EXPLICIT_YEAR.search(lowered) or _MONTH_NAME.search(lowered):
        return ClaimTime(anchor="dated")
    return ClaimTime(anchor="none")


def resolve_mode(claim: str, requested: str = AUTO) -> ClaimTime:
    """Decide which mode this check runs in.

    An explicit choice always wins. Auto-detection is a convenience, and it
    can only read the words in the claim: "the PM resigned" is a claim about
    today for someone who just saw it on television, and no amount of parsing
    reveals that. The person checking knows; the parser does not.
    """
    requested = (requested or AUTO).strip().lower()
    if requested not in MODES:
        requested = AUTO
    if requested == RECENT:
        return ClaimTime(anchor="recent", marker="", requested=RECENT)
    if requested == HISTORICAL:
        return ClaimTime(anchor="none", marker="", requested=HISTORICAL)
    return detect_anchor(claim)


def claim_time_anchor(claim: str) -> ClaimTime:
    """Backwards-compatible alias for :func:`detect_anchor`."""
    return detect_anchor(claim)


def document_is_stale(
    claim_time: ClaimTime,
    published: datetime | None,
    now: datetime | None = None,
) -> bool:
    """True when this document is too old to be reporting the claim's event.

    False whenever there is any doubt — an undated claim, an undated document,
    or a document inside the window. The caller withdraws support on True; it
    never concludes anything against the claim.
    """
    if not claim_time.is_recent or published is None:
        return False
    reference = now or datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    # A document dated in the future is a broken timestamp, not stale.
    return published < reference - timedelta(days=STALE_AFTER_DAYS)


def describe_staleness(published: datetime, now: datetime | None = None) -> str:
    """How old this is, in the words an evidence card should use."""
    reference = now or datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    days = max(0, (reference - published).days)
    if days >= 730:
        return f"was published about {days // 365} years ago"
    if days >= 60:
        return f"was published about {days // 30} months ago"
    return f"was published {days} days ago"
