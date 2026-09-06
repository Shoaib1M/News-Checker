"""
FILE PURPOSE:
Read a publication date out of whatever format a provider happens to use, and
say honestly when there isn't one.

WHY THIS EXISTS:
Nothing in this system used to know when an article was published. The field
was never carried, so nothing could compare "when this was reported" against
"when the claim says it happened" — and a 2014 story about a resignation
entails "the prime minister resigned this morning" perfectly. For today's news
that is the difference between a fact-check and a coincidence.

The dates were always there. Google News ships RFC 2822 in <pubDate>, GNews and
NewsAPI ship ISO 8601 in publishedAt, the Guardian ships ISO 8601 in
webPublicationDate. All three were being decoded and thrown away.

WHY EVERYTHING RETURNS None RATHER THAN A GUESS:
A wrong date is worse than no date. It would be used to discard evidence or to
age a document out of a comparison, so a parser that guesses when it is
uncertain would silently delete real coverage. Every caller must handle None,
because Wikipedia and DuckDuckGo genuinely do not provide one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def _as_utc(value: datetime) -> datetime:
    """Normalise to timezone-aware UTC.

    A naive datetime compared against an aware one raises TypeError, and the
    feeds mix both. Naive values are read as UTC: every format here specifies
    a zone in practice, so a missing one means the feed omitted it, and UTC is
    the only defensible reading.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_rfc2822(value: str | None) -> datetime | None:
    """`Tue, 02 Sep 2025 14:32:00 GMT` — RSS `<pubDate>`."""
    if not value or not value.strip():
        return None
    try:
        return _as_utc(parsedate_to_datetime(value.strip()))
    except (TypeError, ValueError):
        return None


def parse_iso8601(value: str | None) -> datetime | None:
    """`2025-09-02T14:32:00Z` — the JSON APIs."""
    if not value or not value.strip():
        return None
    text = value.strip()
    # fromisoformat only learned to accept a trailing "Z" in 3.11; being
    # explicit costs nothing and keeps this readable.
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return _as_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def parse_any(value: str | None) -> datetime | None:
    """Try both formats, for callers that cannot know which they'll get."""
    return parse_iso8601(value) or parse_rfc2822(value)
