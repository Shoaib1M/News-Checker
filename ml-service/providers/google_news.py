"""Google News RSS search provider — no API key required.

WHY THIS EXISTS:
Without GNEWS_API_KEY / GUARDIAN_API_KEY / NEWSAPI_KEY configured, retrieval
previously fell back entirely to scraping DuckDuckGo's HTML endpoint, which
routinely rate-limits or blocks scripted requests and — when it does answer —
returns general web pages ranked for a search query rather than news coverage
ranked for recency. That combination is what produces the failure people
actually notice: a fresh headline is submitted, the only reachable provider
returns loosely-related evergreen pages, and the system presents them as the
candidates it examined.

Google News publishes a plain RSS search feed. It needs no key, no signup and
no quota, it is indexed for recency, and it returns publisher attribution.
That makes it the right default provider for this project's actual workload
(news headlines) and it runs alongside the keyed providers when those are
configured.

LIMITATIONS (stated honestly, because they matter for verdicts):
- Article URLs are Google redirect links. The article extractor follows
  redirects, so passages are still pulled from the publisher.
- The feed reflects Google News indexing, not the whole press. Absence here
  is evidence only in combination with the other providers and only under the
  narrow conditions in evidence_aggregator.assess_coverage.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen

from providers import SearchResult
from providers.dates import parse_rfc2822

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

# hl/gl/ceid pin the feed to English-language editions; without them Google
# infers a locale from the caller's IP, which makes results non-reproducible
# between a laptop and a deployed container.
FEED_URL = (
    "https://news.google.com/rss/search"
    "?q={query}&hl=en-US&gl=US&ceid=US:en"
)

# Matches a trailing " - Something" in an RSS title. Google appends the
# publisher this way — but plenty of headlines end in a dash clause of their
# own ("Google banned in US - what it means for you"), and treating that as a
# publisher did two things wrong: it truncated the headline, which NLI reads
# as a passage, and it invented a publisher called "what it means for you",
# which then counted as a distinct independent source and inflated confidence.
_TITLE_SOURCE_SUFFIX = re.compile(r"\s+-\s+([^-]{2,40})$")

# Words a headline clause starts with and a publisher name does not.
_NOT_A_PUBLISHER_START = frozenset({
    "what", "why", "how", "here", "when", "where", "who", "which", "this",
    "that", "and", "but", "or", "so", "with", "after", "before", "as",
    "everything", "all", "the latest", "live", "explained", "updates",
})


def _looks_like_publisher(text: str) -> bool:
    """True when a trailing dash clause reads as a masthead, not a headline.

    Publisher names are short, capitalised, and are not sentences. This is a
    shape test, so an unusual masthead may be missed — which is the safe
    direction: the title keeps a few extra words and the publisher falls back
    to the link's host.
    """
    candidate = text.strip()
    if not candidate or candidate[0].islower() or candidate[-1] in "?!.":
        return False
    words = candidate.split()
    if not 1 <= len(words) <= 5:
        return False
    return words[0].lower() not in _NOT_A_PUBLISHER_START

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """RSS descriptions are HTML fragments; passages need plain text."""
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text or "")).strip()


def search(query: str, max_results: int = 5, timeout: int = 6,
           recent_days: int | None = None) -> list[SearchResult]:
    """Return news results for one query.

    Raises on network failure so the provider registry records a diagnostic
    rather than silently reporting an empty press.
    """
    # Google News' search endpoint understands a `when:` operator. Without it
    # the feed is recency-RANKED but not recency-FILTERED, so a query about
    # today's story still returns last year's coverage of the same subject
    # whenever that older coverage matches the words better.
    effective_query = query.strip()
    if recent_days:
        effective_query = f"{effective_query} when:{recent_days}d"
    url = FEED_URL.format(query=quote_plus(effective_query))
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()

    root = ET.fromstring(payload)
    results: list[SearchResult] = []

    for item in root.iterfind(".//item"):
        link = (item.findtext("link") or "").strip()
        title = _strip_html(item.findtext("title") or "")
        if not link or not title:
            continue

        # Publisher: the <source> element when present, otherwise a trailing
        # dash clause that actually reads like a masthead.
        source_el = item.find("source")
        declared = (source_el.text or "").strip() if source_el is not None else ""
        match = _TITLE_SOURCE_SUFFIX.search(title)
        suffix = match.group(1).strip() if match else ""

        if declared:
            source = declared
            # Strip the suffix only when it IS the publisher. Otherwise it is
            # part of the headline and removing it loses meaning: "Google
            # banned in US - what it means for you" is not a story about a ban
            # once the second half is gone.
            strip_suffix = bool(suffix) and suffix.lower() == declared.lower()
        elif suffix and _looks_like_publisher(suffix):
            source = suffix
            strip_suffix = True
        else:
            source = urlparse(link).netloc
            strip_suffix = False

        if strip_suffix:
            title = _TITLE_SOURCE_SUFFIX.sub("", title).strip()

        results.append(SearchResult(
            url=link,
            title=title,
            snippet=_strip_html(item.findtext("description") or "")[:400],
            text="",
            provider="google_news",
            source=source,
            published=parse_rfc2822(item.findtext("pubDate")),
        ))

        if len(results) >= max_results:
            break

    return results
