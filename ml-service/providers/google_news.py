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

# Matches the trailing " - Publisher Name" Google appends to every RSS title.
_TITLE_SOURCE_SUFFIX = re.compile(r"\s+-\s+([^-]{2,40})$")

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """RSS descriptions are HTML fragments; passages need plain text."""
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text or "")).strip()


def search(query: str, max_results: int = 5, timeout: int = 6) -> list[SearchResult]:
    """Return news results for one query.

    Raises on network failure so the provider registry records a diagnostic
    rather than silently reporting an empty press.
    """
    url = FEED_URL.format(query=quote_plus(query))
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

        # Publisher: the <source> element when present, otherwise the suffix
        # Google appends to the title, otherwise the link's host.
        source_el = item.find("source")
        if source_el is not None and (source_el.text or "").strip():
            source = source_el.text.strip()
        else:
            match = _TITLE_SOURCE_SUFFIX.search(title)
            source = match.group(1).strip() if match else urlparse(link).netloc

        # Drop the publisher suffix from the title so it doesn't leak into
        # NLI passages as if it were part of the reporting.
        title = _TITLE_SOURCE_SUFFIX.sub("", title).strip()

        results.append(SearchResult(
            url=link,
            title=title,
            snippet=_strip_html(item.findtext("description") or "")[:400],
            text="",
            provider="google_news",
            source=source,
        ))

        if len(results) >= max_results:
            break

    return results
