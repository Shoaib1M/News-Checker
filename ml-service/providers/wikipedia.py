"""Wikipedia search provider — no API key required.

WHY THIS EXISTS:
The news providers are tuned for recency and are close to useless for the
other half of this system's workload: timeless factual claims ("water freezes
at 0°C", "the Eiffel Tower is owned by the city of Paris"). Nothing was
covering that gap except a hand-written pattern table in knowledge_verifier.py,
which only fires on the exact dozen claims someone thought to write down.

Wikipedia's search API is keyless, quota-free, stable, and its articles state
background facts explicitly enough for an NLI model to entail or contradict a
claim against them. It is a *background-knowledge* provider, not a news
provider, and it is weighted accordingly by claim_verifier.classify_source.

LIMITATION:
Wikipedia is a tertiary source and can be wrong or out of date. It is treated
as one more candidate that still has to entail the specific claim before it
can move a verdict — never as an authority whose mere presence settles
anything.
"""

from __future__ import annotations

import json
import re
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

from providers import SearchResult

# The Wikimedia API asks that clients identify themselves; anonymous default
# user agents are rate-limited more aggressively.
USER_AGENT = "NewsChecker/2.0 (fact-check research project)"

API_URL = "https://en.wikipedia.org/w/api.php"
PAGE_URL = "https://en.wikipedia.org/wiki/{title}"

_TAG_RE = re.compile(r"<[^>]+>")

# Search operators that help a news index but only confuse Wikipedia's search:
# it has no boolean OR grouping in this endpoint, and quoted long phrases
# match nothing.
_OPERATOR_RE = re.compile(r'[()"]|(?:\bOR\b)', re.IGNORECASE)


def _strip_html(text: str) -> str:
    """Search snippets come back with <span class="searchmatch"> markup."""
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text or "")).strip()


def _plain_query(query: str) -> str:
    """Reduce a generated search query to bare keywords."""
    cleaned = _OPERATOR_RE.sub(" ", query)
    return re.sub(r"\s+", " ", cleaned).strip()


def search(query: str, max_results: int = 3, timeout: int = 6) -> list[SearchResult]:
    """Return Wikipedia articles matching one query.

    Raises on network or decode failure so the registry records a diagnostic
    instead of the caller reading an empty list as "nothing exists".
    """
    params = urlencode({
        "action": "query",
        "list": "search",
        "srsearch": _plain_query(query),
        "srlimit": max(1, min(max_results, 10)),
        "srprop": "snippet",
        "format": "json",
        "formatversion": "2",
    })
    request = Request(f"{API_URL}?{params}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", errors="ignore"))

    hits = payload.get("query", {}).get("search", []) or []
    results: list[SearchResult] = []

    for hit in hits[:max_results]:
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        results.append(SearchResult(
            url=PAGE_URL.format(title=quote_plus(title.replace(" ", "_"))),
            title=title,
            snippet=_strip_html(hit.get("snippet", "")),
            text="",
            provider="wikipedia",
            source="en.wikipedia.org",
        ))

    return results
