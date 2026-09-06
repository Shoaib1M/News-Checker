"""DuckDuckGo HTML search provider.

Scrapes DuckDuckGo's HTML-only endpoint, which does not require an API key.
This is the always-available fallback when news API keys are missing.
"""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen
import time

from providers import SearchResult

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
SEARCH_URL = "https://duckduckgo.com/html/?q={query}"


def _fetch(url: str, timeout: int = 6, retries: int = 1) -> str:
    """Fetch a URL with a tight timeout and at most one retry.

    Timeouts and retries are deliberately small: DuckDuckGo routinely blocks
    or stalls scripted requests, and this runs once per generated query. The
    old 10s/3-attempt settings meant a single blocked claim could burn ~138
    seconds here alone (4 queries x 3 attempts x 10s + backoff) — longer than
    any sane request timeout upstream.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="ignore")
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.0)
    raise last_error


def _clean_url(url: str) -> str:
    """DuckDuckGo wraps links in their own redirect. Unwrap them."""
    if not url:
        return ""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "uddg" in query:
        return unquote(query["uddg"][0])
    return url


class _DDGParser(HTMLParser):
    """Extracts search results from DuckDuckGo's HTML response."""

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._in_link = False
        self._in_snippet = False
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []
        # The result under construction. DuckDuckGo emits the title link
        # first and the snippet element after it, so a result cannot be
        # finished when its title's </a> fires — which is what the previous
        # version did, giving every DuckDuckGo result an empty snippet.
        # Relevance was then scored on the headline alone, and since
        # DuckDuckGo is the always-on fallback in an unconfigured checkout,
        # that was the default experience.
        self._pending: dict | None = None

    def _flush(self) -> None:
        """Emit the result under construction, if it has a URL and a title."""
        if self._pending and self._pending.get("url") and self._pending.get("title"):
            self.results.append(self._pending)
        self._pending = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        cls = attrs.get("class", "")
        if tag == "a" and "result__a" in cls:
            # A new result begins; the previous one is complete.
            self._flush()
            self._in_link = True
            self._pending = {"url": _clean_url(attrs.get("href", "")),
                             "title": "", "snippet": ""}
            self._title_parts = []
            self._snippet_parts = []
        if tag in {"a", "div", "span"} and "result__snippet" in cls:
            self._in_snippet = True
            self._snippet_parts = []

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link:
            self._in_link = False
            if self._pending is not None:
                self._pending["title"] = " ".join(self._title_parts).strip()
        if tag in {"a", "div", "span"} and self._in_snippet:
            self._in_snippet = False
            if self._pending is not None:
                self._pending["snippet"] = " ".join(self._snippet_parts).strip()

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._in_snippet:
            self._snippet_parts.append(text)
        elif self._in_link:
            self._title_parts.append(text)

    def close(self):
        """Emit the final result, which has no following result to flush it."""
        super().close()
        self._flush()


def search(query: str, max_results: int = 10) -> list[SearchResult]:
    """Execute a DuckDuckGo HTML search and return normalized results."""
    encoded = quote_plus(query)
    html = _fetch(SEARCH_URL.format(query=encoded))
    parser = _DDGParser()
    parser.feed(html)
    parser.close()  # flushes the last result

    seen: set[str] = set()
    results: list[SearchResult] = []
    for item in parser.results:
        url = item["url"]
        if url in seen:
            continue
        seen.add(url)
        results.append(SearchResult(
            url=url,
            title=item["title"],
            snippet=item.get("snippet", ""),
            provider="duckduckgo",
            source=urlparse(url).netloc,
        ))
        if len(results) >= max_results:
            break
    return results
