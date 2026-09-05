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


def _fetch(url: str, timeout: int = 10, retries: int = 2) -> str:
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
                time.sleep(1.5 * (attempt + 1))
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
        self._url = ""
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        cls = attrs.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._in_link = True
            self._url = _clean_url(attrs.get("href", ""))
            self._title_parts = []
            self._snippet_parts = []
        if tag in {"a", "div"} and "result__snippet" in cls:
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link:
            self._in_link = False
            title = " ".join(self._title_parts).strip()
            snippet = " ".join(self._snippet_parts).strip()
            if self._url and title:
                self.results.append({
                    "url": self._url, "title": title, "snippet": snippet,
                })
        if tag in {"a", "div"} and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._in_link:
            self._title_parts.append(text)
        if self._in_snippet:
            self._snippet_parts.append(text)


def search(query: str, max_results: int = 10) -> list[SearchResult]:
    """Execute a DuckDuckGo HTML search and return normalized results."""
    encoded = quote_plus(query)
    html = _fetch(SEARCH_URL.format(query=encoded))
    parser = _DDGParser()
    parser.feed(html)

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
