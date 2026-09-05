"""News API search providers — NewsAPI, GNews, Guardian.

Each function follows the same contract:
    search(query, max_results) -> list[SearchResult]
and returns an empty list (not an error) when the required API key is missing.
"""

from __future__ import annotations

import json
import os
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from providers import SearchResult

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

NEWSAPI_URL = "https://newsapi.org/v2/everything"
GNEWS_URL = "https://gnews.io/api/v4/search"
GUARDIAN_URL = "https://content.guardianapis.com/search"


def _fetch_json(url: str, timeout: int = 10) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset, errors="ignore"))


# ── NewsAPI ──────────────────────────────────────────────────────────
def search_newsapi(query: str, max_results: int = 10) -> list[SearchResult]:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return []

    params = {
        "q": query.strip(),
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_results,
        "apiKey": api_key,
    }
    payload = _fetch_json(f"{NEWSAPI_URL}?{urlencode(params)}")

    results: list[SearchResult] = []
    for article in payload.get("articles", []):
        url = article.get("url", "")
        title = article.get("title", "") or ""
        if not url or not title:
            continue
        results.append(SearchResult(
            url=url,
            title=title,
            snippet=article.get("description", "") or "",
            text=article.get("content", "") or "",
            provider="newsapi",
            source=article.get("source", {}).get("name", "") or "NewsAPI",
        ))
    return results


# ── GNews ────────────────────────────────────────────────────────────
def search_gnews(query: str, max_results: int = 10) -> list[SearchResult]:
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return []

    params = {
        "q": query.strip(),
        "lang": "en",
        "max": max_results,
        "apikey": api_key,
    }
    payload = _fetch_json(f"{GNEWS_URL}?{urlencode(params)}")

    results: list[SearchResult] = []
    for article in payload.get("articles", []):
        url = article.get("url", "")
        title = article.get("title", "") or ""
        if not url or not title:
            continue
        results.append(SearchResult(
            url=url,
            title=title,
            snippet=article.get("description", "") or "",
            text=article.get("content", "") or "",
            provider="gnews",
            source=article.get("source", {}).get("name", "") or "GNews",
        ))
    return results


# ── Guardian ─────────────────────────────────────────────────────────
def search_guardian(query: str, max_results: int = 10) -> list[SearchResult]:
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        return []

    params = {
        "q": query.strip(),
        "api-key": api_key,
        "page-size": max_results,
        "show-fields": "headline,trailText,bodyText",
        "order-by": "relevance",
    }
    payload = _fetch_json(f"{GUARDIAN_URL}?{urlencode(params)}")

    results: list[SearchResult] = []
    for article in payload.get("response", {}).get("results", []):
        fields = article.get("fields", {})
        url = article.get("webUrl", "")
        title = fields.get("headline") or article.get("webTitle", "") or ""
        if not url or not title:
            continue
        results.append(SearchResult(
            url=url,
            title=title,
            snippet=fields.get("trailText", "") or "",
            text=fields.get("bodyText", "") or "",
            provider="guardian",
            source="The Guardian",
        ))
    return results
