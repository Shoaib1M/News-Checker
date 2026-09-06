"""News API search providers — NewsAPI, GNews, Guardian.

Each function follows the same contract:
    search(query, max_results) -> list[SearchResult]
and returns an empty list (not an error) when the required API key is missing.

A note on ``text``: NewsAPI and GNews do not return article bodies on their
free tiers. They return a ~200-character excerpt with a literal truncation
marker appended ("… [+2345 chars]"). That excerpt is a snippet, not a body,
and is treated as one here — otherwise the pipeline mistakes it for full text,
skips fetching the real article, and hands the marker to NLI as if it were
prose.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from providers import SearchResult
from providers.dates import parse_iso8601


def _days_ago(days: int) -> datetime:
    """The cutoff a date-filtered query should ask from."""
    return datetime.now(timezone.utc) - timedelta(days=days)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

NEWSAPI_URL = "https://newsapi.org/v2/everything"
GNEWS_URL = "https://gnews.io/api/v4/search"
GUARDIAN_URL = "https://content.guardianapis.com/search"


# Matches the truncation marker NewsAPI and GNews append to excerpts.
_TRUNCATION_MARKER = re.compile(r"[…\.]{0,3}\s*\[\+\d+\s*chars?\]\s*$")


def _clean_excerpt(text: str) -> str:
    """Strip the provider's truncation marker from an excerpt."""
    return _TRUNCATION_MARKER.sub("", (text or "").strip()).strip()


def _fetch_json(url: str, timeout: int = 8) -> dict:
    """Fetch and decode a provider's JSON, raising on an error payload.

    Some providers answer a bad key or an exhausted quota with HTTP 200 and
    an error object rather than a status code. Returning that as zero
    articles would be recorded as "no_results" — indistinguishable from the
    press genuinely having nothing on the claim, which is the confusion this
    whole codebase exists to avoid.
    """
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        payload = json.loads(response.read().decode(charset, errors="ignore"))

    if isinstance(payload, dict):
        if payload.get("status") == "error":
            raise RuntimeError(
                f"{payload.get('code', 'error')}: {payload.get('message', 'unknown')}"
            )
        errors = payload.get("errors")
        if errors:
            raise RuntimeError(f"provider error: {errors}")
    return payload


# ── NewsAPI ──────────────────────────────────────────────────────────
def search_newsapi(query: str, max_results: int = 10,
                   recent_days: int | None = None) -> list[SearchResult]:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return []

    params = {
        "q": query.strip(),
        "language": "en",
        # Relevancy is right for an undated claim and wrong for today's news:
        # it ranks the best-worded match, which for a breaking story is
        # routinely last year's article about the same subject.
        "sortBy": "publishedAt" if recent_days else "relevancy",
        "pageSize": max_results,
        "apiKey": api_key,
    }
    if recent_days:
        params["from"] = _days_ago(recent_days).strftime("%Y-%m-%d")
    payload = _fetch_json(f"{NEWSAPI_URL}?{urlencode(params)}")

    results: list[SearchResult] = []
    for article in payload.get("articles", []):
        url = article.get("url", "")
        title = article.get("title", "") or ""
        if not url or not title:
            continue
        # `content` is an excerpt, not a body — keep the longer of it and
        # `description` as the snippet, and leave `text` empty so the
        # pipeline knows to fetch the real article.
        description = article.get("description", "") or ""
        excerpt = _clean_excerpt(article.get("content", ""))
        results.append(SearchResult(
            url=url,
            title=title,
            snippet=excerpt if len(excerpt) > len(description) else description,
            text="",
            provider="newsapi",
            source=article.get("source", {}).get("name", "") or "NewsAPI",
            published=parse_iso8601(article.get("publishedAt")),
        ))
    return results


# ── GNews ────────────────────────────────────────────────────────────
def search_gnews(query: str, max_results: int = 10,
                 recent_days: int | None = None) -> list[SearchResult]:
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return []

    params = {
        "q": query.strip(),
        "lang": "en",
        "max": max_results,
        "apikey": api_key,
    }
    if recent_days:
        params["from"] = _days_ago(recent_days).strftime("%Y-%m-%dT%H:%M:%SZ")
        params["sortby"] = "publishedAt"
    payload = _fetch_json(f"{GNEWS_URL}?{urlencode(params)}")

    results: list[SearchResult] = []
    for article in payload.get("articles", []):
        url = article.get("url", "")
        title = article.get("title", "") or ""
        if not url or not title:
            continue
        description = article.get("description", "") or ""
        excerpt = _clean_excerpt(article.get("content", ""))
        results.append(SearchResult(
            url=url,
            title=title,
            snippet=excerpt if len(excerpt) > len(description) else description,
            text="",
            provider="gnews",
            source=article.get("source", {}).get("name", "") or "GNews",
            published=parse_iso8601(article.get("publishedAt")),
        ))
    return results


# ── Guardian ─────────────────────────────────────────────────────────
def search_guardian(query: str, max_results: int = 10,
                    recent_days: int | None = None) -> list[SearchResult]:
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        return []

    params = {
        "q": query.strip(),
        "api-key": api_key,
        "page-size": max_results,
        "show-fields": "headline,trailText,bodyText",
        "order-by": "newest" if recent_days else "relevance",
    }
    if recent_days:
        params["from-date"] = _days_ago(recent_days).strftime("%Y-%m-%d")
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
            # The Guardian is the one provider here that returns a real
            # article body, so this text is genuine and no fetch is needed.
            snippet=fields.get("trailText", "") or "",
            text=fields.get("bodyText", "") or "",
            provider="guardian",
            source="The Guardian",
            published=parse_iso8601(article.get("webPublicationDate")),
        ))
    return results
