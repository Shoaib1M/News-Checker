"""Provider registry — orchestrates multiple search providers with diagnostics.

The registry runs each enabled provider, tracks per-query outcomes, deduplicates
results, and returns a combined list of SearchResults along with diagnostics
that the health endpoint and the API response can report.
"""

from __future__ import annotations

import os
import re
from typing import Callable

from providers import SearchResult, ProviderDiagnostic
from providers.duckduckgo import search as ddg_search
from providers.news_apis import search_newsapi, search_gnews, search_guardian


# ── Provider configuration ───────────────────────────────────────────
PROVIDERS: list[tuple[str, str, Callable]] = [
    # (name, env_key, search_function)
    ("gnews", "GNEWS_API_KEY", search_gnews),
    ("guardian", "GUARDIAN_API_KEY", search_guardian),
    ("newsapi", "NEWSAPI_KEY", search_newsapi),
]


# ── Deduplication ────────────────────────────────────────────────────
def _title_key(title: str) -> str:
    key = re.sub(r"[^a-z0-9\s]", "", title.strip().lower())
    key = re.sub(
        r"\b(reuters|associated press|ap news|the guardian|bbc news)\b",
        "", key,
    )
    return re.sub(r"\s+", " ", key).strip()


def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """Remove exact URL duplicates and near-duplicate syndicated titles."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    title_token_sets: list[set[str]] = []
    unique: list[SearchResult] = []

    for r in results:
        if not r.url or r.url in seen_urls:
            continue

        tk = _title_key(r.title)
        if tk and tk in seen_titles:
            continue

        tokens = set(tk.split())
        if len(tokens) >= 6 and any(
            len(tokens & prev) / len(tokens | prev) >= 0.85
            for prev in title_token_sets
        ):
            continue

        seen_urls.add(r.url)
        if tk:
            seen_titles.add(tk)
        if len(tokens) >= 6:
            title_token_sets.append(tokens)
        unique.append(r)

    return unique


# ── Orchestration ────────────────────────────────────────────────────
def search_all_providers(
    queries: list[str],
    max_per_provider: int = 5,
    use_duckduckgo: bool = True,
) -> tuple[list[SearchResult], list[ProviderDiagnostic]]:
    """Run all configured providers with multiple query variants.

    Returns (deduplicated_results, diagnostics).
    """
    all_results: list[SearchResult] = []
    diagnostics: list[ProviderDiagnostic] = []
    seen_urls: set[str] = set()

    for query in queries[:4]:  # Limit to top 4 queries
        for name, env_key, search_fn in PROVIDERS:
            diag = ProviderDiagnostic(
                provider=name, query=query,
                enabled=bool(os.getenv(env_key)),
            )
            if not diag.enabled:
                diagnostics.append(diag)
                continue
            try:
                provider_results = search_fn(
                    query, max_results=max(2, max_per_provider)
                )
                diag.status = "success" if provider_results else "no_results"
                diag.raw_result_count = len(provider_results)
                for sr in provider_results:
                    if sr.url and sr.url not in seen_urls:
                        seen_urls.add(sr.url)
                        all_results.append(sr)
                        diag.new_result_count += 1
            except Exception as exc:
                diag.status = "failed"
                diag.error = str(exc)
            diagnostics.append(diag)

    # DuckDuckGo fallback — always available
    if use_duckduckgo:
        for query in queries[:4]:
            diag = ProviderDiagnostic(
                provider="duckduckgo", query=query, enabled=True,
            )
            try:
                ddg_results = ddg_search(query, max_results=max(3, max_per_provider // 2))
                diag.status = "success" if ddg_results else "no_results"
                diag.raw_result_count = len(ddg_results)
                for sr in ddg_results:
                    if sr.url and sr.url not in seen_urls:
                        seen_urls.add(sr.url)
                        all_results.append(sr)
                        diag.new_result_count += 1
            except Exception as exc:
                diag.status = "failed"
                diag.error = str(exc)
            diagnostics.append(diag)

    return deduplicate(all_results), diagnostics
