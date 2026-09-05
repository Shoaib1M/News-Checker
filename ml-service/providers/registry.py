"""Provider registry — orchestrates multiple search providers with diagnostics.

The registry runs each enabled provider, tracks per-query outcomes, deduplicates
results, and returns a combined list of SearchResults along with diagnostics
that the health endpoint and the API response can report.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
def _run_one(
    name: str, query: str, search_fn: Callable, max_results: int, enabled: bool,
) -> tuple[ProviderDiagnostic, list[SearchResult]]:
    """Run a single provider/query pair, capturing its outcome as a diagnostic."""
    diag = ProviderDiagnostic(provider=name, query=query, enabled=enabled)
    if not enabled:
        return diag, []
    try:
        results = search_fn(query, max_results=max_results)
        diag.status = "success" if results else "no_results"
        diag.raw_result_count = len(results)
        return diag, results
    except Exception as exc:
        diag.status = "failed"
        diag.error = str(exc)
        return diag, []


def search_all_providers(
    queries: list[str],
    max_per_provider: int = 5,
    use_duckduckgo: bool = True,
    deadline: float | None = None,
) -> tuple[list[SearchResult], list[ProviderDiagnostic]]:
    """Run all configured providers with multiple query variants, concurrently.

    Every provider/query pair is independent, so they run in a thread pool
    rather than one after another — previously a single blocked provider
    stalled every remaining query in sequence.

    ``deadline`` is an absolute ``time.monotonic()`` value. Work still
    outstanding when it passes is abandoned and reported as a ``timeout``
    diagnostic, so retrieval degrades to partial results instead of
    overrunning the caller's request budget.

    Returns (deduplicated_results, diagnostics).
    """
    jobs: list[tuple[str, str, Callable, int, bool]] = []

    for query in queries[:4]:  # Limit to top 4 queries
        for name, env_key, search_fn in PROVIDERS:
            jobs.append((
                name, query, search_fn,
                max(2, max_per_provider), bool(os.getenv(env_key)),
            ))
        if use_duckduckgo:
            jobs.append((
                "duckduckgo", query, ddg_search,
                max(3, max_per_provider // 2), True,
            ))

    completed: list[tuple[ProviderDiagnostic, list[SearchResult]]] = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [(pool.submit(_run_one, *job), job) for job in jobs]
        for future, job in futures:
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    future.cancel()
                    completed.append((ProviderDiagnostic(
                        provider=job[0], query=job[1], enabled=job[4],
                        status="timeout", error="search budget exhausted",
                    ), []))
                    continue
            try:
                completed.append(future.result(timeout=remaining))
            except Exception as exc:
                completed.append((ProviderDiagnostic(
                    provider=job[0], query=job[1], enabled=job[4],
                    status="timeout", error=str(exc) or "search budget exhausted",
                ), []))

    # Merge results, attributing each new URL to the provider that found it.
    all_results: list[SearchResult] = []
    seen_urls: set[str] = set()
    diagnostics: list[ProviderDiagnostic] = []
    for diag, results in completed:
        for sr in results:
            if sr.url and sr.url not in seen_urls:
                seen_urls.add(sr.url)
                all_results.append(sr)
                diag.new_result_count += 1
        diagnostics.append(diag)

    return deduplicate(all_results), diagnostics
