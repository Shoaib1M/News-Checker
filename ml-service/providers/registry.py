"""Provider registry — orchestrates multiple search providers with diagnostics.

The registry runs each enabled provider, tracks per-query outcomes, deduplicates
results, and returns a combined list of SearchResults along with diagnostics
that the health endpoint and the API response can report.
"""

from __future__ import annotations

import inspect
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from event_vocabulary import ANTONYMS, SURFACE_TO_EVENT
from providers import SearchResult, ProviderDiagnostic
from providers.duckduckgo import search as ddg_search
from providers.google_news import search as google_news_search
from providers.news_apis import search_newsapi, search_gnews, search_guardian
from providers.wikipedia import search as wikipedia_search


# ── Provider configuration ───────────────────────────────────────────
# Keyed providers. Each is skipped, with an "disabled" diagnostic, when its
# API key is absent — never silently, so a demo that returns thin evidence
# can be traced to missing configuration rather than to the world being empty.
PROVIDERS: list[tuple[str, str, Callable]] = [
    # (name, env_key, search_function)
    ("gnews", "GNEWS_API_KEY", search_gnews),
    ("guardian", "GUARDIAN_API_KEY", search_guardian),
    ("newsapi", "NEWSAPI_KEY", search_newsapi),
]

# Keyless providers, on by default. These exist so an unconfigured checkout
# still retrieves real, recent, attributable coverage: previously the only
# keyless path was scraping DuckDuckGo, which is frequently blocked and is
# not a news index, so a fresh headline came back with evergreen web pages
# that looked to the user like the system had misunderstood the claim.
#
# Each can be turned off with its env flag (e.g. GOOGLE_NEWS_ENABLED=false).
KEYLESS_PROVIDERS: list[tuple[str, str, Callable, int]] = [
    # (name, env_flag, search_function, results_per_query)
    ("google_news", "GOOGLE_NEWS_ENABLED", google_news_search, 5),
    ("wikipedia", "WIKIPEDIA_ENABLED", wikipedia_search, 3),
]


def _flag_enabled(name: str, default: bool = True) -> bool:
    """Read an on-by-default boolean environment flag."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ── Deduplication ────────────────────────────────────────────────────
# Words whose presence in one headline and absence in the other means the two
# say OPPOSITE things, however similar the rest of the wording is.
#
# "Court rules Google must be banned in all US cities" and "Court rules Google
# must not be banned in all US cities" share 11 of 12 tokens — 0.92 Jaccard,
# comfortably over the near-duplicate threshold. Merging them kept whichever
# arrived first and silently discarded the other. For a fact-checker that is
# the worst possible thing to delete: surfacing contradictions is the job.
_POLARITY_MARKERS = frozenset({
    "not", "no", "never", "without", "denies", "denied", "deny", "refutes",
    "refuted", "debunked", "false", "untrue", "rejects", "rejected",
    "isnt", "arent", "wasnt", "werent", "doesnt", "dont", "didnt", "wont",
    "cannot", "cant", "hasnt", "havent",
})


def _opposite_meanings(tokens_a: set[str], tokens_b: set[str]) -> bool:
    """True when two near-identical headlines assert opposite things.

    Checks the tokens that differ between them for polarity markers, and for
    words belonging to opposite events in the shared vocabulary ("approves"
    against "rejects").
    """
    difference = tokens_a.symmetric_difference(tokens_b)
    if difference & _POLARITY_MARKERS:
        return True

    events_a = {SURFACE_TO_EVENT[t] for t in tokens_a if t in SURFACE_TO_EVENT}
    events_b = {SURFACE_TO_EVENT[t] for t in tokens_b if t in SURFACE_TO_EVENT}
    return any(ANTONYMS.get(event) in events_b for event in events_a)


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
            and not _opposite_meanings(tokens, prev)
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
def _accepts_recent_days(search_fn: Callable) -> bool:
    """Whether this provider can filter by date at all.

    Asked of the function rather than tracked in a table, so a provider that
    gains date support starts being asked for one without a second edit
    somewhere else — the kind of pair that drifts.
    """
    try:
        return "recent_days" in inspect.signature(search_fn).parameters
    except (TypeError, ValueError):
        return False


def _run_one(
    name: str, query: str, search_fn: Callable, max_results: int, enabled: bool,
    recent_days: int | None = None,
) -> tuple[ProviderDiagnostic, list[SearchResult]]:
    """Run a single provider/query pair, capturing its outcome as a diagnostic."""
    diag = ProviderDiagnostic(provider=name, query=query, enabled=enabled)
    if not enabled:
        return diag, []
    try:
        kwargs = {"max_results": max_results}
        if recent_days and _accepts_recent_days(search_fn):
            kwargs["recent_days"] = recent_days
        results = search_fn(query, **kwargs)
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
    recent_days: int | None = None,
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
    jobs: list[tuple[str, str, Callable, int, bool, int | None]] = []

    for query in queries[:4]:  # Limit to top 4 queries
        for name, env_key, search_fn in PROVIDERS:
            jobs.append((
                name, query, search_fn,
                max(2, max_per_provider), bool(os.getenv(env_key)), recent_days,
            ))
        for name, env_flag, search_fn, per_query in KEYLESS_PROVIDERS:
            # Wikipedia is a tertiary source that lags a news cycle by days, so
            # in recent mode it contributes evergreen background that crowds
            # out the coverage being looked for. It is the right source for an
            # undated claim and the wrong one for today.
            enabled = _flag_enabled(env_flag)
            if recent_days and name == "wikipedia":
                enabled = False
            jobs.append((name, query, search_fn, per_query, enabled, recent_days))
        if use_duckduckgo:
            jobs.append((
                "duckduckgo", query, ddg_search,
                max(3, max_per_provider // 2), _flag_enabled("DUCKDUCKGO_ENABLED"),
                recent_days,
            ))

    completed: list[tuple[ProviderDiagnostic, list[SearchResult]]] = []

    with ThreadPoolExecutor(max_workers=12) as pool:
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
