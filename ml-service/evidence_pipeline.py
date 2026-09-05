"""Evidence pipeline — the clean replacement for evidence_scraper's collect_evidence.

Pipeline stages:
  1. Query generation (via query_generator.py)
  2. Multi-provider search with diagnostics (via providers/registry.py)
  3. Deduplication (via providers/registry.py)
  4. Relevance filtering (via relevance_filter.py)
  5. Article extraction + passage selection (via article_extractor.py)
  6. NLI classification (via nli_service.py)
  7. Source quality classification (via claim_verifier.py)
  8. Evidence aggregation (via evidence_aggregator.py)

A source only becomes "evidence" after step 6.  Before that it is a "candidate".
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import NamedTuple

from article_extractor import extract_article, extract_passages
from claim_verifier import classify_source
from evidence_aggregator import ClassifiedEvidence, compute_stance
from nli_service import get_nli_service
from providers import SearchResult
from providers.registry import search_all_providers
from query_generator import QueryGenerator
from relevance_filter import RelevanceFilter


@dataclass
class EvidenceResult:
    """A source that has passed through the complete pipeline."""
    url: str
    title: str
    snippet: str
    similarity: float
    text_length: int
    provider: str
    source: str
    support_score: float = 0.0
    contradiction_score: float = 0.0
    stance: str = "unclear"
    best_sentence: str = ""
    source_tier: str = "unclassified"
    source_weight: float = 0.0
    nli_available: bool = False


class PipelineOutcome(NamedTuple):
    """Everything the API needs from one claim's evidence pipeline run."""
    stance: dict
    evidence: list[EvidenceResult]
    retrieval_status: str
    diagnostics: list[dict]
    candidate_count: int
    relevant_count: int


# ── Singletons ───────────────────────────────────────────────────────
_query_generator = QueryGenerator()
_relevance_filter = RelevanceFilter()


DEFAULT_BUDGET_SECONDS = 45.0
# Share of the budget retrieval may consume before article extraction starts.
_SEARCH_BUDGET_SHARE = 0.5


def run_pipeline(
    claim: str,
    max_results: int = 8,
    fetch_articles: bool = True,
    deadline: float | None = None,
) -> PipelineOutcome:
    """Execute the complete evidence pipeline for a single claim.

    ``deadline`` is an absolute ``time.monotonic()`` value bounding the whole
    run. Every network stage checks it, so a blocked search provider or a
    stalling news site degrades this to partial results instead of running
    until the caller's HTTP timeout fires. Defaults to
    ``DEFAULT_BUDGET_SECONDS`` from now.

    Returns a PipelineOutcome containing the stance summary, classified
    evidence, retrieval status, and per-provider diagnostics.
    """
    if deadline is None:
        deadline = time.monotonic() + DEFAULT_BUDGET_SECONDS

    # ── Stage 1: Query generation ────────────────────────────────────
    query_variants = _query_generator.generate_queries(claim)
    queries = [q["query"] for q in query_variants[:4]]
    if not queries:
        queries = [claim]

    # ── Stage 2: Multi-provider search ───────────────────────────────
    # Cap search at half the budget so extraction and NLI still get a turn.
    search_deadline = min(
        deadline,
        time.monotonic() + (deadline - time.monotonic()) * _SEARCH_BUDGET_SHARE,
    )
    raw_results, diagnostics = search_all_providers(queries, deadline=search_deadline)
    diagnostics_dicts = [d.to_dict() for d in diagnostics]
    candidate_count = len(raw_results)

    if not raw_results:
        # Determine if providers failed or simply returned nothing
        any_success = any(d.status == "success" for d in diagnostics)
        status = "NO_RESULTS" if any_success else "SEARCH_FAILED"
        return PipelineOutcome(
            stance=compute_stance([]),
            evidence=[],
            retrieval_status=status,
            diagnostics=diagnostics_dicts,
            candidate_count=0,
            relevant_count=0,
        )

    # ── Stage 3: Relevance filtering ─────────────────────────────────
    documents = [
        {
            "url": r.url,
            "title": r.title,
            "snippet": r.snippet,
            "text": r.text or r.snippet,
            "source": r.source,
            "provider": r.provider,
        }
        for r in raw_results
    ]
    included, _excluded = _relevance_filter.filter_documents(
        claim, documents, strict=True
    )
    relevant_count = len(included)

    if not included:
        return PipelineOutcome(
            stance=compute_stance([]),
            evidence=[],
            retrieval_status="NO_RELEVANT_RESULTS",
            diagnostics=diagnostics_dicts,
            candidate_count=candidate_count,
            relevant_count=0,
        )

    # ── Stage 4: Article extraction ──────────────────────────────────
    # Fetched concurrently: search-result snippets are usually too short to
    # judge, so nearly every candidate needs its page pulled. Doing that one
    # at a time meant a handful of stalling news sites serialized into
    # minutes of wall time.
    selected = included[:max_results]
    fetched_articles: dict[str, tuple[str, str]] = {}

    if fetch_articles:
        needs_fetch = [
            d for d in selected
            if len((d.get("text") or "").split()) < 30
        ]
        if needs_fetch and time.monotonic() < deadline:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    (d["url"], pool.submit(extract_article, d["url"], 6))
                    for d in needs_fetch
                ]
                for url, future in futures:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        future.cancel()
                        continue
                    try:
                        fetched_articles[url] = future.result(timeout=remaining)
                    except Exception:
                        pass  # Fall back to the snippet we already have

    nli_service = get_nli_service()
    evidence_results: list[EvidenceResult] = []

    for doc in selected:
        url = doc["url"]
        title = doc.get("title", "")
        snippet = doc.get("snippet", "")
        full_text = doc.get("text", "")
        source_profile = classify_source(url)

        fetched_title, fetched_text = fetched_articles.get(url, ("", ""))
        if fetched_text and len(fetched_text.split()) > len((full_text or "").split()):
            full_text = fetched_text
        if fetched_title and not title:
            title = fetched_title

        # ── Stage 5: NLI classification ──────────────────────────────
        passages = extract_passages(title, snippet, full_text)
        support_score = 0.0
        contradiction_score = 0.0
        best_sentence = ""
        nli_available = False

        if passages and nli_service.is_available:
            nli_scores = nli_service.score_many(claim, passages)
            if nli_scores and nli_scores[0].get("available"):
                nli_available = True
                best_idx = 0
                best_strength = 0.0
                for i, score in enumerate(nli_scores):
                    strength = max(score["entailment"], score["contradiction"])
                    if strength > best_strength:
                        best_strength = strength
                        best_idx = i
                best = nli_scores[best_idx]
                support_score = best["entailment"]
                contradiction_score = best["contradiction"]
                best_sentence = passages[best_idx] if best_idx < len(passages) else ""

        # Determine stance
        if nli_available:
            if support_score > contradiction_score and support_score > 0.35:
                stance = "supports"
            elif contradiction_score > support_score and contradiction_score > 0.35:
                stance = "contradicts"
            else:
                stance = "unclear"
        else:
            stance = "unclear"

        evidence_results.append(EvidenceResult(
            url=url,
            title=title,
            snippet=snippet,
            similarity=0.0,  # TF-IDF similarity not used as a score
            text_length=len(full_text or ""),
            provider=doc.get("provider", "unknown"),
            source=doc.get("source", ""),
            support_score=support_score,
            contradiction_score=contradiction_score,
            stance=stance,
            best_sentence=best_sentence,
            source_tier=source_profile.tier,
            source_weight=source_profile.weight,
            nli_available=nli_available,
        ))

    # ── Stage 6: Evidence aggregation ────────────────────────────────
    classified = [
        ClassifiedEvidence(
            url=e.url,
            source=e.source,
            source_tier=e.source_tier,
            source_weight=e.source_weight,
            support_score=e.support_score,
            contradiction_score=e.contradiction_score,
            nli_available=e.nli_available,
            stance=e.stance,
        )
        for e in evidence_results
    ]
    stance = compute_stance(classified)

    # Determine overall retrieval status
    any_success = any(d.status == "success" for d in diagnostics)
    all_success = all(
        d.status in {"success", "no_results"} or not d.enabled
        for d in diagnostics
    )
    if all_success:
        retrieval_status = "SEARCH_SUCCESS"
    elif any_success:
        retrieval_status = "SEARCH_PARTIAL"
    else:
        retrieval_status = "SEARCH_FAILED"

    # Add retrieval metadata to stance
    stance["retrieval_status"] = retrieval_status
    stance["retrieval_diagnostics"] = diagnostics_dicts
    stance["candidate_count"] = candidate_count
    stance["relevant_source_count"] = relevant_count

    return PipelineOutcome(
        stance=stance,
        evidence=evidence_results,
        retrieval_status=retrieval_status,
        diagnostics=diagnostics_dicts,
        candidate_count=candidate_count,
        relevant_count=relevant_count,
    )
