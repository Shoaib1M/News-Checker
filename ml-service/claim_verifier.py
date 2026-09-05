"""Claim extraction, source classification, and evidence tiering.

This module deliberately separates *finding an article about a claim* from
*establishing whether that article entails the claim*.  Search relevance is
only used to choose candidate passages; it is never used as a truth score.

NLI scoring is handled by ``nli_service.py`` — the single authoritative
NLI service.  This module no longer contains an NLI scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlparse


# These tiers are a retrieval policy, not a declaration that a source is
# always correct.  A high-quality source still has to entail the specific
# claim before it can affect a verdict.
PRIMARY_SOURCE_DOMAINS = {
    "gov", "gov.uk", "europa.eu", "who.int", "cdc.gov", "nih.gov",
    "pubmed.ncbi.nlm.nih.gov", "worldbank.org", "imf.org", "un.org",
    "oecd.org", "sec.gov", "census.gov", "bls.gov", "data.gov",
}
FACT_CHECK_DOMAINS = {
    "politifact.com", "factcheck.org", "fullfact.org", "snopes.com",
    "factcheck.afp.com", "reuters.com/fact-check",
}
REPUTABLE_REPORTING_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "theguardian.com", "ft.com", "bloomberg.com", "nature.com",
    "science.org", "nytimes.com", "washingtonpost.com", "wsj.com",
}


@dataclass(frozen=True)
class SourceProfile:
    tier: str
    weight: float


def _matches_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def classify_source(url: str) -> SourceProfile:
    """Return a transparent source tier used for aggregation safeguards."""
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.lower()

    if any(_matches_domain(host, domain) for domain in PRIMARY_SOURCE_DOMAINS):
        return SourceProfile("primary", 1.0)
    if any(_matches_domain(host, domain) for domain in FACT_CHECK_DOMAINS):
        return SourceProfile("fact-check", 0.95)
    if host == "reuters.com" and path.startswith("/fact-check"):
        return SourceProfile("fact-check", 0.95)
    if any(_matches_domain(host, domain) for domain in REPUTABLE_REPORTING_DOMAINS):
        return SourceProfile("reporting", 0.8)
    return SourceProfile("unclassified", 0.0)


def extract_claims(text: str, max_claims: int = 6) -> list[str]:
    """Extract atomic-looking declarative claims without changing user wording.

    News articles often contain lists or semicolon-separated claims.  Splitting
    those gives each claim its own retrieval and verdict.  Very short fragments
    are ignored because they cannot be checked meaningfully.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    candidates = re.split(r"(?:\n+|(?<=[.!?])\s+|;\s+)", normalized)
    claims = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip(" -•\t")
        key = candidate.lower()
        if len(candidate.split()) < 4 or key in seen:
            continue
        seen.add(key)
        claims.append(candidate)
        if len(claims) >= max_claims:
            break
    return claims or [normalized]
