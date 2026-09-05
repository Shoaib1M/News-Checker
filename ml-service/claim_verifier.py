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


# Wikipedia and comparable reference works are real background evidence for
# timeless claims, but they are tertiary sources: below original reporting,
# above an unknown blog.
REFERENCE_DOMAINS = {
    "en.wikipedia.org", "wikipedia.org", "britannica.com",
}

# Aggregators serve other publishers' journalism from their own hostname.
# Tiering by that hostname would file every Reuters story that arrived via
# Google News as "unclassified", and counting independence by it would
# collapse ten different newsrooms into one "independent" group. Both are
# resolved by the publisher name the provider reports alongside the link.
AGGREGATOR_HOSTS = {"news.google.com", "google.com"}

# Publisher names as news feeds write them, mapped to the domain the tier
# tables are keyed on.
_PUBLISHER_DOMAINS = {
    "reuters": "reuters.com", "associated press": "apnews.com",
    "ap news": "apnews.com", "bbc": "bbc.com", "bbc news": "bbc.com",
    "npr": "npr.org", "the guardian": "theguardian.com",
    "guardian": "theguardian.com", "financial times": "ft.com",
    "bloomberg": "bloomberg.com", "nature": "nature.com",
    "science": "science.org", "the new york times": "nytimes.com",
    "new york times": "nytimes.com", "the washington post": "washingtonpost.com",
    "washington post": "washingtonpost.com", "the wall street journal": "wsj.com",
    "wall street journal": "wsj.com", "politifact": "politifact.com",
    "factcheck.org": "factcheck.org", "full fact": "fullfact.org",
    "snopes": "snopes.com",
}


@dataclass(frozen=True)
class SourceProfile:
    tier: str
    weight: float


def _matches_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def resolve_publisher_host(url: str, source_name: str = "") -> str:
    """Return the host that identifies who actually published this article.

    For a direct link that is the URL's own host. For an aggregator link it
    is derived from the publisher name the search provider supplied, falling
    back to a slug of that name so two different publishers never collapse
    into one identity.
    """
    host = urlparse(url).netloc.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]

    if host not in AGGREGATOR_HOSTS or not source_name:
        return host

    normalized = re.sub(r"\s+", " ", source_name.strip().lower())
    mapped = _PUBLISHER_DOMAINS.get(normalized)
    if mapped:
        return mapped
    if "." in normalized and " " not in normalized:
        return normalized  # the feed already gave us a domain
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return f"{slug}.publisher" if slug else host


def classify_source(url: str, source_name: str = "") -> SourceProfile:
    """Return a transparent source tier used for aggregation safeguards.

    ``source_name`` is the publisher the search provider reported. It is what
    makes tiering work for aggregator links, whose URL host names the
    aggregator rather than the newsroom.
    """
    host = resolve_publisher_host(url, source_name)
    path = urlparse(url).path.lower()

    if any(_matches_domain(host, domain) for domain in PRIMARY_SOURCE_DOMAINS):
        return SourceProfile("primary", 1.0)
    if any(_matches_domain(host, domain) for domain in FACT_CHECK_DOMAINS):
        return SourceProfile("fact-check", 0.95)
    if host == "reuters.com" and path.startswith("/fact-check"):
        return SourceProfile("fact-check", 0.95)
    if any(_matches_domain(host, domain) for domain in REPUTABLE_REPORTING_DOMAINS):
        return SourceProfile("reporting", 0.8)
    if any(_matches_domain(host, domain) for domain in REFERENCE_DOMAINS):
        return SourceProfile("reference", 0.5)
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
