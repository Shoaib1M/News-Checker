"""Claim extraction, source classification, and NLI-backed evidence scoring.

This module deliberately separates *finding an article about a claim* from
*establishing whether that article entails the claim*.  Search relevance is
only used to choose candidate passages; it is never used as a truth score.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Callable, Iterable
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


class NLIScorer:
    """Lazy Hugging Face NLI wrapper.

    Importing/downloading the model is deferred until evidence is available.
    If the model cannot load, callers receive an explicit unavailable result;
    the application must abstain instead of reverting to keyword heuristics.
    """

    def __init__(self, pipeline_factory: Callable | None = None, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "NLI_MODEL", "cross-encoder/nli-deberta-v3-small"
        )
        self._pipeline_factory = pipeline_factory
        self._pipeline = None
        self.error: str | None = None

    def _load(self):
        if self._pipeline is not None or self.error is not None:
            return
        # The default Render instance has limited RAM. Keep the optional
        # transformer model disabled there unless explicitly enabled.
        if self._pipeline_factory is None and os.getenv("NLI_ENABLED", "false").lower() not in {
            "1", "true", "yes", "on"
        }:
            self.error = "NLI disabled; set NLI_ENABLED=true to enable it"
            return
        try:
            factory = self._pipeline_factory
            if factory is None:
                from transformers import pipeline
                factory = pipeline
            self._pipeline = factory(
                "text-classification", model=self.model_name, tokenizer=self.model_name,
                device=-1,
            )
        except Exception as error:  # model/network failures must cause abstention
            self.error = str(error)

    @staticmethod
    def _normalise_label(label: str) -> str:
        label = label.lower().replace("_", " ")
        if "entail" in label or label in {"label 2", "label_2"}:
            return "entailment"
        if "contradict" in label or label in {"label 0", "label_0"}:
            return "contradiction"
        return "neutral"

    def score_many(self, claim: str, passages: Iterable[str]) -> list[dict]:
        passages = list(passages)
        if not passages:
            return []
        self._load()
        if self._pipeline is None:
            return [
                {
                    "entailment": 0.0,
                    "contradiction": 0.0,
                    "neutral": 1.0,
                    "available": False,
                }
                for _ in passages
            ]

        pairs = [{"text": passage, "text_pair": claim} for passage in passages]
        try:
            results = self._pipeline(pairs, top_k=None, truncation=True, max_length=512)
        except TypeError:
            # Keeps injected test doubles and older pipeline versions usable.
            results = [self._pipeline(pair) for pair in pairs]
        except Exception as error:
            self.error = str(error)
            return self.score_many(claim, passages)

        scores = []
        for output in results:
            if isinstance(output, dict):
                output = [output]
            values = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            for item in output:
                values[self._normalise_label(str(item.get("label", "")))] = float(item.get("score", 0.0))
            values["available"] = True
            scores.append(values)
        return scores
