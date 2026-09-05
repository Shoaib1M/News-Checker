"""Regression tests for the evidence pipeline's time budget.

These exist because of a real production failure: a request would hang long
enough that the Node proxy gave up with "ML service timed out." The pipeline
had no overall deadline, so a blocked search provider cost ~138s for a single
claim (4 queries x 3 attempts x 10s + backoff, all sequential), and a
multi-claim statement multiplied that by the number of claims.

The guarantee under test: however badly the network misbehaves, the pipeline
returns within its budget and reports honest diagnostics, rather than running
until something upstream times out.
"""

import sys
import time
from pathlib import Path
import unittest
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from evidence_pipeline import run_pipeline
from providers import SearchResult
import providers.registry as registry


def _hanging_urlopen(request, timeout=10, **kwargs):
    """Simulate a socket that stalls until the caller's timeout expires."""
    time.sleep(timeout)
    raise TimeoutError("simulated network hang")


class PipelineBudgetTests(unittest.TestCase):
    def test_blocked_search_returns_within_budget(self):
        """A totally blocked search provider must not outlive the budget."""
        budget = 3.0
        with patch("providers.duckduckgo.urlopen", _hanging_urlopen), \
             patch("article_extractor.urlopen", _hanging_urlopen), \
             patch("providers.duckduckgo.time.sleep", lambda s: None):
            start = time.monotonic()
            outcome = run_pipeline(
                "The United States is banning google across all its countries.",
                max_results=8, fetch_articles=True,
                deadline=time.monotonic() + budget,
            )
            elapsed = time.monotonic() - start

        # Generous margin for scheduling, but far below the ~138s the old
        # sequential-with-retries path took for this exact case.
        self.assertLess(elapsed, budget + 8.0)
        self.assertEqual(outcome.retrieval_status, "SEARCH_FAILED")

    def test_hanging_article_pages_return_within_budget(self):
        """Search succeeding but every article page stalling is also bounded."""
        fake_results = [
            SearchResult(
                url=f"https://news{i}.example.com/google-ban-{i}",
                title=title,
                snippet="US officials are considering banning Google nationwide.",
                provider="duckduckgo", source=f"news{i}.example.com",
            )
            for i, title in enumerate([
                "US government considers banning Google in antitrust move",
                "White House weighs Google ban as regulators escalate",
                "Justice Department pushes to ban Google search deals",
                "Congress debates banning Google across federal agencies",
            ])
        ]
        budget = 3.0
        with patch.object(registry, "ddg_search", lambda q, max_results=10: fake_results), \
             patch("article_extractor.urlopen", _hanging_urlopen):
            start = time.monotonic()
            run_pipeline(
                "The United States is banning google across all its countries.",
                max_results=8, fetch_articles=True,
                deadline=time.monotonic() + budget,
            )
            elapsed = time.monotonic() - start

        self.assertLess(elapsed, budget + 8.0)

    def test_expired_deadline_reports_timeout_diagnostics(self):
        """An exhausted budget must surface as a diagnostic, not silence.

        A provider we never got to is not the same as a provider that
        returned nothing — conflating them would hide why evidence is thin.
        """
        with patch.object(registry, "ddg_search", lambda q, max_results=10: []):
            _results, diagnostics = registry.search_all_providers(
                ["some query", "another query"],
                deadline=time.monotonic() - 1.0,  # already expired
            )

        self.assertTrue(diagnostics)
        self.assertTrue(all(d.status == "timeout" for d in diagnostics))


class MultiClaimBudgetTests(unittest.TestCase):
    def test_multi_claim_statement_shares_one_budget(self):
        """Three claims must share one budget, not take three budgets.

        The old code ran a fresh unbounded pipeline per claim, so a
        three-sentence statement cost roughly 3x the single-claim worst case.
        """
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed")

        import main
        from claim_verifier import extract_claims

        statement = (
            "The United States is banning google across all its countries. "
            "The European Union fined Apple ten billion euros last week. "
            "India announced a complete ban on TikTok yesterday."
        )
        self.assertEqual(len(extract_claims(statement, max_claims=3)), 3)

        budget = 3.0
        with patch.object(main, "EVIDENCE_BUDGET_SECONDS", budget), \
             patch("providers.duckduckgo.urlopen", _hanging_urlopen), \
             patch("article_extractor.urlopen", _hanging_urlopen), \
             patch("providers.duckduckgo.time.sleep", lambda s: None):
            with TestClient(main.app) as client:
                start = time.monotonic()
                response = client.post("/api/check", json={"statement": statement})
                elapsed = time.monotonic() - start

        self.assertEqual(response.status_code, 200)
        # One shared budget: total time tracks the budget, not budget x claims.
        self.assertLess(elapsed, budget + 10.0)

        body = response.json()
        self.assertEqual(body["retrieval"]["status"], "SEARCH_FAILED")
        self.assertEqual(body["nli"]["classified_count"], 0)
        # A blocked search is never evidence of falsehood.
        self.assertEqual(body["verdict"], "insufficient evidence")


if __name__ == "__main__":
    unittest.main()
