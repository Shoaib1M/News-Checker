"""Parser tests for the keyless search providers.

SCOPE — read this before trusting these tests:
These verify the *parsing and normalisation* of provider payloads against
fixtures that reproduce the real response shapes. They deliberately do not
make network calls, so they prove the code handles a Google News RSS document
or a Wikipedia search response correctly; they do NOT prove either endpoint is
reachable from a given machine. Run `python -m providers.google_news` style
checks, or the live smoke check in the README, to confirm reachability.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from claim_verifier import classify_source, resolve_publisher_host  # noqa: E402
from providers import google_news, wikipedia  # noqa: E402


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for the object urlopen returns as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# A real Google News RSS document, trimmed to three items. Note the two things
# that matter: links are news.google.com redirects, and every title carries a
# " - Publisher" suffix.
GOOGLE_NEWS_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>India prime minister - Google News</title>
<item>
  <title>India's prime minister resigns after coalition talks fail - Reuters</title>
  <link>https://news.google.com/rss/articles/CBMiaHR0cHM6Ly9leGFtcGxl?oc=5</link>
  <description>&lt;a href="https://news.google.com/x"&gt;India's prime minister resigns&lt;/a&gt;&amp;nbsp;&lt;font&gt;Reuters&lt;/font&gt;</description>
  <source url="https://www.reuters.com">Reuters</source>
</item>
<item>
  <title>Delhi reacts to resignation - The Times of India</title>
  <link>https://news.google.com/rss/articles/QkJDaGVsbG8?oc=5</link>
  <description>Reaction from across the capital.</description>
  <source url="https://timesofindia.indiatimes.com">The Times of India</source>
</item>
<item>
  <title>Analysis: what happens next - BBC</title>
  <link>https://news.google.com/rss/articles/QW5hbHlzaXM?oc=5</link>
  <description>What the resignation means.</description>
</item>
<item>
  <title>Google banned in US - what it means for you</title>
  <link>https://news.google.com/rss/articles/V2hhdEl0TWVhbnM?oc=5</link>
  <description>The consequences explained.</description>
</item>
</channel></rss>"""

WIKIPEDIA_JSON = json.dumps({
    "batchcomplete": True,
    "query": {
        "search": [
            {"title": "Eiffel Tower",
             "snippet": 'The <span class="searchmatch">Eiffel</span> Tower is owned by the city of Paris.'},
            {"title": "List of tallest structures in France", "snippet": "Structures ranked by height."},
        ]
    },
}).encode()


class TestGoogleNewsParsing(unittest.TestCase):

    def _search(self, payload=GOOGLE_NEWS_RSS, **kwargs):
        with patch.object(google_news, "urlopen", lambda *a, **k: _FakeResponse(payload)):
            return google_news.search("india prime minister", **kwargs)

    def test_parses_every_item(self):
        results = self._search()
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.provider == "google_news" for r in results))

    def test_publisher_suffix_is_stripped_from_the_title(self):
        """Left in, " - Reuters" leaks into the passages NLI reads."""
        results = self._search()
        self.assertEqual(
            results[0].title,
            "India's prime minister resigns after coalition talks fail",
        )
        self.assertEqual(results[0].source, "Reuters")

    def test_publisher_falls_back_to_the_title_suffix(self):
        """The third item has no <source> element."""
        results = self._search()
        self.assertEqual(results[2].source, "BBC")
        self.assertEqual(results[2].title, "Analysis: what happens next")

    def test_description_html_is_stripped(self):
        results = self._search()
        self.assertNotIn("<", results[0].snippet)
        self.assertNotIn("</a>", results[0].snippet)

    def test_max_results_is_respected(self):
        self.assertEqual(len(self._search(max_results=2)), 2)

    def test_a_headline_clause_is_not_mistaken_for_a_publisher(self):
        """"- what it means for you" is part of the headline, not a masthead.

        Treating it as one truncated the headline — which NLI reads as a
        passage — and invented a publisher called "what it means for you",
        which then counted as a distinct independent source and inflated
        confidence in the verdict.
        """
        result = self._search()[3]
        self.assertEqual(result.title, "Google banned in US - what it means for you")
        self.assertNotIn("what it means", result.source)

    def test_a_declared_source_wins_over_a_dash_clause(self):
        """The <source> element is authoritative when Google provides it."""
        results = self._search()
        self.assertEqual(results[0].source, "Reuters")

    def test_network_failure_propagates(self):
        """The registry must record a diagnostic, not read silence as an empty press."""
        def boom(*a, **k):
            raise OSError("connection reset")
        with patch.object(google_news, "urlopen", boom):
            with self.assertRaises(OSError):
                google_news.search("anything")

    def test_malformed_feed_raises_rather_than_returning_nothing(self):
        with patch.object(google_news, "urlopen",
                          lambda *a, **k: _FakeResponse(b"<not xml")):
            with self.assertRaises(Exception):
                google_news.search("anything")


class TestAggregatorPublisherResolution(unittest.TestCase):
    """Aggregator links must not collapse every newsroom into one identity."""

    def test_tier_comes_from_the_publisher_not_the_redirect_host(self):
        profile = classify_source(
            "https://news.google.com/rss/articles/abc", "Reuters"
        )
        self.assertEqual(profile.tier, "reporting")
        self.assertGreater(profile.weight, 0)

    def test_two_publishers_behind_one_aggregator_stay_distinct(self):
        first = resolve_publisher_host("https://news.google.com/rss/a", "Reuters")
        second = resolve_publisher_host("https://news.google.com/rss/b", "The Times of India")
        self.assertNotEqual(first, second)

    def test_unknown_publisher_still_gets_a_stable_distinct_identity(self):
        host = resolve_publisher_host("https://news.google.com/rss/a", "Some Local Paper")
        self.assertEqual(host, "some-local-paper.publisher")

    def test_direct_links_are_unaffected(self):
        self.assertEqual(
            resolve_publisher_host("https://www.apnews.com/story", "AP News"),
            "apnews.com",
        )


class TestWikipediaParsing(unittest.TestCase):

    def _search(self, payload=WIKIPEDIA_JSON, **kwargs):
        with patch.object(wikipedia, "urlopen", lambda *a, **k: _FakeResponse(payload)):
            return wikipedia.search("eiffel tower ownership", **kwargs)

    def test_builds_real_article_urls(self):
        results = self._search()
        self.assertEqual(results[0].url, "https://en.wikipedia.org/wiki/Eiffel_Tower")

    def test_search_match_markup_is_stripped(self):
        results = self._search()
        self.assertNotIn("searchmatch", results[0].snippet)
        self.assertIn("owned by the city of Paris", results[0].snippet)

    def test_reference_tier_is_below_reporting(self):
        reference = classify_source("https://en.wikipedia.org/wiki/Eiffel_Tower")
        reporting = classify_source("https://reuters.com/story")
        self.assertEqual(reference.tier, "reference")
        self.assertLess(reference.weight, reporting.weight)
        self.assertGreater(reference.weight, 0)

    def test_generated_query_operators_are_removed(self):
        """Boolean syntax that helps a news index matches nothing on Wikipedia."""
        cleaned = wikipedia._plain_query('"Elon Musk" "Eiffel Tower" (false OR debunked)')
        self.assertNotIn('"', cleaned)
        self.assertNotIn("(", cleaned)
        self.assertNotIn(" OR ", cleaned)
        self.assertIn("Elon Musk", cleaned)

    def test_empty_result_set_is_not_an_error(self):
        payload = json.dumps({"query": {"search": []}}).encode()
        self.assertEqual(self._search(payload), [])


class TestRegistryWiring(unittest.TestCase):

    def test_keyless_providers_are_on_by_default(self):
        from providers import registry
        names = {name for name, _flag, _fn, _n in registry.KEYLESS_PROVIDERS}
        self.assertEqual(names, {"google_news", "wikipedia"})
        for _name, flag, _fn, _n in registry.KEYLESS_PROVIDERS:
            with patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop(flag, None)
                self.assertTrue(registry._flag_enabled(flag))

    def test_a_keyless_provider_can_be_switched_off(self):
        from providers import registry
        with patch.dict("os.environ", {"WIKIPEDIA_ENABLED": "false"}):
            self.assertFalse(registry._flag_enabled("WIKIPEDIA_ENABLED"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
