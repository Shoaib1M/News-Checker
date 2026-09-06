"""Keyed news providers — payload normalisation and error signalling.

WHY THIS EXISTS:
NewsAPI and GNews do not return article bodies on their free tiers. They
return a ~200-character excerpt with a literal "… [+2345 chars]" marker.
Treating that as ``text`` had two consequences, both worst on the
highest-quality sources a user would actually pay for:

  - The excerpt is ~31 words. The pipeline fetched the real article only when
    text was under 30 words, so these results were never fetched and NLI
    judged them on two lines.
  - The truncation marker itself reached NLI as though it were prose.

These tests make no network calls; they drive the parsers with the response
shapes the providers actually return.
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

from providers import news_apis  # noqa: E402
from providers.news_apis import (  # noqa: E402
    _clean_excerpt,
    search_gnews,
    search_guardian,
    search_newsapi,
)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    class _Headers:
        @staticmethod
        def get_content_charset():
            return "utf-8"

    @property
    def headers(self):
        return self._Headers()


def respond(payload):
    return lambda *a, **k: _FakeResponse(json.dumps(payload).encode())


# The excerpt shape both providers actually return.
TRUNCATED = (
    "The prime minister announced on Tuesday that he would be stepping down "
    "from his role with immediate effect, ending weeks of speculation about "
    "the future of the governing coalition… [+2345 chars]"
)

NEWSAPI_PAYLOAD = {
    "status": "ok",
    "articles": [{
        "url": "https://reuters.com/world/india-pm-resigns",
        "title": "India's prime minister resigns",
        "description": "Short description.",
        "content": TRUNCATED,
        "source": {"name": "Reuters"},
    }],
}

GNEWS_PAYLOAD = {
    "totalArticles": 1,
    "articles": [{
        "url": "https://bbc.com/news/india-pm",
        "title": "PM steps down",
        "description": "Short description.",
        "content": TRUNCATED,
        "source": {"name": "BBC News"},
    }],
}

GUARDIAN_PAYLOAD = {
    "response": {
        "results": [{
            "webUrl": "https://theguardian.com/world/pm-resigns",
            "webTitle": "PM resigns",
            "fields": {
                "headline": "Prime minister resigns",
                "trailText": "The resignation ends weeks of speculation.",
                "bodyText": "Full article body. " * 200,
            },
        }]
    }
}


class TestTruncationMarker(unittest.TestCase):

    def test_the_marker_is_stripped(self):
        for raw in (
            "Text here… [+2345 chars]",
            "Text here [+12 chars]",
            "Text here...  [+1 char]",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(_clean_excerpt(raw), "Text here")

    def test_text_without_a_marker_is_untouched(self):
        self.assertEqual(_clean_excerpt("Normal text"), "Normal text")

    def test_empty_input_is_safe(self):
        self.assertEqual(_clean_excerpt(""), "")
        self.assertEqual(_clean_excerpt(None), "")


class TestExcerptsAreNotBodies(unittest.TestCase):
    """The excerpt must land in `snippet`, leaving `text` empty for a fetch."""

    def _newsapi(self):
        with patch.dict("os.environ", {"NEWSAPI_KEY": "k"}), \
             patch.object(news_apis, "urlopen", respond(NEWSAPI_PAYLOAD)):
            return search_newsapi("india pm")

    def _gnews(self):
        with patch.dict("os.environ", {"GNEWS_API_KEY": "k"}), \
             patch.object(news_apis, "urlopen", respond(GNEWS_PAYLOAD)):
            return search_gnews("india pm")

    def test_newsapi_leaves_text_empty_so_the_article_gets_fetched(self):
        result = self._newsapi()[0]
        self.assertEqual(result.text, "")

    def test_gnews_leaves_text_empty_so_the_article_gets_fetched(self):
        result = self._gnews()[0]
        self.assertEqual(result.text, "")

    def test_the_marker_never_reaches_the_snippet(self):
        for result in (self._newsapi()[0], self._gnews()[0]):
            with self.subTest(provider=result.provider):
                self.assertNotIn("chars]", result.snippet)

    def test_the_longer_of_excerpt_and_description_is_kept(self):
        """The excerpt carries more of the story than a one-line description."""
        result = self._newsapi()[0]
        self.assertIn("stepping down", result.snippet)

    def test_publisher_attribution_is_preserved(self):
        self.assertEqual(self._newsapi()[0].source, "Reuters")
        self.assertEqual(self._gnews()[0].source, "BBC News")

    def test_the_guardian_body_is_real_text_and_is_kept(self):
        """The Guardian is the one provider here that returns a full body."""
        with patch.dict("os.environ", {"GUARDIAN_API_KEY": "k"}), \
             patch.object(news_apis, "urlopen", respond(GUARDIAN_PAYLOAD)):
            result = search_guardian("india pm")[0]
        self.assertGreater(len(result.text.split()), 100)
        self.assertEqual(result.title, "Prime minister resigns")


class TestErrorPayloads(unittest.TestCase):
    """A bad key must not look like the press having nothing to say."""

    def test_a_newsapi_error_object_raises(self):
        payload = {"status": "error", "code": "apiKeyInvalid", "message": "Your API key is invalid."}
        with patch.dict("os.environ", {"NEWSAPI_KEY": "bad"}), \
             patch.object(news_apis, "urlopen", respond(payload)):
            with self.assertRaises(RuntimeError) as caught:
                search_newsapi("anything")
        self.assertIn("apiKeyInvalid", str(caught.exception))

    def test_a_gnews_errors_array_raises(self):
        payload = {"errors": ["Invalid API key"]}
        with patch.dict("os.environ", {"GNEWS_API_KEY": "bad"}), \
             patch.object(news_apis, "urlopen", respond(payload)):
            with self.assertRaises(RuntimeError):
                search_gnews("anything")

    def test_a_genuinely_empty_result_set_is_not_an_error(self):
        with patch.dict("os.environ", {"NEWSAPI_KEY": "k"}), \
             patch.object(news_apis, "urlopen", respond({"status": "ok", "articles": []})):
            self.assertEqual(search_newsapi("anything"), [])

    def test_a_missing_key_returns_nothing_without_calling_out(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(search_newsapi("anything"), [])
            self.assertEqual(search_gnews("anything"), [])
            self.assertEqual(search_guardian("anything"), [])


class TestPipelineFetchThreshold(unittest.TestCase):

    def test_an_api_excerpt_is_short_enough_to_trigger_a_full_fetch(self):
        """The regression this file exists for: 31 words vs a 30-word gate."""
        from evidence_pipeline import MIN_WORDS_WITHOUT_FETCH
        self.assertLess(
            len(_clean_excerpt(TRUNCATED).split()), MIN_WORDS_WITHOUT_FETCH,
            "a provider excerpt must count as too short to judge a claim on",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
