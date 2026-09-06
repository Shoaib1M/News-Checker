"""DuckDuckGo HTML parsing — the always-on fallback provider.

WHY THIS EXISTS:
DuckDuckGo is what an unconfigured checkout retrieves from, so its output
quality is the default experience. Its parser emitted a result the moment the
title link closed — but DuckDuckGo puts the snippet element *after* the title
link, so every result came back with an empty snippet.

That is not a cosmetic loss. With no snippet and no body text, the relevance
filter scored these candidates on the headline alone, and the pipeline had to
fetch every one of them to see anything at all.
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from providers import duckduckgo  # noqa: E402
from providers.duckduckgo import _DDGParser, _clean_url, search  # noqa: E402


# DuckDuckGo's html endpoint markup, trimmed: title link, then snippet.
RESULTS_HTML = """
<div class="result results_links">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="/l/?uddg=https%3A%2F%2Freuters.com%2Fa">India PM resigns</a>
  </h2>
  <a class="result__snippet" href="/l/?uddg=https%3A%2F%2Freuters.com%2Fa">
     The prime minister resigned on Tuesday after coalition talks failed.</a>
</div>
<div class="result results_links">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="/l/?uddg=https%3A%2F%2Fbbc.com%2Fb">PM steps down</a>
  </h2>
  <a class="result__snippet" href="/l/?uddg=https%3A%2F%2Fbbc.com%2Fb">
     He stepped down amid coalition infighting, the BBC understands.</a>
</div>
"""


def parse(html: str) -> list[dict]:
    parser = _DDGParser()
    parser.feed(html)
    parser.close()
    return parser.results


class TestSnippetsAreCaptured(unittest.TestCase):

    def test_every_result_has_its_snippet(self):
        results = parse(RESULTS_HTML)
        self.assertEqual(len(results), 2)
        for result in results:
            with self.subTest(url=result["url"]):
                self.assertTrue(
                    result["snippet"],
                    "the snippet element follows the title link, so a result "
                    "cannot be finished when the title closes",
                )

    def test_the_snippet_belongs_to_the_right_result(self):
        results = parse(RESULTS_HTML)
        self.assertIn("coalition talks failed", results[0]["snippet"])
        self.assertIn("BBC understands", results[1]["snippet"])

    def test_the_last_result_is_not_dropped(self):
        """Nothing follows it to trigger a flush, so close() must emit it."""
        self.assertEqual(len(parse(RESULTS_HTML)), 2)

    def test_titles_and_snippets_do_not_bleed_together(self):
        results = parse(RESULTS_HTML)
        self.assertEqual(results[0]["title"], "India PM resigns")
        self.assertNotIn("prime minister", results[0]["title"])

    def test_a_result_without_a_title_is_not_emitted(self):
        html = """<a class="result__a" href="/l/?uddg=https%3A%2F%2Fx.com%2Fa"></a>"""
        self.assertEqual(parse(html), [])

    def test_a_snippet_in_a_div_is_also_captured(self):
        html = """
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fx.com%2Fa">Headline</a>
        <div class="result__snippet">Body text for the result goes here.</div>
        """
        results = parse(html)
        self.assertEqual(len(results), 1)
        self.assertIn("Body text", results[0]["snippet"])


class TestUrlUnwrapping(unittest.TestCase):

    def test_the_redirect_wrapper_is_removed(self):
        self.assertEqual(
            _clean_url("/l/?uddg=https%3A%2F%2Freuters.com%2Fworld%2Fstory"),
            "https://reuters.com/world/story",
        )

    def test_a_direct_url_is_left_alone(self):
        self.assertEqual(_clean_url("https://reuters.com/a"), "https://reuters.com/a")

    def test_an_empty_href_is_safe(self):
        self.assertEqual(_clean_url(""), "")


class TestSearchNormalisation(unittest.TestCase):

    def _search(self, html=RESULTS_HTML, **kwargs):
        class _Response(io.BytesIO):
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                self_inner.close()
                return False

            class _Headers:
                @staticmethod
                def get_content_charset():
                    return "utf-8"

            @property
            def headers(self_inner):
                return self_inner._Headers()

        with patch.object(duckduckgo, "urlopen", lambda *a, **k: _Response(html.encode())):
            return search("india pm resigned", **kwargs)

    def test_results_carry_their_snippets_through(self):
        results = self._search()
        self.assertEqual(len(results), 2)
        self.assertIn("coalition talks failed", results[0].snippet)

    def test_the_publisher_is_taken_from_the_unwrapped_url(self):
        self.assertEqual(self._search()[0].source, "reuters.com")

    def test_duplicate_urls_are_collapsed(self):
        doubled = RESULTS_HTML + RESULTS_HTML
        self.assertEqual(len(self._search(doubled)), 2)

    def test_max_results_is_respected(self):
        self.assertEqual(len(self._search(max_results=1)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
