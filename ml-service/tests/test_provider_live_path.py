"""Providers fetched over a real socket, against a real HTTP server.

WHY THIS EXISTS:
The other provider tests stub `urlopen`, so they verify parsing but never
execute the fetch: the URL construction, the request headers, the HTTP round
trip, the charset handling, and the size cap are all skipped. A provider could
be built with a malformed URL or a header a server rejects and every one of
those tests would still pass.

This serves the real payloads from a local HTTP server on 127.0.0.1 and points
each provider at it. Everything except the remote host being reachable is
exercised for real — a genuine socket, a genuine HTTP response, genuine
decoding.

WHAT IT STILL CANNOT TELL YOU:
whether news.google.com is reachable from your machine, or whether your API
keys are valid. Nothing offline can. `python check_providers.py` answers that
against the live internet in about ten seconds.
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import article_extractor  # noqa: E402
from providers import google_news, wikipedia  # noqa: E402


GOOGLE_NEWS_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item>
  <title>India's prime minister resigns after coalition talks fail - Reuters</title>
  <link>https://news.google.com/rss/articles/CBMiaHR0cHM6</link>
  <description>&lt;a href="x"&gt;The prime minister resigned on Tuesday.&lt;/a&gt;</description>
  <source url="https://www.reuters.com">Reuters</source>
</item>
<item>
  <title>Delhi reacts to the resignation - The Times of India</title>
  <link>https://news.google.com/rss/articles/QkJDaGVsbG8</link>
  <description>Reaction from across the capital.</description>
  <source url="https://timesofindia.indiatimes.com">The Times of India</source>
</item>
</channel></rss>"""

WIKIPEDIA_JSON = json.dumps({
    "query": {"search": [
        {"title": "Eiffel Tower",
         "snippet": 'The <span class="searchmatch">Eiffel</span> Tower is owned by Paris.'},
        {"title": "List of tallest structures", "snippet": "Ranked by height."},
    ]}
}).encode()

ARTICLE_HTML = (
    b"<html><head><title>India's prime minister resigns</title></head><body>"
    b"<p>The prime minister resigned on Tuesday after coalition talks failed.</p>"
    b"<p>His deputy will serve in an acting capacity until a vote is held.</p>"
    b"</body></html>"
)


class _Handler(BaseHTTPRequestHandler):
    """Serves each payload on its own path, recording the requests it saw."""

    requests: list[tuple[str, dict]] = []

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's interface
        type(self).requests.append((self.path, dict(self.headers)))
        if self.path.startswith("/rss"):
            body, content_type = GOOGLE_NEWS_RSS, "application/rss+xml; charset=utf-8"
        elif self.path.startswith("/w/api.php"):
            body, content_type = WIKIPEDIA_JSON, "application/json; charset=utf-8"
        elif self.path.startswith("/article"):
            body, content_type = ARTICLE_HTML, "text/html; charset=utf-8"
        elif self.path.startswith("/binary"):
            body, content_type = b"%PDF-1.7 not text at all", "application/pdf"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep the test output clean


class LiveFetchPathTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 127.0.0.1 is in the proxy's noProxy list, so this is a direct socket.
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Handler.requests.clear()

    # ── Google News ──────────────────────────────────────────────────
    def test_google_news_fetches_and_parses_over_a_real_socket(self):
        with patch.object(google_news, "FEED_URL", self.base + "/rss?q={query}"):
            results = google_news.search("india prime minister resigned")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title,
                         "India's prime minister resigns after coalition talks fail")
        self.assertEqual(results[0].source, "Reuters")
        self.assertIn("resigned on Tuesday", results[0].snippet)

    def test_the_query_reaches_the_server_url_encoded(self):
        with patch.object(google_news, "FEED_URL", self.base + "/rss?q={query}"):
            google_news.search('india "prime minister" resigned')
        path, _headers = _Handler.requests[0]
        self.assertIn("prime", path)
        self.assertNotIn(" ", path, "spaces must be percent-encoded")

    def test_a_user_agent_is_sent(self):
        """Some feeds reject the default Python user agent outright."""
        with patch.object(google_news, "FEED_URL", self.base + "/rss?q={query}"):
            google_news.search("test")
        _path, headers = _Handler.requests[0]
        self.assertIn("Mozilla", headers.get("User-Agent", ""))

    # ── Wikipedia ────────────────────────────────────────────────────
    def test_wikipedia_fetches_and_parses_over_a_real_socket(self):
        with patch.object(wikipedia, "API_URL", self.base + "/w/api.php"):
            results = wikipedia.search("eiffel tower ownership")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Eiffel Tower")
        self.assertIn("owned by Paris", results[0].snippet)
        self.assertNotIn("searchmatch", results[0].snippet)

    def test_wikipedia_sends_its_identifying_user_agent(self):
        """Wikimedia rate-limits anonymous default agents more aggressively."""
        with patch.object(wikipedia, "API_URL", self.base + "/w/api.php"):
            wikipedia.search("test")
        _path, headers = _Handler.requests[0]
        self.assertIn("NewsChecker", headers.get("User-Agent", ""))

    def test_search_operators_are_stripped_before_the_request(self):
        with patch.object(wikipedia, "API_URL", self.base + "/w/api.php"):
            wikipedia.search('"Elon Musk" "Eiffel Tower" (false OR debunked)')
        path, _headers = _Handler.requests[0]
        self.assertNotIn("%22", path, "quotes should not reach Wikipedia search")

    # ── Article extraction ───────────────────────────────────────────
    def test_an_article_is_fetched_and_its_text_extracted(self):
        title, text = article_extractor.extract_article(self.base + "/article")
        self.assertEqual(title, "India's prime minister resigns")
        self.assertIn("prime minister resigned on Tuesday", text)

    def test_a_non_html_response_is_refused_over_a_real_socket(self):
        """The content-type guard has to work against a real server's headers."""
        with self.assertRaises(Exception):
            article_extractor.extract_article(self.base + "/binary")

    def test_a_missing_page_raises_rather_than_returning_empty_text(self):
        with self.assertRaises(Exception):
            article_extractor.extract_article(self.base + "/nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
