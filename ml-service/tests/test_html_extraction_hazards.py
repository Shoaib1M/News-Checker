"""What the HTML parser reads, and what it must refuse to read.

WHY THIS EXISTS:
Two failures found by feeding the parser the page shapes news sites actually
serve, rather than the tidy HTML the other tests use.

1. PAGE JAVASCRIPT BECAME ARTICLE PROSE. HTMLParser hands back the body of
   <script> and <style> as ordinary character data. Inline scripts sit inside
   content blocks constantly — ad slots, embeds, analytics beacons — and any
   <script> inside a <p> had its source appended to the paragraph. A config
   blob reading {"headline": "Google banned in all US cities"} was extracted
   as a sentence the publisher had written, and would be handed to NLI as
   evidence. That is the one kind of text that must never be treated as
   reporting: it is not the publisher's prose, and on many pages it is not
   even the publisher's content.

2. PAGES WITHOUT <p> CONTRIBUTED NOTHING. A body built from <div> — AMP
   templates and several large CMSs — yielded zero paragraphs, so the document
   fell back to whatever snippet the provider had, no matter what it reported.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from article_extractor import (  # noqa: E402
    _ArticleParser,
    _text_without_markup,
)

FABRICATED = "Google banned in all US cities"


def paragraphs(html: str) -> list[str]:
    parser = _ArticleParser()
    parser.feed(html)
    return parser.paragraphs


class TestNonProseElementsAreNeverRead(unittest.TestCase):

    def test_a_script_inside_a_paragraph_does_not_join_the_text(self):
        html = (
            "<html><head><title>PM resigns</title></head><body>"
            "<p>The prime minister resigned on Tuesday after the vote."
            f'<script>var d={{"headline":"{FABRICATED}"}};</script>'
            "</p></body></html>"
        )
        (paragraph,) = paragraphs(html)
        self.assertEqual(
            paragraph, "The prime minister resigned on Tuesday after the vote.")
        self.assertNotIn(FABRICATED, paragraph)

    def test_style_rules_do_not_become_sentences(self):
        html = ("<html><body><p>Officials in Delhi confirmed the resignation late on Tuesday evening."
                "<style>.hero{background:url(x);color:red}</style></p></body></html>")
        (paragraph,) = paragraphs(html)
        self.assertNotIn("background", paragraph)

    def test_a_json_ld_block_is_not_read_as_reporting(self):
        html = ("<html><body><p>The vote was held on Tuesday evening in Delhi."
                f'<script type="application/ld+json">{{"headline":"{FABRICATED}"}}'
                "</script></p></body></html>")
        self.assertNotIn(FABRICATED, " ".join(paragraphs(html)))

    def test_ordinary_paragraphs_are_untouched(self):
        html = ("<html><body><p>The prime minister resigned on Tuesday after "
                "coalition talks collapsed in Delhi.</p></body></html>")
        self.assertEqual(len(paragraphs(html)), 1)


class TestPagesWithoutParagraphElements(unittest.TestCase):

    DIV_PAGE = (
        "<html><head><title>Div article</title></head><body>"
        "<nav>Home World Business Sport Culture</nav>"
        "<div class='article-body'>"
        "<div>The minister resigned after the vote was held on Tuesday evening.</div>"
        "<div>Markets fell sharply the following day across most of Asia.</div>"
        "</div>"
        f'<script>var cfg={{"headline":"{FABRICATED}"}};</script>'
        "<footer>All rights reserved. Contact us.</footer></body></html>"
    )

    def test_the_normal_parser_finds_nothing_here(self):
        """The premise: this page contributed no text at all."""
        self.assertEqual(paragraphs(self.DIV_PAGE), [])

    def test_the_fallback_recovers_the_reporting(self):
        text = _text_without_markup(self.DIV_PAGE)
        self.assertIn("The minister resigned after the vote", text)
        self.assertIn("Markets fell sharply", text)

    def test_the_fallback_does_not_reintroduce_the_script_leak(self):
        self.assertNotIn(FABRICATED, _text_without_markup(self.DIV_PAGE))

    def test_navigation_does_not_run_into_the_first_sentence(self):
        """Without block boundaries the nav bar and the lede became one
        unsplittable sentence, so neither could be filtered from the other."""
        text = _text_without_markup(self.DIV_PAGE)
        self.assertNotIn("Home World Business", text)
        self.assertTrue(text.startswith("The minister resigned"), text[:80])

    def test_the_footer_is_left_behind(self):
        self.assertNotIn("All rights reserved", _text_without_markup(self.DIV_PAGE))

    def test_short_unpunctuated_labels_are_not_treated_as_prose(self):
        html = ("<html><body><div>Share</div><div>Politics</div>"
                "<div>Sign in</div><div>Published 4 hours ago</div></body></html>")
        self.assertEqual(_text_without_markup(html), "")

    def test_entities_are_decoded(self):
        html = ("<html><body><div>The minister&rsquo;s resignation was "
                "confirmed by his office on Tuesday.</div></body></html>")
        self.assertIn("minister’s", _text_without_markup(html))


if __name__ == "__main__":
    unittest.main(verbosity=2)
