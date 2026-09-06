"""Passage selection and HTML parsing — the narrowest point in the pipeline.

WHY THIS EXISTS:
NLI only ever sees the passages ``extract_passages`` returns. A sentence it
drops cannot support or contradict anything, however decisive it is, so a
selection bug here presents as a *stance* bug several stages later: the
article is scored neutral, filed as coverage that does not address the claim,
and — for a high-salience claim — can push the aggregator into concluding
nobody reported an event that was in fact reported.
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

import article_extractor  # noqa: E402
from article_extractor import (  # noqa: E402
    _ArticleParser,
    extract_article,
    extract_passages,
    split_sentences,
)


# A news story shaped the way news stories actually are: seven sentences of
# scene-setting, then the fact.
BURIED_LEDE = " ".join([
    "The meeting began early on Tuesday in the capital city.",
    "Officials from several departments attended the session.",
    "Reporters gathered outside the building from dawn.",
    "The agenda covered a wide range of economic topics.",
    "Analysts had expected a routine set of announcements.",
    "Markets were broadly flat ahead of the statement.",
    "Security was tight around the perimeter all morning.",
    "The prime minister resigned, effective immediately, citing health reasons.",
    "His deputy will serve in an acting capacity.",
])


class TestPassageSelection(unittest.TestCase):

    def test_the_decisive_sentence_is_not_crowded_out_by_the_opening(self):
        passages = extract_passages(
            "Cabinet meets in the capital",
            "Officials gathered Tuesday.",
            BURIED_LEDE,
            claim="The prime minister of India resigned this morning",
        )
        self.assertTrue(
            any("resigned" in p for p in passages),
            "the sentence reporting the claimed event must reach NLI",
        )

    def test_title_and_snippet_always_lead(self):
        passages = extract_passages(
            "Cabinet meets in the capital", "Officials gathered Tuesday.",
            BURIED_LEDE, claim="prime minister resigned",
        )
        self.assertEqual(passages[0], "Cabinet meets in the capital")
        self.assertEqual(passages[1], "Officials gathered Tuesday.")

    def test_without_a_claim_it_degrades_to_document_order(self):
        """No claim ⇒ nothing to rank by ⇒ behave exactly as before."""
        passages = extract_passages("T", "S", BURIED_LEDE)
        self.assertEqual(passages[2], "The meeting began early on Tuesday in the capital city.")
        self.assertEqual(passages[3], "Officials from several departments attended the session.")

    def test_a_claim_sharing_no_vocabulary_also_degrades_to_order(self):
        passages = extract_passages("T", "S", BURIED_LEDE, claim="quantum chromodynamics lattice")
        self.assertEqual(passages[2], "The meeting began early on Tuesday in the capital city.")

    def test_duplicate_title_and_snippet_do_not_waste_a_slot(self):
        """Google News descriptions frequently restate the headline verbatim."""
        passages = extract_passages(
            "India's prime minister resigns", "india's prime minister resigns!",
            BURIED_LEDE, claim="prime minister resigned",
        )
        self.assertEqual(passages[0], "India's prime minister resigns")
        self.assertNotIn("resigns!", passages[1])

    def test_the_passage_cap_is_respected(self):
        passages = extract_passages("T", "S", BURIED_LEDE, max_passages=4, claim="resigned")
        self.assertEqual(len(passages), 4)

    def test_empty_inputs_do_not_produce_empty_passages(self):
        self.assertEqual(extract_passages("", "", ""), [])
        self.assertEqual(extract_passages("", "", "", claim="anything"), [])


class TestSentenceSplitting(unittest.TestCase):

    def test_abbreviations_do_not_split_sentences(self):
        sentences = split_sentences(
            "Dr. Smith met the U.S. delegation on Monday morning in Geneva. "
            "The talks lasted several hours without a formal agreement."
        )
        self.assertEqual(len(sentences), 2)
        self.assertIn("U.S.", sentences[0])

    def test_fragments_are_dropped(self):
        self.assertEqual(split_sentences("Yes. No. Maybe."), [])


class TestHtmlParsing(unittest.TestCase):

    def test_void_elements_do_not_desynchronise_the_tag_stack(self):
        parser = _ArticleParser()
        parser.feed(
            "<html><body>"
            "<p>First paragraph with enough words here to count ok</p>"
            "<p>Second paragraph <br> continues past the break with more words</p>"
            "<div>NAVIGATION MENU COOKIE BANNER SUBSCRIBE NOW FOOTER JUNK</div>"
            "<p>Third real paragraph of the article with plenty of words</p>"
            "</body></html>"
        )
        self.assertEqual(len(parser.paragraphs), 3)
        self.assertFalse(
            any("COOKIE BANNER" in p for p in parser.paragraphs),
            "navigation text outside <p> must never enter the article body",
        )
        self.assertEqual(parser._tag_stack, [])

    def test_an_unclosed_tag_does_not_corrupt_everything_after_it(self):
        parser = _ArticleParser()
        parser.feed(
            "<html><body><div><span>"
            "<p>A paragraph of real article text with enough words to keep</p>"
            "</div>"
            "<p>Another paragraph of real article text with enough words</p>"
            "</body></html>"
        )
        self.assertEqual(len(parser.paragraphs), 2)

    def test_the_title_is_captured(self):
        parser = _ArticleParser()
        parser.feed("<html><head><title>Headline here</title></head><body></body></html>")
        self.assertEqual(parser.title, "Headline here")


class _FakeResponse(io.BytesIO):
    def __init__(self, payload, content_type="text/html", charset="utf-8"):
        super().__init__(payload)
        self._content_type = content_type
        self._charset = charset

    class _Headers:
        def __init__(self, ct, cs):
            self._ct, self._cs = ct, cs

        def get_content_type(self):
            return self._ct

        def get_content_charset(self):
            return self._cs

    @property
    def headers(self):
        return self._Headers(self._content_type, self._charset)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class TestFetchGuards(unittest.TestCase):

    def test_non_html_responses_are_refused(self):
        """A search result pointing at a PDF must not be decoded as prose."""
        with patch.object(
            article_extractor, "urlopen",
            lambda *a, **k: _FakeResponse(b"%PDF-1.7 binary", content_type="application/pdf"),
        ):
            with self.assertRaises(Exception):
                extract_article("https://example.com/report.pdf")

    def test_oversized_pages_are_truncated_not_streamed_whole(self):
        html = b"<html><body><p>" + b"word " * 200 + b"</p></body></html>"
        with patch.object(article_extractor, "MAX_ARTICLE_BYTES", 64), \
             patch.object(article_extractor, "urlopen",
                          lambda *a, **k: _FakeResponse(html)):
            _title, text = extract_article("https://example.com/big")
        self.assertLess(len(text), 200)

    def test_html_is_parsed_normally(self):
        html = (b"<html><head><title>Real headline</title></head><body>"
                b"<p>The prime minister resigned on Tuesday according to two officials.</p>"
                b"</body></html>")
        with patch.object(article_extractor, "urlopen", lambda *a, **k: _FakeResponse(html)):
            title, text = extract_article("https://example.com/story")
        self.assertEqual(title, "Real headline")
        self.assertIn("prime minister resigned", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestClaimSplitting(unittest.TestCase):
    """Splitting *user input* — the same abbreviation trap, higher stakes.

    article_extractor protected abbreviations; claim_verifier had its own
    splitter without that protection, so the guard existed in the codebase but
    not where the user's own words were being cut up.
    """

    def claims(self, statement):
        from claim_verifier import extract_claims
        return extract_claims(statement, max_claims=3)

    def test_a_country_abbreviation_does_not_end_a_sentence(self):
        """"The U.S. government banned Google" lost its subject entirely."""
        self.assertEqual(
            self.claims("The U.S. government banned Google across all cities in 2024."),
            ["The U.S. government banned Google across all cities in 2024."],
        )

    def test_titles_and_months_do_not_end_a_sentence(self):
        for statement in (
            "Dr. Smith announced the vaccine trial results on Tuesday morning.",
            "Inflation rose 3.2% in Aug. and is expected to keep climbing.",
            "Sen. Warren said the U.K. and E.U. agreed on Sept. 3 to new rules.",
        ):
            with self.subTest(statement=statement):
                self.assertEqual(self.claims(statement), [statement])

    def test_a_split_that_orphans_a_fragment_is_abandoned(self):
        """Dropping "Apple, Google" silently checked only Microsoft."""
        statement = "Apple, Google; and Microsoft were all fined by regulators this year."
        self.assertEqual(self.claims(statement), [statement])

    def test_genuine_multi_claim_statements_still_split(self):
        self.assertEqual(
            self.claims("The prime minister resigned this morning. "
                        "The finance minister was arrested."),
            ["The prime minister resigned this morning.",
             "The finance minister was arrested."],
        )

    def test_a_single_claim_is_returned_whole(self):
        statement = "The prime minister of India resigned this morning"
        self.assertEqual(self.claims(statement), [statement])

    def test_empty_input_yields_no_claims(self):
        self.assertEqual(self.claims("   "), [])
