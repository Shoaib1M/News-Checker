"""Site furniture must not reach NLI, and must not displace real text.

WHY THIS EXISTS:
A paywalled article ships a few hundred words of subscription pitch and none
of the story. Two things went wrong with that page:

  1. Passage selection sent the pitch to NLI. For "the prime minister of India
     resigned this morning", four of the five passages classified were
     "Subscribe today to continue reading this article", "Your subscription
     helps fund our newsroom", "Choose a plan that works for you" and
     "Unlimited digital access from just $1 a week".

  2. Worse, the pipeline preferred the fetched page to the provider's snippet
     whenever it was LONGER. The snippet was the one real sentence in the
     document — "India's prime minister resigned on Tuesday after coalition
     talks collapsed" — 13 words against 38 words of marketing copy. The fetch
     deleted the only usable text in the document.

The filter is phrase-based and has a claim-overlap escape, because the
dangerous failure runs the other way: an article about streaming prices is
full of the word "subscription", and a filter that ate it would delete the
story instead of the furniture.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from article_extractor import (  # noqa: E402
    extract_passages,
    is_boilerplate,
    strip_boilerplate,
)

PAYWALL_STUB = (
    "Support independent journalism. Subscribe today to continue reading this "
    "article. Already a subscriber? Sign in. Your subscription helps fund our "
    "newsroom. Choose a plan that works for you. Unlimited digital access from "
    "just 1 dollar a week. Cancel anytime."
)
RESIGNATION_CLAIM = "The prime minister of India resigned this morning"


class TestFurnitureIsRecognised(unittest.TestCase):

    def test_the_common_paywall_lines(self):
        for line in (
            "Subscribe today to continue reading this article.",
            "Already a subscriber? Sign in.",
            "Unlimited digital access from just 1 dollar a week.",
            "This article is for subscribers only.",
            "Sign up for our newsletter to get the day's headlines.",
            "We use cookies to improve your experience.",
            "Follow us on X for the latest updates.",
            "All rights reserved.",
        ):
            with self.subTest(line=line):
                self.assertTrue(is_boilerplate(line), line)

    def test_ordinary_news_sentences_are_not_furniture(self):
        for line in (
            "The prime minister resigned on Tuesday after coalition talks collapsed.",
            "Regulators confirmed no prohibition is in force.",
            "The vaccine was 95% effective in the trial.",
            "Shares fell 3% in early trading.",
        ):
            with self.subTest(line=line):
                self.assertFalse(is_boilerplate(line), line)


class TestTheClaimOverlapEscape(unittest.TestCase):
    """The filter must never delete the story it is meant to clean up."""

    def test_an_article_about_subscription_prices_keeps_its_content(self):
        claim = "Netflix raised its subscription prices in the United States"
        text = ("Netflix said its standard subscription will cost two dollars "
                "more a month. Your subscription will renew automatically at "
                "the new rate. Subscribe today to continue reading this "
                "article. Analysts expect the increase to slow signups.")
        kept = strip_boilerplate(text, claim)
        self.assertIn("standard subscription will cost", kept)
        self.assertIn("renew automatically", kept,
                      "a sentence sharing the claim's vocabulary was deleted")
        self.assertNotIn("continue reading", kept)

    def test_an_article_about_cookie_rules_keeps_its_content(self):
        claim = "The EU fined Meta over its cookie consent banners"
        text = ("Regulators said the company's cookie policy misled users. "
                "We use cookies to improve your experience. "
                "The fine is the largest issued under the privacy rules.")
        kept = strip_boilerplate(text, claim)
        self.assertIn("misled users", kept)
        self.assertIn("largest issued", kept)

    def test_without_a_claim_furniture_still_goes(self):
        self.assertEqual(strip_boilerplate(PAYWALL_STUB).strip(), "")


class TestPassageSelection(unittest.TestCase):

    def test_a_paywalled_page_sends_no_marketing_copy_to_nli(self):
        passages = extract_passages(
            "India PM resigns after coalition talks collapse", "",
            PAYWALL_STUB, claim=RESIGNATION_CLAIM,
        )
        for passage in passages:
            with self.subTest(passage=passage):
                self.assertFalse(is_boilerplate(passage, set()), passage)

    def test_the_headline_still_survives(self):
        passages = extract_passages(
            "India PM resigns after coalition talks collapse", "",
            PAYWALL_STUB, claim=RESIGNATION_CLAIM,
        )
        self.assertIn("India PM resigns after coalition talks collapse", passages)

    def test_real_sentences_are_kept_alongside_furniture(self):
        text = ("India's prime minister resigned on Tuesday. "
                "Subscribe today to continue reading this article. "
                "His deputy will serve in an acting capacity.")
        passages = extract_passages("", "", text, claim=RESIGNATION_CLAIM)
        joined = " ".join(passages)
        self.assertIn("resigned on Tuesday", joined)
        self.assertIn("acting capacity", joined)
        self.assertNotIn("continue reading", joined)


class TestFetchDoesNotDisplaceRealText(unittest.TestCase):
    """The comparison the pipeline makes when deciding which text to keep."""

    SNIPPET = ("India's prime minister resigned on Tuesday after coalition "
               "talks collapsed, his office confirmed.")

    def test_the_stub_is_longer_but_says_nothing(self):
        self.assertGreater(len(PAYWALL_STUB.split()), len(self.SNIPPET.split()),
                           "the premise of the bug: the stub wins on length")

    def test_it_no_longer_wins_on_content(self):
        stub = strip_boilerplate(PAYWALL_STUB, RESIGNATION_CLAIM)
        snippet = strip_boilerplate(self.SNIPPET, RESIGNATION_CLAIM)
        self.assertGreater(
            len(snippet.split()), len(stub.split()),
            "a paywall stub would still replace the provider's real sentence",
        )

    def test_a_genuine_article_body_still_wins(self):
        """The fetch must still replace a short snippet when it has the story."""
        body = ("India's prime minister resigned on Tuesday after coalition "
                "talks collapsed. His deputy will serve in an acting capacity "
                "until a parliamentary vote is held next month. The rupee fell "
                "on the news before recovering.")
        self.assertGreater(
            len(strip_boilerplate(body, RESIGNATION_CLAIM).split()),
            len(strip_boilerplate(self.SNIPPET, RESIGNATION_CLAIM).split()),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
