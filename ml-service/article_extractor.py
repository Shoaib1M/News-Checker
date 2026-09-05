"""Article text extraction and passage splitting.

Downloads an article URL and extracts readable paragraphs, stripping
navigation, ads, footers, and cookie text.  Provides passage-level
splitting for NLI scoring.

WHY PASSAGE SELECTION MATTERS:
NLI only ever sees the handful of passages this module returns. Anything it
drops cannot support or contradict anything, no matter how decisive it is.
That makes ``extract_passages`` the narrowest point in the whole pipeline —
and it used to take the article's *first* few sentences, which in news
writing are scene-setting. A story reporting "the prime minister resigned,
effective immediately" in its eighth sentence was handed to NLI as six
sentences about the weather and the security cordon, scored neutral, and
filed as coverage that does not address the claim.
"""

from __future__ import annotations

import re
import time
from html.parser import HTMLParser
from urllib.request import Request, urlopen

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


# Elements with no closing tag. Pushing them onto the stack desynchronises
# it: the matching </p> pops the void element instead, leaving "p" on the
# stack for the rest of the document.
_VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
})


class _ArticleParser(HTMLParser):
    """Extracts title and readable paragraphs from raw HTML."""

    def __init__(self):
        super().__init__()
        self.title = ""
        self.paragraphs: list[str] = []
        self._tag_stack: list[str] = []
        self._title_parts: list[str] = []
        self._paragraph_parts: list[str] = []
        self._heading_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _VOID_ELEMENTS:
            return
        self._tag_stack.append(tag)
        if tag == "p":
            self._paragraph_parts = []
        if tag in {"h2", "h3"}:
            self._heading_parts = []

    def handle_endtag(self, tag):
        if tag in _VOID_ELEMENTS:
            return
        if tag == "title":
            self.title = " ".join(self._title_parts).strip()
        if tag == "p":
            paragraph = " ".join(self._paragraph_parts).strip()
            if len(paragraph.split()) >= 8:
                self.paragraphs.append(paragraph)
            self._paragraph_parts = []
        if tag in {"h2", "h3"}:
            heading = " ".join(self._heading_parts).strip()
            if len(heading.split()) >= 3:
                self.paragraphs.append(heading)
            self._heading_parts = []
        # Unwind to the matching open tag. Real-world HTML routinely omits
        # closing tags; popping blindly leaves the stack permanently wrong.
        if tag in self._tag_stack:
            while self._tag_stack:
                if self._tag_stack.pop() == tag:
                    break

    def handle_data(self, data):
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._tag_stack and self._tag_stack[-1] == "title":
            self._title_parts.append(text)
        elif self._tag_stack and self._tag_stack[-1] in {"h2", "h3"}:
            self._heading_parts.append(text)
        elif "p" in self._tag_stack:
            self._paragraph_parts.append(text)


# Enough for any article; a cap is needed because search results sometimes
# point at very large pages, and the whole body is read into memory.
MAX_ARTICLE_BYTES = 2_000_000


def _fetch_html(url: str, timeout: int = 6, retries: int = 0) -> str:
    """Fetch an article page. No retry by default.

    This runs once per candidate document (up to 8 per claim). A news site
    that blocks or stalls a scraper on the first attempt almost never
    succeeds on a retry, so retrying just multiplies the wall-clock cost of
    a failure across every candidate.

    Non-HTML responses raise rather than being decoded: search results do
    sometimes point at a PDF or an image, and decoding those as text yields
    binary noise that would be handed to NLI as if it were prose.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                content_type = (response.headers.get_content_type() or "").lower()
                if content_type and not (
                    content_type.startswith("text/")
                    or content_type.endswith("+xml")
                    or content_type == "application/xhtml+xml"
                ):
                    raise ValueError(f"not an HTML document: {content_type}")
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read(MAX_ARTICLE_BYTES).decode(charset, errors="ignore")
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.0)
    raise last_error


def extract_article(url: str, timeout: int = 6) -> tuple[str, str]:
    """Download a URL and return (title, full_text)."""
    html = _fetch_html(url, timeout=timeout)
    parser = _ArticleParser()
    parser.feed(html)
    return parser.title, " ".join(parser.paragraphs)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling abbreviations like 'Mr.' and 'U.S.'."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    protected = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|est|govt|dept|U\.S|U\.K)\.",
        lambda m: m.group(0).replace(".", "<DOT>"),
        cleaned,
        flags=re.IGNORECASE,
    )
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(\[])', protected)
    return [
        part.replace("<DOT>", ".").strip()
        for part in parts
        if len(part.split()) >= 5
    ]


# Words too common to indicate that a sentence is about the claim.
_PASSAGE_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "from", "with", "as", "is", "are", "was", "were", "be",
    "been", "being", "has", "have", "had", "do", "does", "did", "will",
    "would", "can", "could", "may", "might", "must", "should", "that",
    "this", "these", "those", "it", "its", "they", "them", "he", "she",
    "his", "her", "their", "we", "you", "not", "no", "than", "then", "so",
    "said", "says", "after", "before", "over", "into", "about", "more",
    "also", "who", "which", "when", "where", "what", "how", "why",
})


def _content_words(text: str) -> set[str]:
    """Lowercased words that carry topic, for overlap scoring."""
    return {
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in _PASSAGE_STOPWORDS
    }


def extract_passages(
    title: str,
    snippet: str,
    full_text: str,
    max_passages: int = 8,
    claim: str = "",
) -> list[str]:
    """Return the passages most worth showing NLI for this claim.

    The title and snippet always come first — they are curated summaries of
    what the article is about. The remaining slots go to the article
    sentences with the most content-word overlap with ``claim``.

    WHY IT RANKS RATHER THAN TRUNCATES:
    This used to return the article's first ``max_passages`` sentences. News
    writing opens with scene-setting and states the specific fact several
    paragraphs down, so the decisive sentence routinely fell outside the
    window: a story reporting a resignation in its eighth sentence reached
    NLI as six sentences about the weather and the security cordon, was
    scored neutral, and was filed as coverage that does not address the
    claim. For a high-salience claim that is worse than unhelpful — enough
    such articles and the aggregator concludes nobody reported an event that
    was, in fact, reported.

    With no ``claim`` (or a claim sharing no vocabulary with the article)
    every sentence scores zero and the original document order is preserved,
    so behaviour degrades exactly to what it was before.
    """
    passages: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        key = re.sub(r"\W+", " ", (text or "").lower()).strip()
        # Google News descriptions frequently restate the headline verbatim;
        # duplicated passages waste NLI slots on the same sentence.
        if key and key not in seen:
            seen.add(key)
            passages.append(text)

    add(title)
    add(snippet)

    sentences = split_sentences(full_text)
    claim_words = _content_words(claim)
    if claim_words:
        # Sort by overlap, then by original position so ties keep article
        # order and the lede still wins when nothing matches.
        ranked = sorted(
            enumerate(sentences),
            key=lambda pair: (-len(claim_words & _content_words(pair[1])), pair[0]),
        )
        sentences = [sentence for _index, sentence in ranked]

    for sentence in sentences:
        if len(passages) >= max_passages:
            break
        add(sentence)

    return passages[:max_passages]
