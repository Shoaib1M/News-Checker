"""Article text extraction and passage splitting.

Downloads an article URL and extracts readable paragraphs, stripping
navigation, ads, footers, and cookie text.  Provides passage-level
splitting for NLI scoring.
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
        self._tag_stack.append(tag)
        if tag == "p":
            self._paragraph_parts = []
        if tag in {"h2", "h3"}:
            self._heading_parts = []

    def handle_endtag(self, tag):
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
        if self._tag_stack:
            self._tag_stack.pop()

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


def _fetch_html(url: str, timeout: int = 10, retries: int = 2) -> str:
    last_error = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="ignore")
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last_error


def extract_article(url: str, timeout: int = 10) -> tuple[str, str]:
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


def extract_passages(
    title: str, snippet: str, full_text: str, max_passages: int = 8
) -> list[str]:
    """Return the most useful passages from an article for NLI scoring.

    The title and snippet are included since they are curated summaries.
    """
    passages: list[str] = []
    if title:
        passages.append(title)
    if snippet:
        passages.append(snippet)
    for sentence in split_sentences(full_text):
        passages.append(sentence)
    return passages[:max_passages]
