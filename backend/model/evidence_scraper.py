from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, urlencode, unquote, urlparse
from urllib.request import Request, urlopen
import json
import os
import re
import sys
import time

import numpy as np

from tfidf import TFIDFVectorizer


SEARCH_URL = "https://duckduckgo.com/html/?q={query}"
NEWSAPI_URL = "https://newsapi.org/v2/everything"
GNEWS_URL = "https://gnews.io/api/v4/search"
GUARDIAN_URL = "https://content.guardianapis.com/search"
ENV_PATHS = [
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path(__file__).resolve().parents[2] / ".env",
]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "he", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "than", "this", "to", "was", "were",
    "will", "with", "any", "country", "countries", "world", "united", "states",
    "unit", "state",
}
CANONICAL_WORDS = {
    "defence": "military",
    "defense": "military",
    "military": "military",
    "pentagon": "military",
    "spend": "spending",
    "spends": "spending",
    "spent": "spending",
    "spending": "spending",
    "expenditure": "spending",
    "expenditures": "spending",
    "budget": "spending",
    "budgets": "spending",
    "nation": "country",
    "nations": "country",
    "countries": "country",
}
NEGATIONS = {
    "no", "not", "never", "none", "without", "cannot", "cant", "can't",
    "isnt", "isn't", "wasnt", "wasn't", "doesnt", "doesn't", "didnt",
    "didn't", "wont", "won't", "false", "fake",
}
OPPOSITE_GROUPS = [
    (
        {"more", "most", "higher", "highest", "greater", "larger", "largest", "above", "over", "exceeds", "lead", "leads", "leading", "top"},
        {"less", "least", "lower", "lowest", "smaller", "smallest", "below", "under", "fewer", "fewest", "behind", "last"},
    ),
    (
        {"increase", "increases", "increased", "increasing", "rise", "rises", "rose", "rising", "growth", "grew", "up"},
        {"decrease", "decreases", "decreased", "decreasing", "fall", "falls", "fell", "falling", "drop", "dropped", "down", "decline"},
    ),
    (
        {"support", "supports", "supported", "approve", "approves", "approved", "favor", "favors", "back", "backs"},
        {"oppose", "opposes", "opposed", "reject", "rejects", "rejected", "against", "block", "blocks"},
    ),
    (
        {"legal", "allowed", "allows", "permit", "permits", "permitted"},
        {"illegal", "banned", "ban", "bans", "prohibit", "prohibits", "prohibited"},
    ),
    (
        {"true", "accurate", "correct", "confirmed", "verified"},
        {"false", "fake", "incorrect", "debunked", "misleading"},
    ),
]
DIRECTION_WORDS = set().union(*[group for pair in OPPOSITE_GROUPS for group in pair])

# Domains that are generally considered reliable sources — gives a small ranking boost.
# Not a whitelist; unknown domains still rank normally.
CREDIBLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "economist.com", "ft.com", "bloomberg.com", "politifact.com",
    "factcheck.org", "snopes.com", "science.org", "nature.com",
    "pubmed.ncbi.nlm.nih.gov", "who.int", "cdc.gov", "gov.uk",
}


@dataclass
class EvidenceResult:
    url: str
    title: str
    snippet: str
    similarity: float
    text_length: int
    provider: str
    source: str
    support_score: float
    contradiction_score: float
    stance: str
    best_sentence: str


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._in_result_link = False
        self._in_snippet = False
        self._current_url = ""
        self._current_title = []
        self._current_snippet = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        class_name = attrs.get("class", "")

        if tag == "a" and "result__a" in class_name:
            self._in_result_link = True
            self._current_url = clean_duckduckgo_url(attrs.get("href", ""))
            self._current_title = []
            self._current_snippet = []

        if tag in {"a", "div"} and "result__snippet" in class_name:
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a" and self._in_result_link:
            self._in_result_link = False
            self._save_current_result()

        if tag in {"a", "div"} and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return

        if self._in_result_link:
            self._current_title.append(text)

        if self._in_snippet:
            self._current_snippet.append(text)

    def _save_current_result(self):
        title = " ".join(self._current_title).strip()
        snippet = " ".join(self._current_snippet).strip()

        if self._current_url and title:
            self.results.append({
                "url": self._current_url,
                "title": title,
                "snippet": snippet,
            })


class ArticleTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.paragraphs = []
        self._tag_stack = []
        self._title_parts = []
        self._paragraph_parts = []
        self._heading_parts = []

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

        # Headings are short but carry strong topical signal — include them
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


def load_env_file(paths=ENV_PATHS):
    for path in paths:
        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key and key not in os.environ:
                    os.environ[key] = value


def fetch_url(url, timeout=10, retries=2):
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


def fetch_json(url, timeout=10):
    return json.loads(fetch_url(url, timeout=timeout))


def clean_duckduckgo_url(url):
    if not url:
        return ""

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    if "uddg" in query:
        return unquote(query["uddg"][0])

    return url


def search_web(statement, max_results=15):
    query = quote_plus(build_search_query(statement))
    html = fetch_url(SEARCH_URL.format(query=query))

    parser = DuckDuckGoHTMLParser()
    parser.feed(html)

    seen = set()
    results = []
    for result in parser.results:
        url = result["url"]
        if url in seen:
            continue
        seen.add(url)
        results.append(result)
        if len(results) >= max_results:
            break

    return results


def search_web_raw(query_string, max_results=10):
    """Search DuckDuckGo using a pre-built query string (not derived from a statement)."""
    query = quote_plus(query_string)
    html = fetch_url(SEARCH_URL.format(query=query))

    parser = DuckDuckGoHTMLParser()
    parser.feed(html)

    seen = set()
    results = []
    for result in parser.results:
        url = result["url"]
        if url in seen:
            continue
        seen.add(url)
        results.append(result)
        if len(results) >= max_results:
            break

    return results


def dedupe_search_results(results):
    """Deduplicate a list of search result dicts by URL."""
    seen = set()
    unique = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(r)
    return unique


def search_newsapi(statement, max_results=10):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return []

    params = {
        "q": build_search_query(statement),
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_results,
        "apiKey": api_key,
    }
    payload = fetch_json(f"{NEWSAPI_URL}?{urlencode(params)}")

    documents = []
    for article in payload.get("articles", []):
        url = article.get("url", "")
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        content = article.get("content", "") or ""
        source = article.get("source", {}).get("name", "") or "NewsAPI"

        if not url or not title:
            continue

        documents.append({
            "provider": "newsapi",
            "source": source,
            "url": url,
            "title": title,
            "snippet": description,
            "text": content,
        })

    return documents


def search_gnews(statement, max_results=10):
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return []

    params = {
        "q": build_search_query(statement),
        "lang": "en",
        "max": max_results,
        "apikey": api_key,
    }
    payload = fetch_json(f"{GNEWS_URL}?{urlencode(params)}")

    documents = []
    for article in payload.get("articles", []):
        url = article.get("url", "")
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        content = article.get("content", "") or ""
        source = article.get("source", {}).get("name", "") or "GNews"

        if not url or not title:
            continue

        documents.append({
            "provider": "gnews",
            "source": source,
            "url": url,
            "title": title,
            "snippet": description,
            "text": content,
        })

    return documents


def search_guardian(statement, max_results=10):
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        return []

    params = {
        "q": build_search_query(statement),
        "api-key": api_key,
        "page-size": max_results,
        "show-fields": "headline,trailText,bodyText",
        "order-by": "relevance",
    }
    payload = fetch_json(f"{GUARDIAN_URL}?{urlencode(params)}")

    documents = []
    for article in payload.get("response", {}).get("results", []):
        fields = article.get("fields", {})
        url = article.get("webUrl", "")
        title = fields.get("headline") or article.get("webTitle", "") or ""
        snippet = fields.get("trailText", "") or ""
        text = fields.get("bodyText", "") or ""

        if not url or not title:
            continue

        documents.append({
            "provider": "guardian",
            "source": "The Guardian",
            "url": url,
            "title": title,
            "snippet": snippet,
            "text": text,
        })

    return documents


def search_api_providers(statement, max_results_per_provider=10):
    providers = [
        ("gnews", search_gnews),
        ("guardian", search_guardian),
        ("newsapi", search_newsapi),
    ]
    documents = []

    for provider_name, provider_func in providers:
        try:
            provider_documents = provider_func(statement, max_results=max_results_per_provider)
            documents.extend(provider_documents)
            if provider_documents:
                print(f"Fetched {len(provider_documents)} results from {provider_name}.")
        except Exception as error:
            print(f"{provider_name} failed: {error}")

    return dedupe_documents(documents)


def dedupe_documents(documents):
    seen_urls = set()
    seen_titles = set()
    unique_documents = []

    for document in documents:
        url = document.get("url", "")
        title = document.get("title", "").strip().lower()
        # Normalise title for comparison: collapse whitespace, strip punctuation
        title_key = re.sub(r"[^a-z0-9\s]", "", title)
        title_key = re.sub(r"\s+", " ", title_key).strip()

        if not url or url in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue

        seen_urls.add(url)
        if title_key:
            seen_titles.add(title_key)
        unique_documents.append(document)

    return unique_documents


def extract_article_text(url, timeout=10):
    html = fetch_url(url, timeout=timeout)
    parser = ArticleTextParser()
    parser.feed(html)
    text = " ".join(parser.paragraphs)
    return parser.title, text


def cosine_similarity(vector_a, vector_b):
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)


def normalize_word(word):
    word = word.lower()
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and len(word) > 4:
        word = word[:-1]

    return CANONICAL_WORDS.get(word, word)


def keywords(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    normalized_words = {normalize_word(word) for word in words}
    return {
        word for word in normalized_words
        if word not in STOPWORDS and len(word) > 2
    }


def mentions_united_states(statement):
    return bool(re.search(r"\b(united states|u\.s\.|us|usa|america|american)\b", statement.lower()))


def ordered_keywords(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    ordered = []
    seen = set()

    for word in words:
        normalized = normalize_word(word)
        if normalized in STOPWORDS or len(normalized) <= 2 or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)

    return ordered


def build_search_query(statement):
    important_words = ordered_keywords(statement)

    if not important_words:
        return statement

    # Pull named entities: capitalised words from the original text (names, places, orgs)
    named_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", statement)
    entity_words = []
    entity_keys = set()
    for entity in named_entities:
        entity_lower = entity.lower()
        if entity_lower not in STOPWORDS and len(entity_lower) > 2:
            entity_words.append(entity)
            # Track individual tokens so they are suppressed from remaining words
            for token in entity_lower.split():
                entity_keys.add(token)

    # Pull numbers and percentages — they are claim-specific and make queries precise
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", statement)

    # Direction/comparison words that characterise the claim
    direction_words_in_claim = [w for w in important_words if w in DIRECTION_WORDS]
    direction_keys = set(direction_words_in_claim)

    # Remaining content words — use the original (non-normalised) tokens for readability
    original_tokens = re.findall(r"[a-z][a-z0-9]+", statement.lower())
    seen_remaining = set()
    remaining = []
    for token in original_tokens:
        if token in STOPWORDS or len(token) <= 2:
            continue
        if token in entity_keys or token in direction_keys:
            continue
        if token not in seen_remaining:
            seen_remaining.add(token)
            remaining.append(token)

    # Build query: named entities first, then content keywords, then numbers.
    # Direction words (less/more/higher etc.) are intentionally excluded — including them
    # biases search results toward pages that echo the claim's framing instead of facts.
    query_parts = entity_words[:3] + remaining[:5] + numbers[:2]

    # Deduplicate while preserving order — case-insensitive so "Hillary" blocks "hillary"
    seen = set()
    deduped = []
    for part in query_parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(part)

    return " ".join(deduped) or statement


def build_opposite_search_query(statement):
    """Build a second query that flips direction words to actively pull contradicting evidence.

    E.g. 'US spends less on defense' -> search 'US defense spending highest most'
    Run alongside the normal query so we don't miss sources that contradict the claim.
    Returns None if the claim has no directional words.
    """
    FLIP_MAP = {}
    for pos_words, neg_words in OPPOSITE_GROUPS:
        for w in pos_words:
            FLIP_MAP[w] = next(iter(neg_words))
        for w in neg_words:
            FLIP_MAP[w] = next(iter(pos_words))

    important_words = ordered_keywords(statement)
    direction_words_in_claim = [w for w in important_words if w in DIRECTION_WORDS]

    if not direction_words_in_claim:
        return None  # no directional claim, nothing to flip

    named_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", statement)
    entity_words = []
    entity_keys = set()
    for entity in named_entities:
        entity_lower = entity.lower()
        if entity_lower not in STOPWORDS and len(entity_lower) > 2:
            entity_words.append(entity)
            for token in entity_lower.split():
                entity_keys.add(token)

    direction_keys = set(direction_words_in_claim)
    original_tokens = re.findall(r"[a-z][a-z0-9]+", statement.lower())
    seen_remaining = set()
    remaining = []
    for token in original_tokens:
        if token in STOPWORDS or len(token) <= 2:
            continue
        if token in entity_keys or token in direction_keys:
            continue
        if token not in seen_remaining:
            seen_remaining.add(token)
            remaining.append(token)

    flipped_direction = [FLIP_MAP.get(w, w) for w in direction_words_in_claim[:2]]
    query_parts = entity_words[:3] + flipped_direction + remaining[:4]

    seen = set()
    deduped = []
    for part in query_parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(part)

    return " ".join(deduped) or None


def keyword_coverage(statement, document_text):
    statement_keywords = keywords(statement)
    if not statement_keywords:
        return 0.0

    document_keywords = keywords(document_text)
    matches = statement_keywords.intersection(document_keywords)
    return len(matches) / len(statement_keywords)


def split_sentences(text):
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    # Protect common abbreviations so they don't cause false splits
    protected = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|est|govt|dept|U\.S|U\.K)\.",
        lambda m: m.group(0).replace(".", "<DOT>"),
        cleaned,
        flags=re.IGNORECASE,
    )
    # Split on sentence-ending punctuation followed by whitespace + capital letter
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(\[])', protected)
    sentences = []
    for part in parts:
        part = part.replace("<DOT>", ".").strip()
        if len(part.split()) >= 5:
            sentences.append(part)
    return sentences


def has_any(words, text):
    text_words = keywords(text)
    return bool(text_words.intersection(words))


def opposite_direction_score(claim, sentence):
    claim_words = keywords(claim)
    sentence_words = keywords(sentence)
    score = 0

    for positive_words, negative_words in OPPOSITE_GROUPS:
        claim_positive = bool(claim_words.intersection(positive_words))
        claim_negative = bool(claim_words.intersection(negative_words))
        sentence_positive = bool(sentence_words.intersection(positive_words))
        sentence_negative = bool(sentence_words.intersection(negative_words))

        if (claim_positive and sentence_negative) or (claim_negative and sentence_positive):
            score += 1

    return min(score / 2, 1.0)


def same_direction_score(claim, sentence):
    claim_words = keywords(claim)
    sentence_words = keywords(sentence)

    # Require at least one shared non-direction content keyword before direction agreement counts.
    # Without this, 'US spends less on defense' vs 'defense budget was less than last decade'
    # both have 'less' so they score as agreeing — but they're about different things.
    content_overlap = (claim_words - DIRECTION_WORDS).intersection(sentence_words - DIRECTION_WORDS)
    if not content_overlap:
        return 0.0

    score = 0
    for positive_words, negative_words in OPPOSITE_GROUPS:
        if claim_words.intersection(positive_words) and sentence_words.intersection(positive_words):
            score += 1
        if claim_words.intersection(negative_words) and sentence_words.intersection(negative_words):
            score += 1

    return min(score / 2, 1.0)


def has_directional_claim(claim):
    return bool(keywords(claim).intersection(DIRECTION_WORDS))


def negation_flip_score(claim, sentence):
    claim_has_negation = has_any(NEGATIONS, claim)
    sentence_has_negation = has_any(NEGATIONS, sentence)
    return 1.0 if claim_has_negation != sentence_has_negation else 0.0


def extract_numbers(text):
    numbers = []
    for match in re.findall(r"\$?\b\d+(?:,\d{3})*(?:\.\d+)?%?", text):
        cleaned = match.replace("$", "").replace(",", "").replace("%", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            pass
    return numbers


def numeric_alignment_score(claim, sentence):
    claim_numbers = extract_numbers(claim)
    sentence_numbers = extract_numbers(sentence)

    if not claim_numbers:
        return 0.5, 0.0

    if not sentence_numbers:
        return 0.0, 0.0

    aligned = 0
    contradicted = 0

    for claim_number in claim_numbers:
        closest_distance = min(abs(claim_number - sentence_number) for sentence_number in sentence_numbers)
        tolerance = max(abs(claim_number) * 0.10, 1.0)

        if closest_distance <= tolerance:
            aligned += 1
        else:
            contradicted += 1

    return aligned / len(claim_numbers), contradicted / len(claim_numbers)


def sentence_relevance(claim, sentence):
    coverage = keyword_coverage(claim, sentence)
    claim_words = keywords(claim)
    sentence_words = keywords(sentence)

    if not claim_words or not sentence_words:
        return 0.0

    jaccard = len(claim_words.intersection(sentence_words)) / len(claim_words.union(sentence_words))
    return (0.75 * coverage) + (0.25 * jaccard)


def score_sentence_stance(claim, sentence):
    relevance = sentence_relevance(claim, sentence)
    if relevance < 0.15:
        return 0.0, 0.0, relevance

    same_direction = same_direction_score(claim, sentence)
    opposite_direction = opposite_direction_score(claim, sentence)
    negation_flip = negation_flip_score(claim, sentence)
    number_support, number_conflict = numeric_alignment_score(claim, sentence)
    support_base = 0.20 if has_directional_claim(claim) else 0.45

    support = relevance * (
        support_base +
        0.35 * same_direction +
        0.20 * number_support
    ) * (
        1 -
        0.90 * opposite_direction -
        0.55 * negation_flip -
        0.45 * number_conflict
    )
    contradiction = relevance * (
        0.08 +
        0.90 * opposite_direction +
        0.50 * negation_flip +
        0.40 * number_conflict
    ) * (
        1 -
        0.25 * same_direction -
        0.15 * number_support
    )

    return max(0.0, min(support, 1.0)), max(0.0, min(contradiction, 1.0)), relevance


def score_document_stance(statement, document):
    title = document.get("title", "")
    snippet = document.get("snippet", "")
    text = document.get("text", "")
    sentences = []

    # Title and snippet get scored with a weight boost since they are curated summaries
    if title:
        sentences.append((title, 1.5))
    if snippet:
        sentences.append((snippet, 1.2))
    for sentence in split_sentences(text):
        sentences.append((sentence, 1.0))

    if not sentences:
        return 0.0, 0.0, ""

    scored_sentences = []
    for sentence, weight in sentences:
        support, contradiction, relevance = score_sentence_stance(statement, sentence)
        scored_sentences.append((support, contradiction, relevance, sentence, weight))

    # Sort by the strongest signal per sentence
    scored_sentences.sort(
        key=lambda item: max(item[0], item[1], item[2]) * item[4],
        reverse=True,
    )
    top_sentences = scored_sentences[:5]

    # Weighted average instead of raw max — reduces noise from one outlier sentence
    total_weight = sum(item[4] for item in top_sentences)
    support = sum(item[0] * item[4] for item in top_sentences) / total_weight
    contradiction = sum(item[1] * item[4] for item in top_sentences) / total_weight
    best_sentence = max(top_sentences, key=lambda item: max(item[0], item[1]))[3]

    return support, contradiction, best_sentence


def stance_label(support, contradiction):
    margin = support - contradiction
    if max(support, contradiction) < 0.10:
        return "unclear"
    if margin >= 0.06:
        return "supports"
    if margin <= -0.06:
        return "contradicts"
    return "mixed"


def source_credibility_boost(url):
    """Returns a small multiplier boost for known credible domains."""
    try:
        netloc = urlparse(url).netloc.lower().lstrip("www.")
        for domain in CREDIBLE_DOMAINS:
            if netloc == domain or netloc.endswith("." + domain):
                return 1.15
    except Exception:
        pass
    return 1.0


def rank_by_similarity(statement, documents):
    texts = [statement]
    for document in documents:
        article_preview = " ".join(document.get("text", "").split()[:500])
        combined_text = " ".join([
            document.get("title", ""),
            document.get("snippet", ""),
            article_preview,
        ])
        texts.append(combined_text)

    vectorizer = TFIDFVectorizer(ngram_range=(1, 2), min_df=1)
    vectorizer.build_vocab(texts)
    vectors = vectorizer.transform(texts)

    statement_vector = vectors[0]
    ranked = []

    for index, document in enumerate(documents, start=1):
        article_preview = " ".join(document.get("text", "").split()[:500])
        combined_text = " ".join([
            document.get("title", ""),
            document.get("snippet", ""),
            article_preview,
        ])
        tfidf_score = cosine_similarity(statement_vector, vectors[index])
        coverage_score = keyword_coverage(statement, combined_text)

        # Title is a strong signal — score it separately and blend in
        title_coverage = keyword_coverage(statement, document.get("title", ""))
        similarity = (0.35 * tfidf_score) + (0.45 * coverage_score) + (0.20 * title_coverage)

        # Small boost for credible sources — keeps ranking mostly content-driven
        similarity = min(similarity * source_credibility_boost(document.get("url", "")), 1.0)

        support_score, contradiction_score, best_sentence = score_document_stance(statement, document)
        ranked.append(EvidenceResult(
            url=document["url"],
            title=document.get("title", ""),
            snippet=document.get("snippet", ""),
            similarity=similarity,
            text_length=len(document.get("text", "")),
            provider=document.get("provider", "unknown"),
            source=document.get("source", ""),
            support_score=support_score,
            contradiction_score=contradiction_score,
            stance=stance_label(support_score, contradiction_score),
            best_sentence=best_sentence,
        ))

    return sorted(ranked, key=lambda result: result.similarity, reverse=True)


def evidence_score(results, top_k=5):
    if not results:
        return 0.0

    top_results = results[:top_k]
    weights = np.array([1 / (index + 1) for index in range(len(top_results))])
    similarities = np.array([result.similarity for result in top_results])

    return float(np.sum(similarities * weights) / np.sum(weights))


def stance_summary(results, top_k=5):
    relevant_results = [result for result in results if result.similarity >= 0.25]

    if not relevant_results:
        return {
            "support": 0.0,
            "contradiction": 0.0,
            "net": 0.0,
            "verdict": "not enough relevant evidence",
        }

    top_results = relevant_results[:top_k]
    weights = np.array([
        result.similarity / (index + 1)
        for index, result in enumerate(top_results)
    ])

    if np.sum(weights) == 0:
        weights = np.ones(len(top_results))

    support_scores = np.array([result.support_score for result in top_results])
    contradiction_scores = np.array([result.contradiction_score for result in top_results])

    support = float(np.sum(support_scores * weights) / np.sum(weights))
    contradiction = float(np.sum(contradiction_scores * weights) / np.sum(weights))
    net = support - contradiction

    if max(support, contradiction) < 0.10:
        verdict = "not enough stance evidence"
    elif net >= 0.06:
        verdict = "evidence mostly supports the claim"
    elif net <= -0.06:
        verdict = "evidence mostly contradicts the claim"
    else:
        verdict = "evidence is mixed"

    return {
        "support": support,
        "contradiction": contradiction,
        "net": net,
        "verdict": verdict,
    }


def collect_duckduckgo_evidence(statement, max_results=15, fetch_articles=True):
    # Primary search — topic-focused, direction words stripped so results aren't biased
    search_results = search_web(statement, max_results=max_results)

    # Opposite-direction search — actively hunts for sources that contradict the claim.
    # E.g. if the claim says 'US spends less', we also search 'US highest defense spending'
    # so we don't only collect evidence that happens to echo the claim's framing.
    opposite_query = build_opposite_search_query(statement)
    if opposite_query:
        print(f"Running opposite-direction search: {opposite_query}")
        try:
            opposite_results = search_web_raw(opposite_query, max_results=max_results // 2)
            search_results = dedupe_search_results(search_results + opposite_results)
        except Exception as error:
            print(f"Opposite search failed: {error}")

    documents = []
    for result in search_results:
        article_title = ""
        article_text = ""

        if fetch_articles:
            try:
                article_title, article_text = extract_article_text(result["url"])
            except Exception as error:
                article_text = ""
                result["snippet"] = f"{result.get('snippet', '')} [fetch failed: {error}]"

        documents.append({
            "provider": "duckduckgo",
            "source": urlparse(result["url"]).netloc,
            "url": result["url"],
            "title": article_title or result["title"],
            "snippet": result.get("snippet", ""),
            "text": article_text,
        })

    return documents


def enrich_documents_with_article_text(documents, fetch_articles=True):
    if not fetch_articles:
        return documents

    enriched_documents = []
    for document in documents:
        if document.get("text"):
            enriched_documents.append(document)
            continue

        try:
            article_title, article_text = extract_article_text(document["url"])
            document["title"] = article_title or document.get("title", "")
            document["text"] = article_text
        except Exception as error:
            document["snippet"] = f"{document.get('snippet', '')} [fetch failed: {error}]"

        enriched_documents.append(document)

    return enriched_documents


def collect_evidence(statement, max_results=15, fetch_articles=True, use_fallback=True):
    load_env_file()
    documents = search_api_providers(statement, max_results_per_provider=max_results)

    if use_fallback:
        if documents:
            print("Adding DuckDuckGo fallback results for broader evidence coverage.")
        else:
            print("No API results found. Falling back to DuckDuckGo HTML search.")

        fallback_documents = collect_duckduckgo_evidence(
            statement,
            max_results=max_results,
            fetch_articles=fetch_articles,
        )
        documents = dedupe_documents(documents + fallback_documents)

    if documents:
        documents = enrich_documents_with_article_text(documents, fetch_articles=fetch_articles)
    else:
        documents = []

    ranked_results = rank_by_similarity(statement, documents)
    return evidence_score(ranked_results), stance_summary(ranked_results), ranked_results


def main():
    load_env_file()

    if len(sys.argv) > 1:
        statement = " ".join(sys.argv[1:])
    else:
        statement = input("Statement: ").strip()

    if not statement:
        print("Please enter a statement.")
        return

    configured_providers = [
        name for name, env_name in [
            ("GNews", "GNEWS_API_KEY"),
            ("Guardian", "GUARDIAN_API_KEY"),
            ("NewsAPI", "NEWSAPI_KEY"),
        ]
        if os.getenv(env_name)
    ]

    if configured_providers:
        print(f"Configured API providers: {', '.join(configured_providers)}")
    else:
        print("No API keys found. The script will use DuckDuckGo fallback.")

    print(f"Search query: {build_search_query(statement)}")
    opposite = build_opposite_search_query(statement)
    if opposite:
        print(f"Opposite query: {opposite}")
    print("Searching and ranking evidence...")
    score, summary, results = collect_evidence(statement, max_results=15, fetch_articles=True)

    print(f"\nEvidence similarity score: {score:.3f}")
    print(f"Support score: {summary['support']:.3f}")
    print(f"Contradiction score: {summary['contradiction']:.3f}")
    print(f"Net stance score: {summary['net']:.3f}")
    print(f"Verdict: {summary['verdict']}")
    print("\nTop evidence URLs:")
    for index, result in enumerate(results[:10], start=1):
        print(f"\n{index}. {result.title}")
        print(f"   Similarity: {result.similarity:.3f}")
        print(f"   Stance: {result.stance}")
        print(f"   Support: {result.support_score:.3f}")
        print(f"   Contradiction: {result.contradiction_score:.3f}")
        print(f"   Provider: {result.provider}")
        if result.source:
            print(f"   Source: {result.source}")
        print(f"   Text length: {result.text_length}")
        print(f"   URL: {result.url}")
        if result.best_sentence:
            print(f"   Best evidence: {result.best_sentence[:300]}")
        if result.snippet:
            print(f"   Snippet: {result.snippet[:250]}")


if __name__ == "__main__":
    main()