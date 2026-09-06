"""
Relevance Filtering: Strict claim-aware filtering to eliminate irrelevant search results.

This module ensures that only genuinely relevant articles are presented as evidence.
It prevents keyword collisions (e.g., "2050" or "changed" matching irrelevant articles).
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Set
from claim_decomposer import decompose_claim, ClaimDecomposition


# The event vocabulary lives in event_vocabulary.py because query generation
# needs the same notion of "what action does this claim assert" — the two
# stages have to agree, or the retriever searches for one thing and the filter
# scores for another.
from event_vocabulary import ANTONYMS as _ACTION_ANTONYMS  # noqa: E402
from event_vocabulary import EVENT_VERBS as _EVENT_VERBS  # noqa: E402
from event_vocabulary import events_in as _events_in  # noqa: E402


@dataclass
class RelevanceScore:
    """Detailed relevance assessment for a document."""
    entity_match_score: float  # 0.0-1.0: How many key entities are mentioned
    predicate_match_score: float  # 0.0-1.0: How relevant predicates are
    semantic_coherence: float  # 0.0-1.0: How well entities and predicates connect
    keyword_specificity: float  # 0.0-1.0: Avoids generic keyword overlap
    action_match_score: float  # 0.0-1.0: Does the doc discuss the claim's action?
    action_required: bool  # Did the claim assert a recognizable event at all?
    overall_relevance: float  # 0.0-1.0: Final relevance score
    reasons_included: List[str]  # Why this article might be relevant
    reasons_excluded: List[str]  # Why this article should be filtered


class RelevanceFilter:
    """
    Strict, claim-aware filtering for evidence documents.
    
    This prevents irrelevant articles from being presented as evidence while
    preserving genuinely relevant sources.
    """
    
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'have', 'he', 'in', 'is', 'it', 'of', 'on', 'or', 'that',
        'the', 'to', 'was', 'were', 'will', 'with', 'this', 'these',
        'those', 'it', 'its', 'but', 'if', 'not', 'no', 'or', 'which',
        'how', 'what', 'when', 'where', 'why', 'than', 'then', 'so',
        'can', 'could', 'would', 'should', 'may', 'might', 'must',
        'him', 'her', 'them', 'me', 'you', 'us', 'my', 'your', 'their',
        'changed', 'change', 'changing', 'name', 'said', 'said', 'say',  # Common generic words
    }
    
    # Calibrated thresholds: reject collisions, but retain genuinely topical
    # reporting for claims whose wording differs from the article wording.
    #
    # MEASURED, not guessed — see tests/test_relevance_corpus.py, which scores
    # these against twenty hand-labelled (claim, document) pairs:
    #
    #     0.30-0.48   precision 0.91   recall 1.00   F1 0.95
    #     0.50+       precision 1.00   recall 0.80   F1 0.89
    #
    # STRICT_MIN_RELEVANCE sits mid-plateau. Do not raise it to chase that
    # last point of precision: the costs are not symmetric. A document
    # rejected here is gone, and for a high-salience claim enough wrong
    # rejections become "no credible source reports this" — a statement about
    # the world. A document let through still has to be NLI-classified before
    # it counts, and if it says nothing it is shown as "Related coverage"
    # rather than counted as evidence.
    MIN_ENTITY_MATCH = 0.20  # At least 20% of entities must match
    MIN_OVERALL_RELEVANCE = 0.30
    STRICT_MIN_RELEVANCE = 0.42
    
    def __init__(self):
        """Initialize the relevance filter."""
        pass
    
    def assess_document_relevance(
        self,
        claim: str,
        document_title: str,
        document_snippet: str,
        document_text: str,
        decomp: ClaimDecomposition = None,
    ) -> RelevanceScore:
        """
        Assess how relevant a document is to the claim.
        
        Args:
            claim: The original claim text
            document_title: Article title
            document_snippet: Article snippet/summary
            document_text: Full article text (first 2000 chars)
            decomp: Pre-computed claim decomposition
            
        Returns:
            RelevanceScore with detailed assessment
        """
        if decomp is None:
            decomp = decompose_claim(claim)
        
        # Combine document parts for scoring
        doc_combined = f"{document_title} {document_snippet} {document_text}"
        
        # Extract entities and keywords from document
        doc_entities = self._extract_entities(document_title, document_text)
        doc_keywords = self._extract_keywords(doc_combined)
        
        # Assess each dimension
        entity_match = self._score_entity_match(decomp, doc_combined)
        predicate_match = self._score_predicate_match(decomp, doc_keywords)
        semantic_coherence = self._score_semantic_coherence(decomp, doc_entities, doc_keywords)
        keyword_specificity = self._score_keyword_specificity(claim, decomp, doc_combined)
        claim_events = self._claim_events(claim)
        action_match = self._score_action_match(claim_events, doc_combined)

        # Determine overall relevance. Entity match still leads, but it no
        # longer dominates: two entities both appearing is what let articles
        # about the right subjects and the wrong event through.
        weights = {
            'entity_match': 0.32,
            'action_match': 0.23,
            'predicate_match': 0.15,
            'semantic_coherence': 0.15,
            'keyword_specificity': 0.15,
        }

        overall = (
            entity_match * weights['entity_match'] +
            action_match * weights['action_match'] +
            predicate_match * weights['predicate_match'] +
            semantic_coherence * weights['semantic_coherence'] +
            keyword_specificity * weights['keyword_specificity']
        )
        
        # Determine inclusion/exclusion reasons
        reasons_included = []
        reasons_excluded = []
        
        if entity_match > 0.5:
            reasons_included.append("Key entities mentioned")
        if predicate_match > 0.4:
            reasons_included.append("Relevant to claim subject matter")
        if semantic_coherence > 0.5:
            reasons_included.append("Entities connect to claim topic")
        
        if claim_events and action_match >= 1.0:
            reasons_included.append("Discusses the action the claim asserts")

        if entity_match < self.MIN_ENTITY_MATCH:
            reasons_excluded.append("Very few claim entities mentioned")
        if claim_events and action_match == 0.0:
            reasons_excluded.append(
                "Mentions the claim's subjects but not the event it asserts"
            )
        if overall < self.MIN_OVERALL_RELEVANCE:
            reasons_excluded.append("Insufficient relevance to claim")
        if keyword_specificity < 0.20:
            reasons_excluded.append("Generic keyword overlap only")
        
        return RelevanceScore(
            entity_match_score=entity_match,
            predicate_match_score=predicate_match,
            semantic_coherence=semantic_coherence,
            keyword_specificity=keyword_specificity,
            action_match_score=action_match,
            action_required=bool(claim_events),
            overall_relevance=overall,
            reasons_included=reasons_included,
            reasons_excluded=reasons_excluded,
        )
    
    def _extract_entities(self, title: str, text: str) -> Set[str]:
        """Extract capitalized named entities (people, places, organizations)."""
        entities = set()
        
        # Look in title first (more prominent)
        title_text = f"{title} {text[:1200]}"  # Check title + opening of text
        
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        for match in re.finditer(pattern, title_text):
            entity = match.group(1)
            # Skip common non-entity capitalized words
            if entity not in {'The', 'A', 'An', 'In', 'On', 'At', 'Is', 'Was', 'Are', 'Were'}:
                entities.add(entity.lower())

        lowered = title_text.lower()
        # Multi-word names the capitalized-phrase regex above can miss when a
        # publisher lowercases its headlines. These are unambiguous, so a
        # case-insensitive match is safe.
        for phrase in ("united states", "united kingdom", "great wall",
                       "china", "india", "nasa", "earth"):
            if re.search(rf"\b{re.escape(phrase)}\b", lowered):
                entities.add(phrase)

        # Country abbreviations are matched CASE-SENSITIVELY, against the
        # original text. "us" lowercased is the pronoun, and it appears in
        # most English prose: a personal blog post headlined "Google blocked
        # our account and never told us why" scored a perfect entity match
        # for a claim about the United States banning Google, and was
        # admitted as evidence with the reason "Key entities mentioned".
        for pattern, canonical in (
            (r"\b(?:U\.?S\.?A?\.?|USA)\b", "united states"),
            (r"\b(?:U\.?K\.?)\b", "united kingdom"),
        ):
            if re.search(pattern, title_text):
                entities.add(canonical)

        return entities
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        words = re.findall(r'\b([a-z]+(?:[a-z\-]*[a-z])?)\b', text.lower())
        return {w for w in words if len(w) > 2 and w not in self.STOPWORDS}
    
    # Alternative surface forms for entities that are commonly abbreviated.
    # Each is (pattern, case_sensitive). Country abbreviations are matched
    # case-sensitively against the original text: lowercase "us" is the
    # pronoun and appears in most English prose, so treating it as the
    # country gave a personal blog post headlined "Google blocked our account
    # and never told us why" a perfect entity match for a claim about the
    # United States, admitted as evidence under "Key entities mentioned".
    # Countries whose adjectival form is not the name plus a suffix. Headlines
    # use these constantly — "Indian PM", "Chinese regulators", "British
    # officials" — and a plain word-boundary match on the country name rejects
    # every one of them. "Indian PM steps down amid political turmoil" scored
    # ZERO entity match for a claim about India's prime minister resigning,
    # and was filtered out despite being exactly the right article.
    _DEMONYMS: dict[str, tuple[str, ...]] = {
        "china": ("chinese",),
        "britain": ("british", "briton", "britons"),
        "france": ("french",),
        "japan": ("japanese",),
        "spain": ("spanish", "spaniard"),
        "germany": ("german", "germans"),
        "poland": ("polish", "pole", "poles"),
        "turkey": ("turkish", "turk", "turks"),
        "israel": ("israeli", "israelis"),
        "netherlands": ("dutch",),
        "sweden": ("swedish", "swede", "swedes"),
        "denmark": ("danish", "dane", "danes"),
        "scotland": ("scottish", "scots"),
        "ireland": ("irish",),
        "wales": ("welsh",),
        "greece": ("greek", "greeks"),
        "norway": ("norwegian", "norwegians"),
        "finland": ("finnish", "finn", "finns"),
        "switzerland": ("swiss",),
        "portugal": ("portuguese",),
        "thailand": ("thai",),
        "vietnam": ("vietnamese",),
        "philippines": ("filipino", "filipinos", "philippine"),
    }

    # Regular adjectival and plural endings, which cover the rest:
    # India/Indian, America/American, Russia/Russian, Ukraine/Ukrainian.
    # Applied only to entities long enough that a suffix cannot turn one
    # short name into an unrelated word.
    _REGULAR_SUFFIXES = r"(?:n|ns|an|ans|ian|ians|s)?"
    _MIN_LENGTH_FOR_SUFFIX = 4

    _ENTITY_VARIANTS: dict[str, list[tuple[str, bool]]] = {
        "united states": [
            (r"\bunited states\b", False),
            # The noun AND its demonym: "America", "Americas", "American",
            # "Americans". Writing only the demonym form drops the noun.
            (r"\bamerica(?:ns?|s)?\b", False),
            (r"\b(?:U\.?S\.?A?\.?|USA)\b", True),
        ],
        "united kingdom": [
            (r"\bunited kingdom\b", False),
            (r"\bbritain\b", False),
            (r"\bbritish\b", False),
            (r"\b(?:U\.?K\.?)\b", True),
        ],
        "the great wall": [(r"\bgreat wall\b", False)],
    }

    def _default_probes(self, entity: str) -> List[tuple[str, bool]]:
        """Patterns that count as a mention of ``entity``.

        The name itself, its irregular demonyms, and — for names long enough
        to be unambiguous — a regular adjectival or plural ending.
        """
        probes: List[tuple[str, bool]] = []
        if len(entity) >= self._MIN_LENGTH_FOR_SUFFIX and " " not in entity:
            probes.append((rf"\b{re.escape(entity)}{self._REGULAR_SUFFIXES}\b", False))
        else:
            probes.append((rf"\b{re.escape(entity)}\b", False))
        for demonym in self._DEMONYMS.get(entity, ()):
            probes.append((rf"\b{re.escape(demonym)}\b", False))
        return probes

    def _score_entity_match(self, decomp: ClaimDecomposition, document_text: str) -> float:
        """Score how many of the claim's entities actually appear in the document.

        Searches the document text for each claim entity directly, rather
        than extracting the document's own entities and intersecting the two
        sets. That older design failed in both directions, because the two
        extractors did not know the same things: the claim side recognises
        lowercase organisation names ("google"), the document side did not,
        so an all-lowercase headline about Google never matched a Google
        claim — while the document side recognised bare "us", which the claim
        side meant as a country and the document meant as a pronoun.

        Returns 0.0-1.0 where 1.0 means every key entity appears.
        """
        if not decomp.primary_entities:
            return 0.5  # Neutral for non-entity claims

        claim_entities = {e.lower() for e in decomp.primary_entities}
        matched = set()
        for entity in claim_entities:
            probes = self._ENTITY_VARIANTS.get(entity) or self._default_probes(entity)
            for pattern, case_sensitive in probes:
                flags = 0 if case_sensitive else re.IGNORECASE
                if re.search(pattern, document_text, flags):
                    matched.add(entity)
                    break

        return len(matched) / len(claim_entities)
    
    def _entity_tokens(self, decomp: ClaimDecomposition) -> Set[str]:
        """Lowercased word tokens of the claim's own entity, for single-entity claims only.

        For a single-entity claim (e.g. "US government considers banning
        Google"), entity_match_score is nearly binary — any document
        mentioning "Google" already scores 1.0 there, so letting that same
        mention also inflate predicate/coherence scoring would make the
        entity's mere presence carry almost the entire relevance decision,
        regardless of what the document actually says about it.

        Multi-entity claims (e.g. "United States" + "India") are left alone:
        there, matching multiple distinct entities together is itself
        meaningful relational evidence and should keep contributing to
        predicate/coherence overlap.
        """
        if len(decomp.primary_entities) != 1:
            return set()
        tokens: Set[str] = set()
        for entity in decomp.primary_entities:
            tokens.update(entity.lower().split())
        return tokens

    def _score_predicate_match(self, decomp: ClaimDecomposition, doc_keywords: Set[str]) -> float:
        """
        Score how relevant the document is to the claim's predicates.

        Returns 0.0-1.0
        """
        claim_keywords = {
            token
            for predicate in decomp.core_predicates
            for token in predicate.lower().split()
            if token not in {'being', 'is', 'are', 'was', 'were', 'to'}
        }
        claim_keywords.update(decomp.entities_in_claim)
        claim_keywords.difference_update({
            'will', 'next', 'last', 'today', 'tomorrow', 'yesterday',
            'month', 'year', 'day', 'time', 'being',
        })
        claim_keywords.difference_update(self._entity_tokens(decomp))

        if not claim_keywords:
            return 0.5
        
        # Check for keyword overlap
        overlap = claim_keywords.intersection(doc_keywords)
        base_score = len(overlap) / len(claim_keywords)
        
        return min(base_score, 1.0)
    
    def _score_semantic_coherence(
        self,
        decomp: ClaimDecomposition,
        doc_entities: Set[str],
        doc_keywords: Set[str]
    ) -> float:
        """
        Score how well entities and predicates connect in the document.
        
        A good document discusses the same entities in the same context.
        """
        # Check if entities appear with relevant keywords
        claim_keywords = {
            token
            for predicate in decomp.core_predicates
            for token in predicate.lower().split()
            if token not in {'being', 'is', 'are', 'was', 'were', 'to'}
        }
        claim_keywords.update(decomp.entities_in_claim)
        claim_keywords.difference_update(self._entity_tokens(decomp))

        # If both entities and relevant keywords appear, coherence is high
        overlap = len(claim_keywords.intersection(doc_keywords))
        if len(doc_entities) > 0 and overlap > 1:
            return 0.7
        elif overlap > 1:
            return 0.55
        elif len(doc_entities) > 0:
            return 0.4  # Has entities but missing keywords
        else:
            return 0.2  # Missing entities
    
    def _claim_events(self, claim: str) -> Set[str]:
        """Canonical events the claim asserts, from the curated vocabulary.

        Returns an empty set when the claim uses no recognized event verb —
        in which case action scoring stays neutral and this dimension has no
        effect on whether the document is kept.
        """
        return _events_in(claim)

    def _score_action_match(self, claim_events: Set[str], doc_combined: str) -> float:
        """1.0 if the document discusses the claim's event — or its opposite.

        Neutral (0.5) when the claim asserts no recognized event, so claims
        outside the vocabulary are scored exactly as they were before.
        """
        if not claim_events:
            return 0.5
        lowered = f" {doc_combined.lower()} "
        # The claim's own action, plus its opposite: a source that contradicts
        # the claim is about the same event, stated the other way round.
        wanted = set(claim_events)
        wanted.update(
            _ACTION_ANTONYMS[event] for event in claim_events
            if event in _ACTION_ANTONYMS
        )
        for event in wanted:
            for surface in _EVENT_VERBS[event]:
                if re.search(rf"(?<![a-z]){re.escape(surface)}(?![a-z])", lowered):
                    return 1.0
        return 0.0

    def _score_keyword_specificity(
        self,
        claim: str,
        decomp: ClaimDecomposition,
        doc_combined: str,
    ) -> float:
        """
        Prevent irrelevant articles by penalizing generic keyword overlap.
        
        Example: "2050" appears in many articles. "2050" alone should not
        make an unrelated article relevant.
        
        Returns 0.0-1.0 where 1.0 means high specificity (not generic overlap).
        """
        doc_lower = doc_combined.lower()
        
        # Count appearances of generic numerals/temporal markers that might cause collisions
        generic_tokens = []
        if decomp.numerical_values:
            generic_tokens.extend(decomp.numerical_values)
        if decomp.temporal_modifiers:
            generic_tokens.extend([t.lower() for t in decomp.temporal_modifiers])
        
        generic_appearances = sum(1 for token in generic_tokens if token.lower() in doc_lower)
        generic_penalty = generic_appearances * 0.15  # Each generic match reduces specificity
        
        # High specificity: document discusses entities + predicates + context
        claim_keywords = self._extract_keywords(claim)
        doc_keywords = self._extract_keywords(doc_combined)
        
        # Specificity = keyword overlap with substantive words (not just numerals)
        substantial_overlap = len(claim_keywords.intersection(doc_keywords))
        if claim_keywords:
            specificity = substantial_overlap / len(claim_keywords)
        else:
            specificity = 0.5
        
        # Apply penalty for generic token collisions
        specificity = max(0.0, specificity - generic_penalty)
        
        return min(specificity, 1.0)
    
    def should_include_document(self, relevance_score: RelevanceScore, strict: bool = False) -> bool:
        """
        Determine if a document should be included in evidence.
        
        Args:
            relevance_score: The RelevanceScore for the document
            strict: If True, use stricter thresholds for final evidence
            
        Returns:
            True if document should be included, False otherwise
        """
        threshold = self.STRICT_MIN_RELEVANCE if strict else self.MIN_OVERALL_RELEVANCE
        if relevance_score.overall_relevance < threshold:
            return False
        # A document about the claim's subjects that never mentions the event
        # the claim asserts is background, not evidence. Rejecting it is only
        # safe because "we found coverage of these subjects and none of it
        # reports this event" is now itself a reportable outcome
        # (evidence_aggregator.assess_coverage) rather than a dead end.
        if strict and relevance_score.action_required and relevance_score.action_match_score == 0.0:
            return False
        if relevance_score.entity_match_score == 0 and relevance_score.keyword_specificity < 0.35:
            return False
        if relevance_score.entity_match_score < 0.5 and relevance_score.predicate_match_score < 0.5:
            return False
        # A document can discuss exactly the event the claim is about while
        # sharing almost none of the claim's wording — which is typical of
        # contradicting coverage, since it describes the opposite outcome in
        # its own vocabulary. Rejecting on predicate overlap alone dropped
        # those, leaving one-sided "supported" verdicts on contested claims.
        # A definite action match is the stronger predicate signal, so it
        # overrides this floor.
        if relevance_score.predicate_match_score < 0.25 and relevance_score.action_match_score < 1.0:
            return False
        return True
    
    def filter_documents(
        self,
        claim: str,
        documents: List[dict],
        strict: bool = False,
    ) -> tuple[List[dict], List[dict]]:
        """
        Filter a list of documents by relevance to the claim.
        
        Args:
            claim: The claim to verify
            documents: List of documents with 'title', 'snippet', 'text' keys
            strict: If True, use strict filtering for final evidence presentation
            
        Returns:
            Tuple of (included_documents, excluded_documents)
        """
        decomp = decompose_claim(claim)
        included = []
        excluded = []
        
        for doc in documents:
            relevance = self.assess_document_relevance(
                claim,
                doc.get('title', ''),
                doc.get('snippet', ''),
                doc.get('text', '')[:2000],  # Use first 2000 chars
                decomp,
            )
            
            if self.should_include_document(relevance, strict=strict):
                doc_with_score = {**doc, '_relevance_score': relevance}
                included.append(doc_with_score)
            else:
                excluded.append({**doc, '_relevance_score': relevance})
        
        # Sort included by relevance score
        included.sort(key=lambda d: d['_relevance_score'].overall_relevance, reverse=True)
        
        return included, excluded


# Testing
if __name__ == "__main__":
    filter_instance = RelevanceFilter()
    
    test_claim = "The name of united states is being changed to india by 2050."
    
    test_documents = [
        {
            'title': 'Kennedy Center reportedly changed rules before vote to add Trump\'s name',
            'snippet': 'The Kennedy Center made changes to voting procedures...',
            'text': 'The Kennedy Center reportedly changed its rules...',
        },
        {
            'title': 'A job that changed me: Being a theatre usher cracked open my heart to beauty',
            'snippet': 'Working as a theatre usher changed my perspective on life...',
            'text': 'I worked as a theatre usher and it changed everything...',
        },
        {
            'title': 'Geopolitical tensions as India and US relations shift',
            'snippet': 'The relationship between India and the United States continues to evolve...',
            'text': 'Relations between India and the United States have been changing...',
        },
    ]
    
    print(f"Claim: {test_claim}\n")
    
    included, excluded = filter_instance.filter_documents(test_claim, test_documents)
    
    print(f"INCLUDED ({len(included)}):")
    for doc in included:
        score = doc['_relevance_score']
        print(f"  - {doc['title']}")
        print(f"    Overall Relevance: {score.overall_relevance:.2f}")
        print(f"    Entity Match: {score.entity_match_score:.2f}")
        print(f"    Predicate Match: {score.predicate_match_score:.2f}")
        print()
    
    print(f"\nEXCLUDED ({len(excluded)}):")
    for doc in excluded:
        score = doc['_relevance_score']
        print(f"  - {doc['title']}")
        print(f"    Overall Relevance: {score.overall_relevance:.2f}")
        print(f"    Reasons: {', '.join(score.reasons_excluded)}")
        print()
