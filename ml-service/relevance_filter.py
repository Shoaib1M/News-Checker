"""
Relevance Filtering: Strict claim-aware filtering to eliminate irrelevant search results.

This module ensures that only genuinely relevant articles are presented as evidence.
It prevents keyword collisions (e.g., "2050" or "changed" matching irrelevant articles).
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Set
from claim_decomposer import decompose_claim, ClaimDecomposition


@dataclass
class RelevanceScore:
    """Detailed relevance assessment for a document."""
    entity_match_score: float  # 0.0-1.0: How many key entities are mentioned
    predicate_match_score: float  # 0.0-1.0: How relevant predicates are
    semantic_coherence: float  # 0.0-1.0: How well entities and predicates connect
    keyword_specificity: float  # 0.0-1.0: Avoids generic keyword overlap
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
        entity_match = self._score_entity_match(decomp, doc_entities)
        predicate_match = self._score_predicate_match(decomp, doc_keywords)
        semantic_coherence = self._score_semantic_coherence(decomp, doc_entities, doc_keywords)
        keyword_specificity = self._score_keyword_specificity(claim, decomp, doc_combined)
        
        # Determine overall relevance
        weights = {
            'entity_match': 0.40,
            'predicate_match': 0.25,
            'semantic_coherence': 0.20,
            'keyword_specificity': 0.15,
        }
        
        overall = (
            entity_match * weights['entity_match'] +
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
        
        if entity_match < self.MIN_ENTITY_MATCH:
            reasons_excluded.append("Very few claim entities mentioned")
        if overall < self.MIN_OVERALL_RELEVANCE:
            reasons_excluded.append("Insufficient relevance to claim")
        if keyword_specificity < 0.20:
            reasons_excluded.append("Generic keyword overlap only")
        
        return RelevanceScore(
            entity_match_score=entity_match,
            predicate_match_score=predicate_match,
            semantic_coherence=semantic_coherence,
            keyword_specificity=keyword_specificity,
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
        for phrase in ("united states", "the united states", "us", "usa",
                       "great wall", "china", "india", "nasa", "earth"):
            if re.search(rf"\b{re.escape(phrase)}\b", lowered):
                entities.add(phrase)
        
        return entities
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        words = re.findall(r'\b([a-z]+(?:[a-z\-]*[a-z])?)\b', text.lower())
        return {w for w in words if len(w) > 2 and w not in self.STOPWORDS}
    
    def _score_entity_match(self, decomp: ClaimDecomposition, doc_entities: Set[str]) -> float:
        """
        Score how many entities from the claim appear in the document.
        
        Returns 0.0-1.0 where 1.0 means all key entities appear.
        """
        if not decomp.primary_entities:
            return 0.5  # Neutral for non-entity claims
        
        claim_entities = {e.lower() for e in decomp.primary_entities}
        matched = set()
        for entity in claim_entities:
            variants = {entity}
            if entity == "united states":
                variants.update({"us", "usa", "america", "united states"})
            if entity == "the great wall":
                variants.add("great wall")
            if any(
                variant in candidate or candidate in variant
                for variant in variants
                for candidate in doc_entities
            ):
                matched.add(entity)
        
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
        if relevance_score.entity_match_score == 0 and relevance_score.keyword_specificity < 0.35:
            return False
        if relevance_score.entity_match_score < 0.5 and relevance_score.predicate_match_score < 0.5:
            return False
        if relevance_score.predicate_match_score < 0.25:
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
