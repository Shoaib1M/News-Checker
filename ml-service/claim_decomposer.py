"""
Claim Decomposition: Structured understanding of user claims for targeted evidence retrieval.

This module extracts the semantic structure of a claim before searching, enabling
claim-specific query generation and relevance filtering.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Set, List


@dataclass(frozen=True)
class ClaimDecomposition:
    """Structured representation of a claim for targeted evidence retrieval."""
    original_claim: str
    claim_type: str  # "factual", "numerical", "temporal", "geopolitical", "policy", "causal", "comparative"
    primary_entities: List[str]  # Named entities (people, places, organizations)
    entities_in_claim: Set[str]  # All significant nouns
    core_predicates: List[str]  # The main assertions/actions
    numerical_values: List[str]  # Any numbers, percentages, dates
    temporal_modifiers: List[str]  # Time references (by 2050, last year, etc.)
    temporal_constraints: Optional[str]  # "past", "present", "future", or None
    sentiment_direction: Optional[str]  # "positive", "negative", "neutral" for directional claims
    confidence_markers: List[str]  # Words like "might", "likely", "definitely"
    modality: str  # "factual", "hypothetical", "subjective", "speculative"
    contradicting_entities: Set[str]  # Entities that might form contradictory claims
    negation: bool = False
    attribution: Optional[str] = None
    
    def search_focus_entities(self) -> List[str]:
        """Return entities most important for search queries."""
        return self.primary_entities[:3] if self.primary_entities else list(self.entities_in_claim)[:3]


# Capitalized words that are never the entity of a claim. Sentence-initial
# capitalization and mid-sentence emphasis both produce these.
_CAPITALIZED_NON_ENTITIES = {
    'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'Is', 'Was', 'Are',
    'Were', 'Be', 'Been', 'Has', 'Have', 'Had', 'Do', 'Does', 'Did', 'At',
    'By', 'To', 'In', 'On', 'Of', 'And', 'Or', 'But', 'If', 'I', 'As', 'For',
    'With', 'From', 'It', 'Its', 'All', 'Across', 'After', 'Before', 'Not',
    'No', 'New', 'Now', 'Will', 'Would', 'Can', 'Could', 'May', 'Might',
    'Said', 'Says', 'According',
}


def _looks_like_verb(word: str) -> bool:
    """True for single words that are verb forms rather than names.

    Users capitalize verbs mid-sentence surprisingly often ("is Banning
    google"), and a verb admitted as a named entity both dilutes the entity
    match score and sends the retriever looking for the wrong thing.
    """
    lowered = word.lower()
    return (
        " " not in word
        and (lowered.endswith("ing") or lowered.endswith("ed"))
        and len(word) > 4
    )


def extract_entities(text: str) -> List[str]:
    """Extract named entities from text.

    Handles three cases the naive capitalized-phrase regex gets wrong:
    leading articles ("The United States" and "United States" are one
    entity, not two), capitalized verbs, and well-known proper nouns the
    user left lowercase ("google").
    """
    entities = []

    # First, try to match capitalized phrases (standard entity case)
    pattern = r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\b'
    for match in re.finditer(pattern, text):
        entity = match.group(1)
        # "The United States" and "United States" are the same entity; keep
        # one form so a single match doesn't score as a partial one.
        entity = re.sub(r'^(?:The|A|An)\s+', '', entity).strip()
        if not entity or entity in _CAPITALIZED_NON_ENTITIES:
            continue
        if _looks_like_verb(entity):
            continue
        entities.append(entity)

    # Also look for known entities that are commonly written lowercase.
    known_multiword_patterns = [
        r'\bunited\s+states\b',
        r'\bunited\s+kingdom\b',
        r'\bunited\s+nations\b',
        r'\bindia\b',  # Single word but important
        r'\bwater\b',
        r'\btriangle\b',
        r'\bmoon\b',
        r'\bchina\b',
        r'\bnasa\b',
        r'\basteroid\b',
        r'\b(us|usa)\b',  # Abbreviations
        # Organizations that dominate news claims and are routinely typed
        # in lowercase, which the capitalized-phrase regex above misses.
        r'\bgoogle\b', r'\bapple\b', r'\bmicrosoft\b', r'\bamazon\b',
        r'\bmeta\b', r'\bfacebook\b', r'\btwitter\b', r'\btiktok\b',
        r'\bopenai\b', r'\btesla\b',
    ]

    for pattern in known_multiword_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Normalize to title case
            if isinstance(match, tuple):
                normalized = match[0].capitalize()
            else:
                normalized = ' '.join(word.capitalize() for word in match.split())
            # "US"/"USA" is the same entity as "United States". Left distinct,
            # a claim written "US bans Google" produced the entity "Us", which
            # the relevance filter then searched for case-insensitively and
            # matched against the English pronoun.
            if normalized.lower() in {"us", "usa"}:
                normalized = "United States"
            if (
                len(normalized) > 2
                and normalized.lower() not in {"c", "the", "a", "an"}
                and normalized.lower() not in {e.lower() for e in entities}
            ):
                entities.append(normalized)
    
    return list(dict.fromkeys(entities))  # Preserve order, remove duplicates


def extract_temporal_info(text: str) -> tuple[List[str], Optional[str]]:
    """Extract time references and classify as past/present/future."""
    temporal_refs = []
    temporal_constraint = None
    
    # Match various time patterns
    patterns = [
        r'\b(by\s+\d{4}|\d{4})\b',  # "by 2050", "2050"
        r'\b(next|last|this)\s+(year|month|week|decade|century)\b',  # "next year"
        r'\b(tomorrow|today|yesterday)\b',  # Direct time words
        r'\b(in\s+\d+\s+(?:years?|months?|weeks?|days?))\b',  # "in 5 years"
        r'\b(early|late|mid)\s+(2\d{3})\b',  # "early 2050"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                temporal_refs.append(" ".join(part for part in match if part))
            else:
                temporal_refs.append(match)
    
    # Determine temporal constraint
    future_indicators = re.search(r'\b(by|in|will|planned|planned|proposed|estimated)\b', text, re.IGNORECASE)
    past_indicators = re.search(r'\b(was|were|did|had|previously|historically)\b', text, re.IGNORECASE)
    
    if re.search(r'\b(by\s+\d{4}|will|planned|proposed|estimated|future)\b', text, re.IGNORECASE):
        temporal_constraint = "future"
    elif re.search(r'\b(was|were|did|had|previously|already|historically)\b', text, re.IGNORECASE):
        temporal_constraint = "past"
    else:
        temporal_constraint = "present"
    
    return temporal_refs, temporal_constraint


def extract_numerical_info(text: str) -> List[str]:
    """Extract numbers, percentages, years, and amounts."""
    numbers = []
    patterns = [
        r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*%\b',  # Percentages
        r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?\b',  # Dollar amounts
        r'\b\d{4}\b',  # Years
        r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|thousand)\b',  # Large numbers
        r'\b\d+(?:\.\d+)?\s*(?:degrees|meters|kilometers|miles|percent)\b',  # Measurements
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        numbers.extend(matches)
    
    return list(dict.fromkeys(numbers))  # Remove duplicates while preserving order


def classify_claim_type(text: str) -> str:
    """Classify the type of claim based on linguistic patterns."""
    
    if re.search(r'\b(opinion|think|believe|best|worst|beautiful|amazing)\b', text, re.IGNORECASE):
        return "subjective"

    # Numerical claim
    if re.search(r'\b\d+(?:%|million|billion|thousand)\b', text):
        return "numerical"
    
    # Temporal/future claim
    if re.search(r'\b(by|will|planned|proposed|estimated|expected)\s+\d{4}\b|will\s+be|is being', text, re.IGNORECASE):
        return "temporal"
    
    # Policy/governance claim
    if re.search(r'\b(law|policy|rule|regulation|ban|bill|act|government|congress|parliament)\b', text, re.IGNORECASE):
        return "policy"
    
    # Causal claim
    if re.search(r'\b(cause|caused|lead to|leads to|results? in|due to|because of)\b', text, re.IGNORECASE):
        return "causal"
    
    # Comparative claim
    if re.search(r'\b(more|less|greater|smaller|than|vs|compared to)\b', text, re.IGNORECASE):
        return "comparative"
    
    # News/current-event claim
    if re.search(r'\b(today|yesterday|announced|confirms?|resigned?|reported|breaking|this week)\b', text, re.IGNORECASE):
        return "news"

    # Geopolitical claim
    if re.search(r'\b(country|countries|nation|nations|government|president|minister|border|nation|state|renamed|changed name)\b', text, re.IGNORECASE):
        return "geopolitical"
    
    # Default
    return "factual"


def extract_core_predicates(text: str) -> List[str]:
    """Extract main verbs and action phrases."""
    predicates = []

    # Common verb patterns
    verb_patterns = [
        r'\b(is|was|are|were)\s+(?:being\s+)?(\w+(?:\s+\w+)?)\b',  # "is being changed"
        r'\b(\w+ing)\b(?:\s+\w+)?\b',  # "changing", "renaming"
        r'\b(?:will|can|may|should)\s+(\w+)\b',  # "will change", "can increase"
    ]

    for pattern in verb_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                predicates.extend([m for m in match if m])
            else:
                predicates.append(match)

    # Capture ordinary present/past-tense assertions.  A fact-check engine must
    # preserve verbs like "banned", "renamed", "blocked", etc.; otherwise the
    # subject/action/object relationship is lost during retrieval.
    for verb in re.findall(
        r'\b(?:improves?|reduces?|increases?|decreases?|confirms?|announces?|'
        r'resigns?|collapses?|supports?|contradicts?|denies?|fails?|'
        r'passes?|hits?|causes?|bans?|banned|blocks?|blocked|approves?|approved|'
        r'rejects?|rejected|suspends?|suspended|halts?|halted|renames?|renamed|'
        r'changes?|changed|closes?|closed|opens?|opened|registers?|registered|'
        r'launches?|launched|claims?|claimed|called|says?|said|orders?|ordered)\b',
        text,
        re.IGNORECASE,
    ):
        predicates.append(verb)

    # Clean up common words that aren't meaningful predicates
    stop_predicates = {'is', 'was', 'are', 'were', 'be', 'have', 'has', 'do', 'does', 'did'}
    predicates = [p for p in predicates if p.lower() not in stop_predicates]

    return list(dict.fromkeys(predicates))  # Remove duplicates


def extract_modality(text: str) -> str:
    """Determine whether the claim is factual, hypothetical, subjective, or speculative."""
    
    if re.search(r'\b(might|could|may|possibly|perhaps|allegedly|reportedly)\b', text, re.IGNORECASE):
        return "speculative"
    
    if re.search(r'\b(if|would|could|hypothetical)\b', text, re.IGNORECASE):
        return "hypothetical"
    
    if re.search(r'\b(I think|in my opinion|arguably|seems|appears|believe)\b', text, re.IGNORECASE):
        return "subjective"
    
    return "factual"


def detect_contradicting_entities(primary_entities: List[str], text: str) -> Set[str]:
    """Find entities that might form contradictory assertions."""
    contradicting = set()
    
    # Look for common contradictory patterns
    if "renamed" in text.lower() or "changed name" in text.lower():
        # For name-change claims, find the entities being related
        for entity in primary_entities:
            # Look for "X to Y" or "X into Y" patterns
            pattern = rf'{re.escape(entity)}\s+(?:to|into|as)\s+(\w+(?:\s+\w+)?)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            contradicting.update(matches)
    
    return contradicting


def decompose_claim(claim: str) -> ClaimDecomposition:
    """
    Decompose a claim into structured components for targeted retrieval.

    Args:
        claim: The user's claim text

    Returns:
        ClaimDecomposition with structured understanding
    """
    entities = extract_entities(claim)
    temporal_refs, temporal_constraint = extract_temporal_info(claim)
    numbers = extract_numerical_info(claim)
    predicates = extract_core_predicates(claim)
    claim_type = classify_claim_type(claim)
    modality = extract_modality(claim)
    negation = bool(re.search(
        r'\b(?:not|never|no|without|denies|denied|deny|denying|refutes|refuted|refute|didn\'t|did not|doesn\'t|does not|wasn\'t|was not|weren\'t|were not|cannot|can\'t)\b',
        claim,
        re.IGNORECASE,
    ))
    attribution_match = re.search(
        r'\b(?:according to|officials|experts|reportedly|allegedly|said|says|claims?|claimed|denies|denied|confirms?|confirmed|warns?|warned)\b',
        claim,
        re.IGNORECASE,
    )
    attribution = attribution_match.group(0).lower() if attribution_match else None

    # Extract all significant nouns (not just capitalized entities)
    nouns = set(re.findall(r'\b([a-z]+(?:ies)?|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', claim.lower()))
    stopwords = {'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'to', 'for', 'of', 'with', 'from', 'as', 'be', 'this', 'that', 'it', 'they', 'he', 'she', 'we', 'you'}
    entities_in_claim = {n for n in nouns if len(n) > 2 and n not in stopwords}

    # Detect sentiment direction for directional claims
    sentiment_direction = None
    if claim_type in {"numerical", "comparative"}:
        if re.search(r'\b(increase|rise|grow|higher|more|improve|better)\b', claim, re.IGNORECASE):
            sentiment_direction = "positive"
        elif re.search(r'\b(decrease|fall|decline|lower|less|worse)\b', claim, re.IGNORECASE):
            sentiment_direction = "negative"

    # Extract confidence markers
    confidence_markers = re.findall(r'\b(definitely|certainly|undoubtedly|likely|probably|might|could|possibly)\b', claim, re.IGNORECASE)

    contradicting = detect_contradicting_entities(entities, claim)

    return ClaimDecomposition(
        original_claim=claim,
        claim_type=claim_type,
        primary_entities=entities,
        entities_in_claim=entities_in_claim,
        core_predicates=predicates,
        numerical_values=numbers,
        temporal_modifiers=temporal_refs,
        temporal_constraints=temporal_constraint,
        sentiment_direction=sentiment_direction,
        confidence_markers=confidence_markers,
        modality=modality,
        contradicting_entities=contradicting,
        negation=negation,
        attribution=attribution,
    )


# Testing
if __name__ == "__main__":
    test_claims = [
        "The name of united states is being changed to india by 2050.",
        "Water freezes at 0°C at sea level.",
        "A triangle has four sides.",
        "The Great Wall of China is visible from the Moon with the naked eye.",
    ]
    
    for claim in test_claims:
        decomp = decompose_claim(claim)
        print(f"\n{'='*60}")
        print(f"Claim: {claim}")
        print(f"Type: {decomp.claim_type}")
        print(f"Entities: {decomp.primary_entities}")
        print(f"Predicates: {decomp.core_predicates}")
        print(f"Temporal Constraint: {decomp.temporal_constraints}")
        print(f"Numbers: {decomp.numerical_values}")
