"""
Query Generation: Create claim-specific search queries for targeted evidence retrieval.

This module generates multiple query variants designed to find supporting,
contradicting, and contextual evidence specific to the claim.
"""

import re
from typing import List, Set
from claim_decomposer import ClaimDecomposition, decompose_claim
from event_vocabulary import surface_forms_in


# Only the first few queries are ever dispatched (the pipeline sends four),
# so their ORDER decides what the system actually retrieves. Ranking by
# purpose rather than by generation order was worth more than any other
# retrieval change: previously the four queries sent were close to the four
# least useful ones available.
#
# For "The prime minister of India resigned this morning" the dispatched set
# was:
#     "The prime minister of India resigned this morning"   (exact phrase)
#     The prime minister of India resigned this morning
#     "India" "morning"
#     India morning
# — an exact-phrase search for the user's own paraphrase, which matches
# nothing, and two queries built on "morning" because that is the token the
# predicate regex happened to return first. The one query that would have
# found the story, containing "resigned", was generated but ranked below the
# cut.
_PURPOSE_RANK = {
    "proposition": 0,              # the claim as written — the best general query
    "entity_action": 1,            # subject + what it is claimed to have done
    "entity_action_quoted": 2,
    "entity_relationship": 3,      # subject + object, for relational claims
    "fact_check": 4,
    "numerical_full": 5,
    "numerical_entity": 6,
    "contradiction_search": 7,
    "proposition_verification": 8,
    "entity_predicate": 9,         # regex-derived predicates; often junk
    "official_source": 10,
    "policy_context": 11,
    "news_context": 12,
    "entity_temporal": 13,
    "direct_relationship": 14,
    "name_change": 15,
    # An exact-phrase search for a user's paraphrase almost never matches, and
    # returns nothing at all from a news index. Kept last so it only ever runs
    # when a claim produced almost no other queries.
    "exact_claim": 90,
}
_DEFAULT_RANK = 50


class QueryGenerator:
    """Generate multiple search queries optimized for different evidence types."""
    
    def __init__(self):
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'have', 'he', 'in', 'is', 'it', 'of', 'on', 'or', 'that',
            'the', 'to', 'was', 'were', 'will', 'with', 'this', 'these',
            'those', 'it', 'its', 'as', 'but', 'if', 'or', 'not', 'no',
        }
    
    def generate_queries(self, claim: str, decomp: ClaimDecomposition = None) -> List[dict]:
        """
        Generate multiple targeted search queries for different evidence types.
        
        Args:
            claim: The user's claim
            decomp: Pre-computed claim decomposition (will be computed if not provided)
            
        Returns:
            List of query dicts with 'query' and 'purpose' keys
        """
        if decomp is None:
            decomp = decompose_claim(claim)
        
        queries = [{
            'query': f'"{claim.strip().rstrip(".!?")}"',
            'purpose': 'exact_claim',
            'priority': 'lowest',
        }]

        # 1. Preserve the proposition itself. This is especially important for
        # headlines and claims whose wording differs from article wording.
        queries.append({
            'query': claim.strip().rstrip(".!?"),
            'purpose': 'proposition',
            'priority': 'highest',
        })

        # 1b. Subject + the event the claim asserts, in the claim's own
        # wording. This is the query most likely to find the actual story,
        # and it is built from the event vocabulary rather than from the
        # predicate regex, which returns whatever token it matches first.
        queries.extend(self._generate_action_queries(claim, decomp))

        # 2. ENTITY-FOCUSED QUERIES
        # These find articles about the specific entities involved
        if decomp.primary_entities:
            entity_queries = self._generate_entity_queries(decomp)
            queries.extend(entity_queries)
        
        # 3. PREDICATE-FOCUSED QUERIES
        # These find articles about the specific actions/changes being claimed
        if decomp.core_predicates:
            predicate_queries = self._generate_predicate_queries(decomp)
            queries.extend(predicate_queries)
        
        # 4. NUMERICAL/COMPARATIVE QUERIES
        # For claims with numbers, dates, or comparisons
        if decomp.numerical_values or decomp.claim_type == "comparative":
            numerical_queries = self._generate_numerical_queries(decomp)
            queries.extend(numerical_queries)
        
        # 5. CONTRADICTORY/VERIFICATION QUERIES
        # Explicitly search for evidence contradicting or verifying the claim
        contradictory_queries = self._generate_verification_queries(decomp)
        queries.extend(contradictory_queries)
        
        # 6. CONTEXTUAL QUERIES
        # For claims about policies, organizations, or institutions
        if decomp.claim_type in {"policy", "geopolitical"}:
            contextual_queries = self._generate_contextual_queries(decomp)
            queries.extend(contextual_queries)
        
        # 7. FACT-CHECK SPECIFIC QUERIES
        # Explicitly search for fact-checking sources
        factcheck_query = self._generate_factcheck_query(decomp)
        if factcheck_query:
            queries.append(factcheck_query)
        
        # Remove duplicates, then order by how useful the query is likely to
        # be. Only the first few are ever dispatched, so this ordering is
        # effectively the retrieval strategy.
        seen = set()
        unique_queries = []
        for q in queries:
            query_text = q['query'].lower().strip()
            if query_text and query_text not in seen:
                seen.add(query_text)
                unique_queries.append(q)

        unique_queries.sort(
            key=lambda q: _PURPOSE_RANK.get(q['purpose'], _DEFAULT_RANK)
        )
        return unique_queries[:12]

    def _generate_action_queries(
        self, claim: str, decomp: ClaimDecomposition
    ) -> List[dict]:
        """Subject + the event the claim asserts, in the claim's own words."""
        actions = surface_forms_in(claim)
        if not actions:
            return []

        action = actions[0]
        entities = [
            entity for entity in decomp.primary_entities[:2]
            if len(entity.strip()) > 2
        ]
        if not entities:
            return []

        queries = [{
            'query': f'{entities[0]} {action}',
            'purpose': 'entity_action',
            'priority': 'highest',
        }, {
            'query': f'"{entities[0]}" "{action}"',
            'purpose': 'entity_action_quoted',
            'priority': 'high',
        }]
        if len(entities) > 1:
            queries.append({
                'query': f'{entities[0]} {entities[1]} {action}',
                'purpose': 'entity_action',
                'priority': 'highest',
            })
        return queries
    
    def _generate_entity_queries(self, decomp: ClaimDecomposition) -> List[dict]:
        """Generate queries focused on the primary entities."""
        queries = []
        entities = [
            entity for entity in decomp.primary_entities[:3]
            if len(entity.strip()) > 2 and entity.lower() not in {'the', 'water', 'earth', 'c'}
        ]
        
        if len(entities) >= 2 and all(len(entity) > 2 for entity in entities[:2]):
            # Query with both entities
            query_text = '"' + '" "'.join(entities) + '"'
            queries.append({
                'query': query_text,
                'purpose': 'entity_relationship',
                'priority': 'high',
            })
        
        # Individual entity queries combined with predicates
        for entity in entities[:2]:
            if decomp.core_predicates:
                for predicate in decomp.core_predicates[:2]:
                    query_text = f'"{entity}" "{predicate}"'
                    queries.append({
                        'query': query_text,
                        'purpose': 'entity_predicate',
                        'priority': 'high',
                    })
            
            # Entity + temporal constraint
            if decomp.temporal_modifiers:
                for temporal in decomp.temporal_modifiers[:1]:
                    query_text = f'{entity} {temporal}'
                    queries.append({
                        'query': query_text,
                        'purpose': 'entity_temporal',
                        'priority': 'medium',
                    })
        
        return queries
    
    def _generate_predicate_queries(self, decomp: ClaimDecomposition) -> List[dict]:
        """Generate queries focused on the predicates/actions."""
        queries = []
        predicates = [
            predicate for predicate in decomp.core_predicates[:2]
            if predicate.lower() not in {'being', 'is', 'are', 'was', 'were'}
        ]
        
        for predicate in predicates:
            # Predicate + entity
            if decomp.primary_entities:
                entity = decomp.primary_entities[0]
                query_text = f'{entity} {predicate}'
                queries.append({
                    'query': query_text,
                    'purpose': 'entity_predicate',
                    'priority': 'high',
                })
        
        return queries
    
    def _generate_numerical_queries(self, decomp: ClaimDecomposition) -> List[dict]:
        """Generate queries for numerical/comparative claims."""
        queries = []
        
        # Numerical value + primary entity
        if decomp.numerical_values and decomp.primary_entities:
            for number in decomp.numerical_values[:2]:
                for entity in decomp.primary_entities[:2]:
                    query_text = f'{entity} {number}'
                    queries.append({
                        'query': query_text,
                        'purpose': 'numerical_entity',
                        'priority': 'high',
                    })
                    
                    # Add predicate if available
                    if decomp.core_predicates:
                        query_text = f'{entity} {decomp.core_predicates[0]} {number}'
                        queries.append({
                            'query': query_text,
                            'purpose': 'numerical_full',
                            'priority': 'high',
                        })
        
        return queries
    
    def _generate_verification_queries(self, decomp: ClaimDecomposition) -> List[dict]:
        """Generate queries that look for contradictory or verification evidence."""
        queries = []
        
        # Search the proposition as a relationship, preserving subject/object.
        # Gated on the claim actually being about renaming: these three
        # queries were unconditional, so a claim about *buying* the Eiffel
        # Tower spent three of its slots searching for a name change.
        # "renamed", "rebranded", "new name", and either word order of
        # name/change ("changed its name", "the name ... is being changed").
        renaming = bool(re.search(
            r"\b(?:renam\w*|rebrand\w*|new\s+name|"
            r"name\b[\w\s]{0,30}?\bchang\w*|chang\w*[\w\s]{0,20}?\bname)\b",
            decomp.original_claim, re.IGNORECASE,
        ))
        if renaming and len(decomp.primary_entities) >= 2:
            entity, object_entity = decomp.primary_entities[:2]
            for relation in ('renamed', 'name change', 'rename'):
                queries.append({
                    'query': f'"{entity}" "{object_entity}" {relation}',
                    'purpose': 'proposition_verification',
                    'priority': 'highest',
                })

        if decomp.contradicting_entities:
            for entity in decomp.primary_entities[:1]:
                for contra_entity in decomp.contradicting_entities:
                    # Direct "X to Y" or "X become Y" query
                    query_text = f'"{entity}" "{contra_entity}"'
                    queries.append({
                        'query': query_text,
                        'purpose': 'direct_relationship',
                        'priority': 'high',
                    })
                    
                    query_text = f'{entity} renamed {contra_entity}'
                    queries.append({
                        'query': query_text,
                        'purpose': 'name_change',
                        'priority': 'high',
                    })
        
        # Search for denials/contradictions
        if decomp.core_predicates and decomp.primary_entities:
            for entity in decomp.primary_entities[:1]:
                for predicate in decomp.core_predicates[:1]:
                    query_text = f'"{entity}" "{predicate}" (false OR denied OR debunked OR fact check)'
                    queries.append({
                        'query': query_text,
                        'purpose': 'contradiction_search',
                        'priority': 'medium',
                    })
        
        return queries
    
    def _generate_contextual_queries(self, decomp: ClaimDecomposition) -> List[dict]:
        """Generate contextual queries for policy/geopolitical claims."""
        queries = []
        
        if decomp.claim_type == "geopolitical" and decomp.primary_entities:
            entity = decomp.primary_entities[0]
            
            # Government/official sources
            queries.append({
                'query': f'{entity} official government policy',
                'purpose': 'official_source',
                'priority': 'high',
            })
            
            # News about the entity
            queries.append({
                'query': f'{entity} announcement news',
                'purpose': 'news_context',
                'priority': 'medium',
            })
        
        if decomp.claim_type == "policy" and decomp.primary_entities:
            entity = decomp.primary_entities[0]
            
            queries.append({
                'query': f'{entity} law policy bill congress',
                'purpose': 'policy_context',
                'priority': 'high',
            })
        
        return queries
    
    def _generate_factcheck_query(self, decomp: ClaimDecomposition) -> dict or None:
        """Generate a fact-check specific search."""
        if not decomp.primary_entities and not decomp.core_predicates:
            return None
        
        # Fact-check the main entity/predicate combination
        key_term = decomp.primary_entities[0] if decomp.primary_entities else ''
        # Prefer the claim's actual event word over the first regex-derived
        # predicate: "India resigned fact check" finds the story,
        # "India morning fact check" finds nothing.
        actions = surface_forms_in(decomp.original_claim)
        if actions:
            key_term += f' {actions[0]}'
        elif decomp.core_predicates:
            key_term += f' {decomp.core_predicates[0]}'
        
        if key_term:
            return {
                'query': f'{key_term} fact check verify',
                'purpose': 'fact_check',
                'priority': 'high',
            }
        
        return None


def generate_multiple_search_variants(claim: str) -> List[str]:
    """
    Convenience function to generate search queries.
    
    Args:
        claim: The user's claim
        
    Returns:
        List of search query strings
    """
    generator = QueryGenerator()
    decomp = decompose_claim(claim)
    queries = generator.generate_queries(claim, decomp)
    return [q['query'] for q in queries]


# Testing
if __name__ == "__main__":
    test_claims = [
        "The name of united states is being changed to india by 2050.",
        "Water freezes at 0°C at sea level.",
        "Crime is increasing in major US cities.",
        "The Great Wall of China is visible from the Moon with the naked eye.",
    ]
    
    generator = QueryGenerator()
    
    for claim in test_claims:
        print(f"\n{'='*70}")
        print(f"Claim: {claim}")
        print(f"{'-'*70}")
        
        queries = generator.generate_queries(claim)
        for i, q in enumerate(queries, 1):
            print(f"{i}. [{q['purpose']}] {q['query']}")
