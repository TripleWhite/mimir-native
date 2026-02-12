"""Query Analyzer - Analyze and process search queries"""

import re
from enum import Enum, auto
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


class QueryIntent(Enum):
    """Types of query intents"""
    WHEN = "when"          # Temporal questions
    WHAT = "what"          # Factual questions
    WHO = "who"            # Entity questions
    WHERE = "where"        # Location questions
    HOW = "how"            # Process questions
    WHY = "why"            # Reasoning questions
    WOULD = "would"        # Hypothetical questions
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    """Analysis result for a query"""
    original: str
    intent: QueryIntent
    keywords: List[str]
    temporal_expressions: List[str]
    entities: List[str]
    rewritten: Optional[str] = None


class QueryAnalyzer:
    """
    Analyzes queries to determine intent and extract features.
    
    Based on LoCoMo optimization insights.
    """
    
    # Intent patterns
    INTENT_PATTERNS = {
        QueryIntent.WHEN: [
            r'^when\b', r'what time', r'what date', 
            r'how long', r'how soon', r'how often'
        ],
        QueryIntent.WHAT: [
            r'^what\b', r'^which\b', r'^describe\b'
        ],
        QueryIntent.WHO: [
            r'^who\b', r'^whom\b', r'^whose\b'
        ],
        QueryIntent.WHERE: [
            r'^where\b', r'^what location'
        ],
        QueryIntent.HOW: [
            r'^how\b', r'^what way'
        ],
        QueryIntent.WHY: [
            r'^why\b', r'^what reason'
        ],
        QueryIntent.WOULD: [
            r'^would\b', r'^will\b', r'^could\b', r'^should\b'
        ]
    }
    
    # Temporal patterns
    TEMPORAL_PATTERNS = [
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(st|nd|rd|th)?,?\s+20\d{2}',
        r'\d{1,2}(st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}',
        r'(last|this|next)\s+(week|month|year)',
        r'\d{4}',
        r'(week|day|month|year)\s+(before|after)',
        r'yesterday|today|tomorrow'
    ]
    
    def __init__(self):
        self.compiled_intents = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        self.compiled_temporal = [re.compile(p, re.IGNORECASE) for p in self.TEMPORAL_PATTERNS]
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a query.
        
        Args:
            query: The query string
            
        Returns:
            QueryAnalysis with intent and features
        """
        query_lower = query.lower().strip()
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # Extract temporal expressions
        temporal = self._extract_temporal(query)
        
        # Extract entities (simple approach)
        entities = self._extract_entities(query)
        
        # Rewrite query for better retrieval
        rewritten = self._rewrite_query(query, intent)
        
        return QueryAnalysis(
            original=query,
            intent=intent,
            keywords=keywords,
            temporal_expressions=temporal,
            entities=entities,
            rewritten=rewritten
        )
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent"""
        for intent, patterns in self.compiled_intents.items():
            for pattern in patterns:
                if pattern.search(query):
                    return intent
        return QueryIntent.UNKNOWN
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords"""
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                     'be', 'been', 'being', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used',
                     'to', 'of', 'in', 'for', 'on', 'with',
                     'at', 'by', 'from', 'as', 'into', 'through'}
        
        words = re.findall(r'\b[a-z]+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return list(set(keywords))[:10]  # Top 10 unique keywords
    
    def _extract_temporal(self, query: str) -> List[str]:
        """Extract temporal expressions"""
        temporal = []
        for pattern in self.compiled_temporal:
            matches = pattern.findall(query)
            if matches:
                if isinstance(matches[0], tuple):
                    temporal.append(' '.join(matches[0]))
                else:
                    temporal.extend(matches)
        return temporal
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities (simple)"""
        # Look for capitalized words
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        return list(set(entities))
    
    def _rewrite_query(self, query: str, intent: QueryIntent) -> Optional[str]:
        """Rewrite query for better retrieval"""
        # For temporal queries, ensure date formats are consistent
        if intent == QueryIntent.WHEN:
            # Normalize date formats
            rewritten = query
            # Could add more sophisticated rewriting here
            return rewritten
        
        return None
