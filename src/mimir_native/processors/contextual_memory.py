"""
Contextual Memory Generator - Supermemory SOTA Architecture

Converts raw chunks into atomic, contextualized memories.
Based on: https://supermemory.ai/research
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re


@dataclass
class ContextualMemory:
    """
    Atomic fact with context and temporal metadata.
    
    Features:
    - Atomic fact: Standalone, clear piece of information
    - Source chunk: Original text span
    - Context summary: Surrounding context for disambiguation
    - Temporal metadata: Dual timestamp system
    """
    id: str
    fact: str  # Atomic, standalone fact
    source_chunk: str  # Original text
    context_summary: str  # Surrounding context
    document_date: datetime  # When the conversation took place
    event_date: Optional[datetime] = None  # When the event actually occurred
    entities: List[str] = None  # Named entities in the fact
    confidence: float = 1.0  # Extraction confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'fact': self.fact,
            'source_chunk': self.source_chunk,
            'context_summary': self.context_summary,
            'document_date': self.document_date.isoformat() if self.document_date else None,
            'event_date': self.event_date.isoformat() if self.event_date else None,
            'entities': self.entities or [],
            'confidence': self.confidence
        }


class ContextualMemoryGenerator:
    """
    Generates contextual memories from conversation chunks.
    
    Key insight from Supermemory: Don't store raw chunks,
    store atomic facts with context for high-precision retrieval.
    """
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional LLM client for advanced extraction
        """
        self.llm = llm_client
        self.chunk_counter = 0
        
    def generate_memories(
        self,
        chunk: str,
        document_date: datetime,
        context: str = "",
        session_summary: str = ""
    ) -> List[ContextualMemory]:
        """
        Generate contextual memories from a chunk.
        
        Args:
            chunk: Raw text chunk (e.g., conversation turn)
            document_date: When this conversation took place
            context: Surrounding context for disambiguation
            session_summary: High-level summary of the session
            
        Returns:
            List of contextualized atomic memories
        """
        # Method 1: Rule-based extraction (fast, no LLM needed)
        memories = self._rule_based_extraction(chunk, document_date, context)
        
        # Method 2: LLM-based extraction (if available)
        if self.llm and len(chunk) > 50:
            llm_memories = self._llm_based_extraction(chunk, document_date, context, session_summary)
            memories.extend(llm_memories)
        
        # Deduplicate
        seen_facts = set()
        unique_memories = []
        for mem in memories:
            fact_key = mem.fact.lower().strip()
            if fact_key not in seen_facts:
                seen_facts.add(fact_key)
                unique_memories.append(mem)
        
        return unique_memories
    
    def _rule_based_extraction(
        self,
        chunk: str,
        document_date: datetime,
        context: str
    ) -> List[ContextualMemory]:
        """Extract memories using rules and patterns"""
        memories = []
        
        # Pattern 1: Direct statements about people
        # e.g., "Caroline adopted a cat named..."
        person_patterns = [
            r'([A-Z][a-z]+)\s+(adopted|got|bought|received)\s+(a|an)\s+(\w+)',
            r'([A-Z][a-z]+)\s+(is|was)\s+(\w+)',
            r'([A-Z][a-z]+)\s+likes?\s+(to\s+)?(\w+)',
            r'([A-Z][a-z]+)\s+went\s+to\s+([\w\s]+)',
            r'([A-Z][a-z]+)\s+visited\s+([\w\s]+)',
        ]
        
        for pattern in person_patterns:
            matches = re.finditer(pattern, chunk, re.IGNORECASE)
            for match in matches:
                fact = match.group(0)
                entities = self._extract_entities(chunk)
                
                mem = ContextualMemory(
                    id=f"mem_{self.chunk_counter}_{len(memories)}",
                    fact=fact,
                    source_chunk=chunk,
                    context_summary=context[:200] if context else chunk[:200],
                    document_date=document_date,
                    event_date=self._extract_event_date(chunk, document_date),
                    entities=entities,
                    confidence=0.8
                )
                memories.append(mem)
        
        # Pattern 2: Temporal events
        temporal_patterns = [
            r'on\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
            r'last\s+(week|month|year)',
            r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)',
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, chunk, re.IGNORECASE):
                # Extract the sentence containing temporal reference
                sentences = re.split(r'[.!?]+', chunk)
                for sent in sentences:
                    if re.search(pattern, sent, re.IGNORECASE):
                        entities = self._extract_entities(chunk)
                        
                        mem = ContextualMemory(
                            id=f"mem_{self.chunk_counter}_{len(memories)}",
                            fact=sent.strip(),
                            source_chunk=chunk,
                            context_summary=context[:200] if context else chunk[:200],
                            document_date=document_date,
                            event_date=self._extract_event_date(sent, document_date),
                            entities=entities,
                            confidence=0.75
                        )
                        memories.append(mem)
                        break
        
        self.chunk_counter += 1
        return memories
    
    def _llm_based_extraction(
        self,
        chunk: str,
        document_date: datetime,
        context: str,
        session_summary: str
    ) -> List[ContextualMemory]:
        """Extract memories using LLM (more accurate but slower)"""
        memories = []
        
        # If no LLM available, return empty
        if not self.llm:
            return memories
        
        # Construct prompt
        prompt = f"""Extract atomic facts from this conversation text.

Context: {context[:300] if context else 'N/A'}
Session Summary: {session_summary[:200] if session_summary else 'N/A'}

Text to analyze:
{chunk}

Instructions:
1. Extract each fact as a standalone, clear statement
2. Resolve any ambiguous references (pronouns like "he/she/it") using the context
3. Focus on: who, what, when, where
4. Output one fact per line

Format:
- FACT: [clear, standalone fact with all references resolved]

Facts:"""
        
        try:
            # This would call the LLM in a real implementation
            # For now, return empty - rule-based covers most cases
            pass
        except Exception as e:
            print(f"LLM extraction failed: {e}")
        
        return memories
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simple approach)"""
        # Look for capitalized words (names)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Filter out common words
        stop_words = {'I', 'The', 'A', 'An', 'This', 'That', 'It', 'He', 'She', 'They'}
        entities = [e for e in entities if e not in stop_words]
        return list(set(entities))[:5]  # Top 5 unique entities
    
    def _extract_event_date(self, text: str, document_date: datetime) -> Optional[datetime]:
        """
        Extract event date from text.
        
        This is different from document_date:
        - document_date: when the conversation happened
        - event_date: when the described event actually occurred
        """
        # Pattern: "last week" → document_date - 7 days
        if re.search(r'last\s+week', text, re.IGNORECASE):
            return document_date - timedelta(days=7)
        
        if re.search(r'last\s+month', text, re.IGNORECASE):
            return document_date - timedelta(days=30)
        
        if re.search(r'yesterday', text, re.IGNORECASE):
            return document_date - timedelta(days=1)
        
        # Pattern: explicit dates
        date_patterns = [
            r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(20\d{2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(0)
                    for fmt in ['%d %B %Y', '%B %d, %Y', '%B %d %Y']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except:
                            continue
                except:
                    pass
        
        return None


class ContextualRetriever:
    """
    Two-stage retrieval system based on Supermemory architecture.
    
    Stage 1: Search lightweight, high-signal memories
    Stage 2: Inject full source chunks for LLM context
    """
    
    def __init__(self, memory_generator: ContextualMemoryGenerator):
        self.generator = memory_generator
        self.memories: Dict[str, ContextualMemory] = {}
        self.chunks: Dict[str, str] = {}
        
    def index_conversation(
        self,
        conversation: List[Dict[str, Any]],
        document_date: datetime,
        session_summary: str = ""
    ):
        """
        Index an entire conversation.
        
        Args:
            conversation: List of conversation turns
            document_date: When this conversation took place
            session_summary: High-level summary
        """
        # Build context from session summary
        context = session_summary
        
        for i, turn in enumerate(conversation):
            message = turn.get('message', '')
            if not message:
                continue
            
            # Update context with recent history
            if i > 0:
                prev_messages = [t.get('message', '') for t in conversation[max(0, i-3):i]]
                context = ' | '.join(prev_messages[-2:])
            
            # Generate memories
            memories = self.generator.generate_memories(
                chunk=message,
                document_date=document_date,
                context=context,
                session_summary=session_summary
            )
            
            # Store memories and chunks
            for mem in memories:
                self.memories[mem.id] = mem
                self.chunks[mem.id] = message
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval.
        
        Stage 1: Search memories (lightweight, high signal)
        Stage 2: Return with full chunks
        """
        # Stage 1: Memory search (simple keyword matching for now)
        query_lower = query.lower()
        memory_scores = []
        
        for mem_id, mem in self.memories.items():
            score = 0.0
            
            # Keyword match in fact
            if query_lower in mem.fact.lower():
                score += 1.0
            
            # Keyword match in entities
            for entity in mem.entities or []:
                if entity.lower() in query_lower:
                    score += 0.5
            
            # Partial matches
            query_words = set(query_lower.split())
            fact_words = set(mem.fact.lower().split())
            overlap = len(query_words & fact_words)
            score += 0.3 * overlap / max(len(query_words), 1)
            
            if score > 0:
                memory_scores.append((mem_id, score))
        
        # Sort by score
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        top_memories = memory_scores[:top_k]
        
        # Stage 2: Inject chunks
        results = []
        for mem_id, score in top_memories:
            mem = self.memories[mem_id]
            chunk = self.chunks.get(mem_id, mem.source_chunk)
            
            results.append({
                'memory': mem.fact,
                'chunk': chunk,
                'context': mem.context_summary,
                'document_date': mem.document_date,
                'event_date': mem.event_date,
                'entities': mem.entities,
                'score': score,
                'confidence': mem.confidence
            })
        
        return results
