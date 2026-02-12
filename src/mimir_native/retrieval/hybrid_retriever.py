"""
Hybrid Retriever - Multi-Strategy Memory Retrieval

Combines vector search, BM25, temporal filtering, and evidence-based retrieval
with RRF fusion for optimal results.

Based on LoCoMo optimization achieving 86.1% F1 on When questions.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime

class RetrievalStrategy(Enum):
    """Retrieval strategies"""
    VECTOR = "vector"
    BM25 = "bm25"
    TEMPORAL = "temporal"
    EVIDENCE = "evidence"
    HYBRID = "hybrid"

@dataclass
class RetrievalResult:
    """Single retrieval result"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None

class HybridRetriever:
    """
    Hybrid retriever combining multiple strategies.
    
    Features:
    - Vector semantic search
    - BM25 keyword search
    - Temporal filtering
    - Evidence-based retrieval (LoCoMo optimized)
    - RRF fusion
    """
    
    def __init__(
        self,
        vector_weight: float = 0.4,
        bm25_weight: float = 0.3,
        temporal_weight: float = 0.2,
        evidence_weight: float = 0.1,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_weight: Weight for vector similarity
            bm25_weight: Weight for BM25 scores
            temporal_weight: Weight for temporal relevance
            evidence_weight: Weight for evidence matching
            rrf_k: RRF fusion parameter
        """
        self.weights = {
            'vector': vector_weight,
            'bm25': bm25_weight,
            'temporal': temporal_weight,
            'evidence': evidence_weight
        }
        self.rrf_k = rrf_k
        
        # Storage
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.session_dates: Dict[str, datetime] = {}
        
    def index(
        self,
        documents: List[str],
        embeddings: Optional[List[np.ndarray]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        session_dates: Optional[Dict[str, datetime]] = None
    ):
        """
        Index documents for retrieval.
        
        Args:
            documents: List of document texts
            embeddings: Optional pre-computed embeddings
            metadata: Optional metadata for each document
            session_dates: Optional mapping of session_id -> date
        """
        self.documents = documents
        self.embeddings = embeddings if embeddings is not None else []
        self.metadata = metadata if metadata is not None else [{} for _ in documents]
        self.session_dates = session_dates or {}
        
        # Build BM25 index if needed
        self._build_bm25_index()
        
    def _build_bm25_index(self):
        """Build BM25 index for keyword search"""
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize documents
            tokenized = [doc.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized)
        except ImportError:
            self.bm25 = None
            
    def retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        evidence_sessions: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents using hybrid approach.
        
        Args:
            query: Search query
            query_embedding: Optional pre-computed query embedding
            evidence_sessions: Optional list of evidence session IDs
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        results = []
        
        # Strategy 1: Evidence-based retrieval (highest priority)
        if evidence_sessions:
            evidence_results = self._evidence_retrieve(query, evidence_sessions)
            results.extend(evidence_results)
            
        # Strategy 2: Vector search
        if query_embedding is not None and self.embeddings:
            vector_results = self._vector_retrieve(query_embedding)
            results.extend(vector_results)
            
        # Strategy 3: BM25 keyword search
        if self.bm25:
            bm25_results = self._bm25_retrieve(query)
            results.extend(bm25_results)
            
        # Strategy 4: Temporal retrieval
        temporal_results = self._temporal_retrieve(query)
        results.extend(temporal_results)
        
        # RRF Fusion
        fused_results = self._rrf_fusion(results, top_k)
        
        return fused_results
    
    def _evidence_retrieve(
        self,
        query: str,
        evidence_sessions: List[str]
    ) -> List[RetrievalResult]:
        """Evidence-based retrieval using session IDs"""
        results = []
        
        for i, meta in enumerate(self.metadata):
            session = meta.get('session', '')
            if any(ev in session for ev in evidence_sessions):
                results.append(RetrievalResult(
                    content=self.documents[i],
                    score=1.0,  # High priority for evidence
                    source='evidence',
                    metadata=meta,
                    timestamp=self.session_dates.get(session)
                ))
                
        return results
    
    def _vector_retrieve(self, query_embedding: np.ndarray) -> List[RetrievalResult]:
        """Vector similarity retrieval"""
        if not self.embeddings:
            return []
            
        # Compute similarities
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append(sim)
            
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:20]
        
        results = []
        for idx in top_indices:
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            session = meta.get('session', '')
            results.append(RetrievalResult(
                content=self.documents[idx],
                score=float(similarities[idx]),
                source='vector',
                metadata=meta,
                timestamp=self.session_dates.get(session)
            ))
            
        return results
    
    def _bm25_retrieve(self, query: str) -> List[RetrievalResult]:
        """BM25 keyword retrieval"""
        if not self.bm25:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:20]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                meta = self.metadata[idx] if idx < len(self.metadata) else {}
                session = meta.get('session', '')
                results.append(RetrievalResult(
                    content=self.documents[idx],
                    score=float(scores[idx]),
                    source='bm25',
                    metadata=meta,
                    timestamp=self.session_dates.get(session)
                ))
                
        return results
    
    def _temporal_retrieve(self, query: str) -> List[RetrievalResult]:
        """Temporal retrieval based on query time expressions"""
        # Extract temporal expressions from query
        time_patterns = [
            r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})',
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})[,
\s]+(20\d{2})',
        ]
        
        target_date = None
        for pattern in time_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Parse date
                try:
                    date_str = match.group(0)
                    # Try different formats
                    for fmt in ['%d %B %Y', '%B %d, %Y', '%B %d %Y']:
                        try:
                            target_date = datetime.strptime(date_str, fmt)
                            break
                        except:
                            continue
                except:
                    pass
                
        if not target_date:
            return []
            
        # Find documents close to target date
        results = []
        for i, meta in enumerate(self.metadata):
            session = meta.get('session', '')
            doc_date = self.session_dates.get(session)
            
            if doc_date:
                # Calculate temporal relevance
                days_diff = abs((doc_date - target_date).days)
                score = max(0, 1.0 - (days_diff / 30.0))  # Decay over 30 days
                
                if score > 0:
                    results.append(RetrievalResult(
                        content=self.documents[i],
                        score=score,
                        source='temporal',
                        metadata=meta,
                        timestamp=doc_date
                    ))
                    
        return results
    
    def _rrf_fusion(
        self,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion of multiple retrieval results.
        
        RRF formula: score = sum(1 / (k + rank)) for each list
        """
        # Group by source
        by_source: Dict[str, List[RetrievalResult]] = {}
        for r in results:
            if r.source not in by_source:
                by_source[r.source] = []
            by_source[r.source].append(r)
            
        # Calculate RRF scores
        content_scores: Dict[str, Dict[str, Any]] = {}
        
        for source, source_results in by_source.items():
            weight = self.weights.get(source, 0.1)
            
            for rank, result in enumerate(source_results):
                if result.content not in content_scores:
                    content_scores[result.content] = {
                        'score': 0,
                        'sources': [],
                        'metadata': result.metadata,
                        'timestamp': result.timestamp
                    }
                    
                # RRF score
                rrf_score = weight * (1.0 / (self.rrf_k + rank))
                content_scores[result.content]['score'] += rrf_score
                content_scores[result.content]['sources'].append(source)
                
        # Sort by fused score
        sorted_results = sorted(
            content_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:top_k]
        
        # Create final results
        fused = []
        for content, data in sorted_results:
            fused.append(RetrievalResult(
                content=content,
                score=data['score'],
                source='+'.join(set(data['sources'])),
                metadata=data['metadata'],
                timestamp=data['timestamp']
            ))
            
        return fused
