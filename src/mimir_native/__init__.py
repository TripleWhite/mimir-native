"""
Mimir-Native: Powerful Memory System for AI Agents

A robust, extensible memory system supporting multiple content types,
hierarchical storage, and hybrid retrieval.

Author: Arthur + AI Agents
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Arthur + AI Agents"

# Core components
from .core.memory_store import UnifiedMemoryStore, MemoryTier
from .core.memory_entry import MemoryEntry, ContentType
from .core.content_normalizer import ContentNormalizer

# Retrieval components
from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.vector_retriever import VectorRetriever
from .retrieval.bm25_retriever import BM25Retriever
from .retrieval.temporal_retriever import TemporalRetriever

# Processors
from .processors.temporal_normalizer import TemporalNormalizer
from .processors.query_analyzer import QueryAnalyzer, QueryIntent

# Evaluation
from .evaluation.locomo_benchmark import LoCoMoBenchmark

__all__ = [
    # Core
    'UnifiedMemoryStore',
    'MemoryTier',
    'MemoryEntry',
    'ContentType',
    'ContentNormalizer',
    
    # Retrieval
    'HybridRetriever',
    'VectorRetriever',
    'BM25Retriever',
    'TemporalRetriever',
    
    # Processors
    'TemporalNormalizer',
    'QueryAnalyzer',
    'QueryIntent',
    
    # Evaluation
    'LoCoMoBenchmark',
]
