"""
Mimir-Native: Powerful Memory System for AI Agents

A robust, extensible memory system supporting multiple content types,
hierarchical storage, hybrid retrieval, and relational knowledge graphs.

Based on:
- LoCoMo optimization achieving 86.1% F1
- Supermemory SOTA architecture (LongMemEval)
- MemGPT hierarchical memory system

Author: Arthur + AI Agents
Version: 2.1.0 (Supermemory Edition)
"""

__version__ = "2.1.0"
__author__ = "Arthur + AI Agents"

# Core components
from .core.memory_store import UnifiedMemoryStore, MemoryTier
from .core.memory_entry import MemoryEntry, ContentType
from .core.content_normalizer import ContentNormalizer
from .core.relational_graph import (
    RelationalMemoryGraph, 
    RelationType, 
    MemoryNode, 
    Relation
)

# Retrieval components
from .retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from .retrieval.vector_retriever import VectorRetriever
from .retrieval.bm25_retriever import BM25Retriever
from .retrieval.temporal_retriever import TemporalRetriever

# Processors
from .processors.temporal_normalizer import TemporalNormalizer
from .processors.query_analyzer import QueryAnalyzer, QueryIntent
from .processors.contextual_memory import (
    ContextualMemory,
    ContextualMemoryGenerator,
    ContextualRetriever
)

# Evaluation
from .evaluation.locomo_benchmark import LoCoMoBenchmark

__all__ = [
    # Core
    'UnifiedMemoryStore',
    'MemoryTier',
    'MemoryEntry',
    'ContentType',
    'ContentNormalizer',
    'RelationalMemoryGraph',
    'RelationType',
    'MemoryNode',
    'Relation',
    
    # Retrieval
    'HybridRetriever',
    'RetrievalResult',
    'VectorRetriever',
    'BM25Retriever',
    'TemporalRetriever',
    
    # Processors
    'TemporalNormalizer',
    'QueryAnalyzer',
    'QueryIntent',
    'ContextualMemory',
    'ContextualMemoryGenerator',
    'ContextualRetriever',
    
    # Evaluation
    'LoCoMoBenchmark',
]
