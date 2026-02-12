#!/usr/bin/env python3
"""
Test Supermemory SOTA Architecture Implementation

Tests:
1. Contextual Memory Generation
2. Dual Timestamp System
3. Relational Graph (updates/extends/derives)
4. Two-Stage Retrieval
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mimir_native import (
    ContextualMemoryGenerator,
    ContextualRetriever,
    RelationalMemoryGraph,
    RelationType
)


def test_contextual_memory():
    """Test contextual memory generation"""
    print("\n" + "="*60)
    print("Test 1: Contextual Memory Generation")
    print("="*60)
    
    generator = ContextualMemoryGenerator()
    
    # Test conversation chunk
    chunk = "Caroline adopted a cat named Whiskers from the local shelter yesterday."
    document_date = datetime(2023, 5, 15)  # When conversation happened
    
    memories = generator.generate_memories(
        chunk=chunk,
        document_date=document_date,
        context="Caroline has been looking for a pet",
        session_summary="Caroline gets a new pet"
    )
    
    print(f"\nInput chunk: {chunk}")
    print(f"Document date: {document_date.strftime('%Y-%m-%d')}")
    print(f"\nGenerated {len(memories)} memories:")
    
    for i, mem in enumerate(memories, 1):
        print(f"\n  Memory {i}:")
        print(f"    Fact: {mem.fact}")
        print(f"    Entities: {mem.entities}")
        print(f"    Document Date: {mem.document_date.strftime('%Y-%m-%d')}")
        if mem.event_date:
            print(f"    Event Date: {mem.event_date.strftime('%Y-%m-%d')}")
        print(f"    Confidence: {mem.confidence}")
    
    assert len(memories) > 0, "Should generate at least one memory"
    print("\n✅ Contextual Memory Generation: PASSED")


def test_dual_timestamp():
    """Test dual timestamp system"""
    print("\n" + "="*60)
    print("Test 2: Dual Timestamp System")
    print("="*60)
    
    generator = ContextualMemoryGenerator()
    
    # Test case: "yesterday" should resolve to document_date - 1 day
    chunk = "We went hiking yesterday in the mountains."
    document_date = datetime(2023, 5, 15)
    
    memories = generator.generate_memories(chunk, document_date)
    
    print(f"\nChunk: {chunk}")
    print(f"Document date (conversation date): {document_date.strftime('%Y-%m-%d')}")
    
    for mem in memories:
        if mem.event_date:
            print(f"Event date (when hiking happened): {mem.event_date.strftime('%Y-%m-%d')}")
            expected = document_date - timedelta(days=1)
            assert mem.event_date == expected, f"Expected {expected}, got {mem.event_date}"
            print(f"✅ Correctly resolved 'yesterday' to {mem.event_date.strftime('%Y-%m-%d')}")
    
    print("\n✅ Dual Timestamp System: PASSED")


def test_relational_graph():
    """Test relational memory graph"""
    print("\n" + "="*60)
    print("Test 3: Relational Memory Graph")
    print("="*60)
    
    graph = RelationalMemoryGraph()
    
    # Add initial memory
    mem1 = graph.add_memory("Caroline's favorite color is Blue")
    print(f"\nAdded memory 1: {mem1.content}")
    
    # Add update (contradiction)
    mem2 = graph.update_memory(mem1.id, "Caroline's favorite color is now Green")
    print(f"Added memory 2 (update): {mem2.content}")
    
    # Add extension
    mem3 = graph.add_memory("Caroline works at TechCorp as a Senior Engineer")
    mem4 = graph.add_memory("Caroline works at TechCorp")
    graph.add_relation(mem4.id, mem3.id, RelationType.EXTENDS)
    print(f"Added memory 3: {mem3.content}")
    print(f"Added memory 4 (extends #3): {mem4.content}")
    
    # Check version history
    history = graph.get_version_history(mem1.id)
    print(f"\nVersion history for '{mem1.content[:30]}...':")
    for node in history:
        print(f"  v{node.version}: {node.content}")
    
    assert len(history) == 2, "Should have 2 versions"
    
    # Check relations
    related = graph.get_related_memories(mem1.id)
    print(f"\nRelated memories to '{mem1.content[:30]}...':")
    for node, rel_type in related:
        print(f"  - {rel_type.value}: {node.content}")
    
    assert len(related) > 0, "Should have related memories"
    
    # Check stats
    stats = graph.stats()
    print(f"\nGraph stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Relational Memory Graph: PASSED")


def test_two_stage_retrieval():
    """Test two-stage retrieval system"""
    print("\n" + "="*60)
    print("Test 4: Two-Stage Retrieval")
    print("="*60)
    
    generator = ContextualMemoryGenerator()
    retriever = ContextualRetriever(generator)
    
    # Create test conversation
    conversation = [
        {'message': 'Caroline adopted a cat named Whiskers yesterday.'},
        {'message': 'Melanie went to a pottery class last week.'},
        {'message': 'They are planning to meet next weekend.'},
    ]
    document_date = datetime(2023, 5, 15)
    session_summary = "Friends catching up on recent events"
    
    # Index conversation
    retriever.index_conversation(conversation, document_date, session_summary)
    
    print(f"\nIndexed {len(conversation)} conversation turns")
    print(f"Generated {len(retriever.memories)} memories")
    
    # Test retrieval
    query = "When did Caroline get a cat?"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Memory (atomic fact): {result['memory']}")
        print(f"    Chunk (full context): {result['chunk'][:80]}...")
        print(f"    Score: {result['score']:.2f}")
        print(f"    Entities: {result['entities']}")
    
    assert len(results) > 0, "Should retrieve at least one result"
    
    # Verify we got the right memory
    found_caroline = any('Caroline' in r['memory'] for r in results)
    assert found_caroline, "Should retrieve memory about Caroline"
    
    print("\n✅ Two-Stage Retrieval: PASSED")


def test_knowledge_chain():
    """Test knowledge chain (fact evolution)"""
    print("\n" + "="*60)
    print("Test 5: Knowledge Chain (Fact Evolution)")
    print("="*60)
    
    graph = RelationalMemoryGraph()
    
    # Create a chain of updates
    mem1 = graph.add_memory("Caroline lives in New York")
    mem2 = graph.update_memory(mem1.id, "Caroline moved to Los Angeles")
    mem3 = graph.update_memory(mem2.id, "Caroline now lives in San Francisco")
    
    print(f"\nCreated knowledge chain:")
    print(f"  v1: {mem1.content}")
    print(f"  v2: {mem2.content}")
    print(f"  v3: {mem3.content}")
    
    # Get full chain
    chain = graph.get_knowledge_chain(mem1.id)
    
    print(f"\nFull knowledge chain ({len(chain)} versions):")
    for node in chain:
        print(f"  v{node.version}: {node.content}")
    
    assert len(chain) == 3, "Should have 3 versions in chain"
    
    # Get latest
    latest = graph.get_latest_version(mem1.id)
    print(f"\nLatest version: {latest.content}")
    assert latest.id == mem3.id, "Latest should be mem3"
    
    print("\n✅ Knowledge Chain: PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("SUPERMEMORY SOTA ARCHITECTURE TESTS")
    print("="*60)
    
    try:
        test_contextual_memory()
        test_dual_timestamp()
        test_relational_graph()
        test_two_stage_retrieval()
        test_knowledge_chain()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✅")
        print("="*60)
        print("\nSupermemory SOTA architecture components:")
        print("  ✅ Contextual Memory Generation")
        print("  ✅ Dual Timestamp System")
        print("  ✅ Relational Graph (updates/extends/derives)")
        print("  ✅ Two-Stage Retrieval")
        print("  ✅ Knowledge Chains")
        print("\nReady for LoCoMo full benchmark!")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
