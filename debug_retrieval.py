#!/usr/bin/env python3
"""
调试脚本 - 检查记忆提取和检索
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
from mimir_native import MimirMemory
from mimir_native.llm_client import BedrockClient

def debug():
    print("🔍 Debug Mode")
    
    # 初始化
    mimir = MimirMemory(db_path=':memory:')
    llm = BedrockClient()
    print(f"✅ Initialized")
    
    # 测试数据
    test_cases = [
        "Caroline visited the LGBTQ support group on 7 May 2023.",
        "Caroline is a transgender woman.",
        "Melanie painted a sunrise in 2022.",
    ]
    
    print("\n📥 Step 1: Adding memories...")
    for i, text in enumerate(test_cases):
        print(f"\n  [{i+1}] {text[:50]}...")
        
        # 检查事实提取
        print(f"      Calling extract_facts...")
        try:
            facts = llm.extract_facts(text, {})
            print(f"      Facts extracted: {len(facts)}")
            for f in facts:
                print(f"        - {f.get('fact', 'N/A')[:40]}...")
        except Exception as e:
            print(f"      Error: {e}")
        
        # 添加到 mimir
        memories = mimir.add_content(text, content_type='text', user_id='test')
        print(f"      Memories created: {len(memories) if isinstance(memories, list) else 'N/A'}")
    
    # 检查数据库中的记忆
    print("\n📊 Step 2: Database state...")
    all_memories = mimir.memory_agent.db.search_memories('', user_id='test', limit=100)
    print(f"   Total memories: {len(all_memories)}")
    for m in all_memories:
        print(f"   - {m.content[:50]}...")
    
    # 测试检索
    print("\n🔎 Step 3: Retrieval test...")
    query = "When did Caroline go to the LGBTQ support group?"
    print(f"   Query: {query}")
    
    # 直接向量搜索
    from mimir_native.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(mimir.db, llm)
    
    print(f"\n   Vector search (top_k=5):")
    vector_results = retriever._vector_search(query, 'test', 5)
    print(f"   Results: {len(vector_results)}")
    for r in vector_results:
        print(f"   - {r['memory'].content[:40]}... (score: {r['score']:.3f})")
    
    print(f"\n   FTS search (top_k=5):")
    fts_results = retriever._fts_search(query, 'test', 5)
    print(f"   Results: {len(fts_results)}")
    for r in fts_results:
        print(f"   - {r['memory'].content[:40]}... (score: {r['score']:.3f})")
    
    print(f"\n   Hybrid query (top_k=5):")
    hybrid_results = mimir.query(query, user_id='test', top_k=5)
    print(f"   Results: {len(hybrid_results)}")
    for r in hybrid_results:
        mem = r.memory if hasattr(r, 'memory') else r
        score = r.score if hasattr(r, 'score') else 0
        print(f"   - {mem.content[:40]}... (score: {score:.3f})")

if __name__ == "__main__":
    debug()
