#!/usr/bin/env python3
"""
验证 LoCoMo 第一题
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
from mimir_native import MimirMemory
from mimir_native.llm_client import BedrockClient

def test_q1():
    print("🧪 LoCoMo Q1 Test")
    
    # 初始化
    mimir = MimirMemory(db_path=':memory:')
    llm = BedrockClient()
    print("✅ Initialized")
    
    # 手动添加关键记忆
    memories = [
        "Caroline visited the LGBTQ support group on 7 May 2023.",
        "Caroline said she visited the LGBTQ support group.",
        "The group meeting was on 7 May 2023.",
    ]
    
    print("\n📥 Adding memories...")
    for m in memories:
        result = mimir.add_content(m, content_type='text', user_id='test')
        print(f"  -> {len(result)} memories")
    
    # 查询
    query = "When did Caroline go to the LGBTQ support group?"
    print(f"\n🔍 Query: {query}")
    
    results = mimir.query(query, user_id='test', top_k=5)
    print(f"Retrieved: {len(results)} results")
    
    contexts = []
    for r in results:
        content = r.memory.content if hasattr(r, 'memory') else str(r)
        score = r.score if hasattr(r, 'score') else 0
        print(f"  - {content[:60]}... (score: {score:.3f})")
        contexts.append(content)
    
    # 生成答案
    context_text = "\n".join(contexts)[:3000]
    prompt = f"""Based on the context, answer the question concisely.

Context:
{context_text}

Question: {query}

Answer:"""
    
    print("\n📝 Generating answer...")
    try:
        answer = llm.invoke_mistral(prompt)
        print(f"Answer: {answer}")
        
        # 检查是否包含正确答案
        has_date = '7 May' in answer or 'May 7' in answer or '2023' in answer
        print(f"\n✅ Contains date: {has_date}")
        
        return has_date
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_q1()
    sys.exit(0 if success else 1)
