#!/usr/bin/env python3
"""
诊断检索和答案生成问题
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
from mimir_native import MimirMemory
from mimir_native.llm_client import BedrockClient

def diagnose():
    print("🔍 Diagnosing Retrieval & Answer Generation")
    
    # 初始化
    mimir = MimirMemory(db_path=':memory:')
    llm = BedrockClient()
    print("✅ Initialized\n")
    
    # 添加测试数据（模拟 LoCoMo 数据）
    test_memories = [
        "Caroline: I visited the LGBTQ support group today.",
        "Friend: How was it?",
        "Caroline: It was great, very supportive. The meeting was on 7 May 2023.",
        "Caroline: I'm planning to pursue psychology and counseling certification.",
        "Caroline: I researched adoption agencies recently.",
        "Caroline: As a transgender woman, I want to help others.",
    ]
    
    print("📥 Adding memories...")
    for m in test_memories:
        result = mimir.add_content(m, content_type='text', user_id='test')
        print(f"  -> {len(result)} memories")
    
    # 测试 Q1
    query = "When did Caroline go to the LGBTQ support group?"
    print(f"\n🔍 Query: {query}")
    
    # 获取检索结果
    results = mimir.query(query, user_id='test', top_k=5)
    print(f"\n📊 Retrieved {len(results)} results:")
    
    contexts = []
    for i, r in enumerate(results, 1):
        content = r.memory.content if hasattr(r, 'memory') else str(r)
        score = r.score if hasattr(r, 'score') else 0
        print(f"  [{i}] {content}")
        print(f"      Score: {score:.3f}")
        contexts.append(content)
    
    # 构建 prompt
    context_text = "\n".join(contexts)
    prompt = f"""Based on the following context, answer the question concisely.

Context:
{context_text}

Question: {query}

Answer:"""
    
    print(f"\n📝 Prompt (first 500 chars):")
    print(prompt[:500])
    
    # 生成答案
    print(f"\n🤖 Generating answer...")
    try:
        answer = llm.invoke_mistral(prompt)
        print(f"Answer: {answer}")
        
        # 分析问题
        print(f"\n📋 Analysis:")
        if "does not provide" in answer.lower() or "no information" in answer.lower():
            print("  ⚠️  LLM claims no info despite retrieval")
            print(f"  Context contains '7 May': {'7 May' in context_text}")
            print(f"  Context contains 'support group': {'support group' in context_text.lower()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnose()
