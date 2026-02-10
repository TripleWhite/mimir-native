#!/usr/bin/env python3
"""
Mimir-Native 快速验证 - 只测2题
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
import tempfile
from mimir_native import MimirMemory
from mimir_native.llm_client import BedrockClient

def test_2q():
    # 初始化
    print("🚀 Quick Test (2 questions)")
    mimir = MimirMemory(db_path=':memory:')
    llm = BedrockClient()
    print("✅ Initialized")
    
    # 手动添加测试数据
    print("\n📥 Adding test memories...")
    test_memories = [
        "Caroline visited the LGBTQ support group on 7 May 2023.",
        "Caroline is a transgender woman.",
        "Melanie painted a sunrise in 2022.",
    ]
    
    for mem in test_memories:
        mimir.add_content(mem, content_type='text', user_id='test')
    print(f"✅ Added {len(test_memories)} memories")
    
    # 测试2题
    questions = [
        ("When did Caroline go to the LGBTQ support group?", "7 May 2023"),
        ("What is Caroline's identity?", "Transgender woman"),
    ]
    
    print("\n🎯 Testing...")
    for q, expected in questions:
        print(f"\nQ: {q}")
        
        # 检索
        results = mimir.query(q, user_id='test', top_k=3)
        print(f"   Retrieved: {len(results)} results")
        
        # 生成答案
        context = "\n".join([str(r) for r in results])[:3000]
        prompt = f"""Based on context, answer concisely:
Context: {context}
Question: {q}
Answer:"""
        
        try:
            answer = llm.invoke_mistral(prompt)
            print(f"   Answer: {answer[:60]}...")
            print(f"   Expected: {expected}")
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    test_2q()
