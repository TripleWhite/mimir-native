#!/usr/bin/env python3
"""
简化测试 - 验证记忆创建和检索
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mimir_native import MimirMemory

def test():
    print("🧪 Simple Memory Test")
    
    # 创建内存数据库
    mimir = MimirMemory(db_path=':memory:')
    print("✅ Initialized")
    
    # 添加测试数据
    test_data = [
        "Caroline visited the LGBTQ support group on 7 May 2023.",
        "Caroline is a transgender woman.",
        "Melanie painted a sunrise in 2022.",
    ]
    
    print("\n📥 Adding memories...")
    for text in test_data:
        memories = mimir.add_content(text, content_type='text', user_id='test')
        print(f"  '{text[:40]}...' -> {len(memories)} memories")
    
    # 查询
    print("\n🔍 Querying...")
    query = "When did Caroline go to the LGBTQ support group?"
    print(f"Query: {query}")
    
    results = mimir.query(query, user_id='test', top_k=5)
    print(f"Results: {len(results)}")
    
    for i, r in enumerate(results, 1):
        content = r.memory.content if hasattr(r, 'memory') else str(r)
        score = r.score if hasattr(r, 'score') else 0
        print(f"  [{i}] {content[:50]}... (score: {score:.3f})")
    
    # 检查是否包含正确答案
    found_date = any('7 May' in str(r.memory.content if hasattr(r, 'memory') else r) for r in results)
    print(f"\n✅ Contains '7 May': {found_date}")
    
    return found_date

if __name__ == "__main__":
    success = test()
    sys.exit(0 if success else 1)
