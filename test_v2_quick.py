#!/usr/bin/env python3
"""
快速验证 V2 的核心改进：时间转换 + 属性提取
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mimir_native.llm_client import BedrockClient
from mimir_native.batch_processor_v2 import BatchProcessor
from mimir_native import MimirMemory

def test_time_conversion():
    """测试时间转换是否生效"""
    print("🧪 Test 1: Time Conversion")
    print("-" * 60)
    
    llm = BedrockClient()
    mimir = MimirMemory(db_path=':memory:')
    processor = BatchProcessor(mimir, llm)
    
    # 测试数据
    conversations = [{
        'session_date': '8 May 2023',
        'messages': [
            {'speaker': 'Caroline', 'text': 'I visited the LGBTQ support group yesterday.'},
            {'speaker': 'Caroline', 'text': 'Melanie painted a sunrise last year.'},
        ]
    }]
    
    result = processor.process_conversations_batch(
        conversations, 
        user_id='test',
        batch_size=10
    )
    
    print(f"Created {result['memories']} memories")
    
    # 检查记忆内容
    # 使用正确的方法获取记忆
    from mimir_native.database import Memory
    cursor = mimir.memory_agent.db._execute(
        "SELECT * FROM memories WHERE user_id = ? LIMIT 10",
        ('test',)
    )
    rows = cursor.fetchall()
    
    memories = [mimir.memory_agent.db._row_to_memory(row) for row in rows]
    
    found_date = False
    for m in memories:
        content = m.content
        print(f"  Memory: {content[:80]}...")
        if '7 May 2023' in content or '2022' in content or '2023-05-07' in content:
            found_date = True
            print(f"    ✅ Contains converted date!")
    
    return found_date

def test_attribute_extraction():
    """测试属性提取是否生效"""
    print("\n🧪 Test 2: Attribute Extraction")
    print("-" * 60)
    
    llm = BedrockClient()
    mimir = MimirMemory(db_path=':memory:')
    processor = BatchProcessor(mimir, llm)
    
    conversations = [{
        'session_date': '8 May 2023',
        'messages': [
            {'speaker': 'Caroline', 'text': 'I am a transgender woman.'},
            {'speaker': 'Melanie', 'text': 'I am single and happy.'},
        ]
    }]
    
    result = processor.process_conversations_batch(
        conversations,
        user_id='test2',
        batch_size=10
    )
    
    print(f"Created {result['memories']} memories")
    
    # 检索身份和关系状态
    cursor = mimir.memory_agent.db._execute(
        "SELECT * FROM memories WHERE user_id = ? LIMIT 10",
        ('test2',)
    )
    rows = cursor.fetchall()
    memories = [mimir.memory_agent.db._row_to_memory(row) for row in rows]
    
    found_identity = False
    found_relationship = False
    
    for m in memories:
        content = m.content.lower()
        print(f"  Memory: {m.content[:80]}...")
        if 'transgender' in content:
            found_identity = True
            print(f"    ✅ Contains identity!")
        if 'single' in content:
            found_relationship = True
            print(f"    ✅ Contains relationship status!")
    
    return found_identity and found_relationship

if __name__ == "__main__":
    print("=" * 60)
    print("Batch Processor V2 - Quick Verification")
    print("=" * 60)
    
    test1_pass = test_time_conversion()
    test2_pass = test_attribute_extraction()
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Time Conversion: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Attribute Extraction: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 All core fixes verified!")
    else:
        print("\n⚠️  Some issues remain")
