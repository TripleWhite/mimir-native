#!/usr/bin/env python3
"""
验证 V2 + 强制时间替换
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mimir_native import MimirMemory
from mimir_native.llm_client import BedrockClient
from mimir_native.batch_processor_v2 import BatchProcessor

def test_v2_with_post_processing():
    print("🧪 Test V2 + Temporal Post-Processing")
    print("=" * 60)
    
    llm = BedrockClient()
    mimir = MimirMemory(db_path=':memory:')
    processor = BatchProcessor(mimir, llm)
    
    # 测试数据：不同 session 日期
    conversations = [
        {
            'session_date': '8 May 2023',
            'messages': [
                {'speaker': 'Caroline', 'text': 'I visited the LGBTQ support group yesterday.'},
                {'speaker': 'Caroline', 'text': 'Melanie painted a sunrise last year.'},
            ]
        },
        {
            'session_date': '25 May 2023',
            'messages': [
                {'speaker': 'Melanie', 'text': 'I ran a charity race last Sunday.'},
                {'speaker': 'Caroline', 'text': 'I will give a speech at school next week.'},
            ]
        }
    ]
    
    result = processor.process_conversations_batch(
        conversations,
        user_id='test',
        batch_size=10
    )
    
    print(f"\nCreated {result['memories']} memories\n")
    
    # 检查记忆内容
    cursor = mimir.memory_agent.db._execute(
        "SELECT content FROM memories WHERE user_id = ?",
        ('test',)
    )
    rows = cursor.fetchall()
    
    passed = 0
    for row in rows:
        content = row['content']  # 使用字典访问
        print(f"  Memory: {content[:100]}...")
        
        # 检查是否包含绝对日期
        has_date = any(x in content for x in ['2022', '2023', 'May 2023'])
        if has_date:
            print(f"    ✅ Contains absolute date")
            passed += 1
        else:
            print(f"    ⚠️  No absolute date found")
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(rows)} memories have absolute dates")
    
    return passed >= 2

if __name__ == "__main__":
    success = test_v2_with_post_processing()
    print(f"\n{'🎉 PASS' if success else '❌ FAIL'} - Temporal post-processing works!" if success else "")
