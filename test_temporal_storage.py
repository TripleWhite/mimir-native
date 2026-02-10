#!/usr/bin/env python3
"""
验证时序标准化在存储层的生效情况
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mimir_native.content_processor import ContentProcessor, TemporalNormalizer
from mimir_native.llm_client import BedrockClient

def test_temporal_normalization():
    """测试完整的时序标准化流程"""
    print("🧪 测试时序标准化流程")
    print("=" * 60)
    
    llm = BedrockClient()
    processor = ContentProcessor(llm)
    
    # 测试数据
    messages = [
        {'speaker': 'Caroline', 'text': 'I visited the LGBTQ support group yesterday.'},
        {'speaker': 'Caroline', 'text': 'Melanie painted a sunrise last year.'},
        {'speaker': 'Caroline', 'text': 'We are planning to go camping next week.'},
    ]
    session_date = '8 May 2023'
    
    print(f"\nSession date: {session_date}")
    print("\n原始消息:")
    for msg in messages:
        print(f"  {msg['speaker']}: {msg['text']}")
    
    # 处理对话
    print("\n处理中...")
    memories = processor.process_conversation(messages, session_date)
    
    print(f"\n生成了 {len(memories)} 条记忆:\n")
    
    passed = 0
    for i, mem in enumerate(memories, 1):
        content = mem['content']
        has_date = any(x in content for x in ['2023', '2022', 'May 2023', 'June 2023'])
        
        status = "✅" if has_date else "❌"
        print(f"{i}. {status} {content[:100]}...")
        
        if has_date:
            passed += 1
        else:
            print(f"   ⚠️  未找到绝对日期")
    
    print(f"\n{'=' * 60}")
    print(f"结果: {passed}/{len(memories)} 条记忆包含绝对日期")
    
    # 测试 TemporalNormalizer 单独工作
    print("\n" + "=" * 60)
    print("直接测试 TemporalNormalizer:")
    print("=" * 60)
    
    tn = TemporalNormalizer()
    test_cases = [
        ("I visited the group yesterday.", "8 May 2023"),
        ("Melanie painted a sunrise last year.", "8 May 2023"),
        ("We are going next week.", "8 May 2023"),
    ]
    
    for text, ref_date in test_cases:
        result = tn.normalize(text, ref_date)
        has_date = any(x in result for x in ['2023', '2022', 'May', 'June'])
        status = "✅" if has_date else "❌"
        print(f"\n{status} 输入: {text}")
        print(f"   参考: {ref_date}")
        print(f"   输出: {result}")

if __name__ == "__main__":
    test_temporal_normalization()
