#!/usr/bin/env python3
"""
测试 LLM-Native 记忆提取
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mimir_native.llm_client import BedrockClient
from mimir_native.llm_native_memory import LLMNativeMemoryExtractor

def test():
    print("🧪 Testing LLM-Native Memory Extraction")
    
    llm = BedrockClient()
    extractor = LLMNativeMemoryExtractor(llm)
    
    # LoCoMo 测试对话
    conversation = [
        {"speaker": "Caroline", "text": "I visited the LGBTQ support group yesterday."},
        {"speaker": "Friend", "text": "How was it?"},
        {"speaker": "Caroline", "text": "It was great, very supportive. The meeting was on 7 May 2023."},
        {"speaker": "Caroline", "text": "As a transgender woman, I want to help others."},
        {"speaker": "Melanie", "text": "I painted a sunrise last year."},
    ]
    
    print("\n对话内容:")
    for msg in conversation:
        print(f"  {msg['speaker']}: {msg['text']}")
    
    print("\n⏳ 提取记忆中...")
    memories = extractor.extract_memories(conversation, "8 May 2023")
    
    print(f"\n✅ 提取了 {len(memories)} 条记忆:\n")
    for i, m in enumerate(memories, 1):
        print(f"[{i}] {m.content}")
        print(f"    类型: {m.memory_type}")
        print(f"    实体: {m.entities}")
        print(f"    时间: {m.temporal_info}")
        print(f"    置信度: {m.confidence}")
        print()

if __name__ == "__main__":
    test()
