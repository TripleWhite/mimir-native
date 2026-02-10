#!/usr/bin/env python3
"""
检查存储的记忆内容是否包含绝对日期
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mimir_native import MimirMemory
from mimir_native.llm_client import BedrockClient
from mimir_native.ingestion_pipeline import MimirIngestionPipeline
import json

def check_stored_memories():
    print("🔍 检查存储的记忆内容")
    print("=" * 60)
    
    mimir = MimirMemory(db_path=':memory:')
    llm = BedrockClient()
    
    pipeline = MimirIngestionPipeline(
        mimir_db=mimir.memory_agent.db,
        llm_client=llm,
        embedder=llm
    )
    
    # 测试数据
    messages = [
        {'speaker': 'Caroline', 'text': 'I visited the LGBTQ support group yesterday.'},
        {'speaker': 'Caroline', 'text': 'Melanie painted a sunrise last year.'},
    ]
    
    # 摄入
    result = pipeline.ingest_conversation(
        messages=messages,
        session_date='1:56 pm on 8 May, 2023',
        user_id='test'
    )
    
    print(f"摄入了 {result['memories_created']} 条记忆\n")
    
    # 检查存储的记忆
    cursor = mimir.memory_agent.db._execute(
        "SELECT content FROM memories WHERE user_id = ?",
        ('test',)
    )
    rows = cursor.fetchall()
    
    print("存储的记忆内容:")
    for i, row in enumerate(rows, 1):
        content = row['content']
        has_absolute_date = any(x in content for x in ['2023', '2022', 'May 2023'])
        has_relative_time = any(x in content.lower() for x in ['yesterday', 'last year', 'today'])
        
        status = "✅ 有绝对日期" if has_absolute_date else ("⚠️  有相对时间" if has_relative_time else "❓ 无时间")
        print(f"{i}. {status}")
        print(f"   内容: {content}")
        print()
    
    # 测试检索
    print("=" * 60)
    print("测试检索:")
    print("=" * 60)
    
    query = "When did Caroline go to the LGBTQ support group?"
    print(f"\n查询: {query}")
    
    contexts = mimir.query(query, user_id='test', top_k=3)
    print(f"\n检索到 {len(contexts)} 条结果:")
    
    for i, ctx in enumerate(contexts, 1):
        content = str(ctx.memory.content if hasattr(ctx, 'memory') else ctx)
        print(f"{i}. {content[:100]}...")
        
        # 检查是否包含绝对日期
        if any(x in content for x in ['2023', '2022']):
            print("   ✅ 包含绝对日期")
        elif any(x in content.lower() for x in ['yesterday', 'last year']):
            print("   ❌ 包含相对时间")

if __name__ == "__main__":
    check_stored_memories()
