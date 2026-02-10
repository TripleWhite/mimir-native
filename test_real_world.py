#!/usr/bin/env python3
"""
Mimir-Native 真实能力验证测试

场景：模拟用户一周的真实使用，验证记忆召回能力
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
from datetime import datetime, timedelta
from mimir_native import MimirMemory
from mimir_native.llm_client import BedrockClient
from mimir_native.ingestion_pipeline import MimirIngestionPipeline


def create_test_data():
    """创建一周的模拟数据"""
    
    base_date = datetime(2026, 2, 3)  # 上周一
    
    test_data = {
        # Day 1: Claude 对话 - 项目规划
        "day1_claude": {
            "date": (base_date).strftime("%d %B %Y"),
            "source": "claude",
            "messages": [
                {"speaker": "User", "text": "我想做 Mimir 项目，核心是跨 AI 记忆共享"},
                {"speaker": "Claude", "text": "建议分为 3 个模块：存储层、检索层、接入层"},
                {"speaker": "User", "text": "技术栈用 SQLite + 向量索引"},
                {"speaker": "Claude", "text": "sqlite-vec 是个不错的选择，支持向量检索"},
                {"speaker": "User", "text": "目标用户是 AI 重度用户，需要管理大量对话"},
            ]
        },
        
        # Day 2: 收藏文章 - 技术调研
        "day2_article1": {
            "date": (base_date + timedelta(days=1)).strftime("%d %B %Y"),
            "source": "article",
            "content": """
            2024 年向量数据库对比：Pinecone vs Milvus vs 自研
            
            Pinecone: 托管服务，易用但贵
            Milvus: 开源，功能丰富但复杂
            自研 SQLite + sqlite-vec: 轻量，适合个人使用
            """
        },
        
        "day2_article2": {
            "date": (base_date + timedelta(days=1)).strftime("%d %B %Y"),
            "source": "article",
            "content": """
            Chrome Extension Manifest V3 开发指南
            
            - 使用 service worker 替代 background page
            - content script 注入页面
            - 权限模型更严格
            """
        },
        
        # Day 3: ChatGPT 对话 - 代码实现
        "day3_chatgpt": {
            "date": (base_date + timedelta(days=2)).strftime("%d %B %Y"),
            "source": "chatgpt",
            "messages": [
                {"speaker": "User", "text": "帮我写 SQLite 向量检索的代码"},
                {"speaker": "ChatGPT", "text": "可以用 sqlite-vec 扩展，先创建虚拟表..."},
                {"speaker": "User", "text": "需要支持 metadata 过滤"},
                {"speaker": "ChatGPT", "text": "可以在向量检索后用 SQL WHERE 子句过滤"},
                {"speaker": "User", "text": "还要支持混合检索，向量 + BM25"},
            ]
        },
        
        # Day 5: 笔记 - 问题记录
        "day5_note1": {
            "date": (base_date + timedelta(days=4)).strftime("%d %B %Y"),
            "source": "note",
            "content": "LoCoMo 测试 F1 只有 10%，问题在时序解析。需要修复日期格式 '1:56 pm on 8 May' 的解析逻辑。"
        },
        
        "day5_note2": {
            "date": (base_date + timedelta(days=4)).strftime("%d %B %Y"),
            "source": "note",
            "content": "优化方案：1. 日期格式标准化 2. LLM Prompt 优化 3. 答案后处理。目标 F1 提升到 20%。"
        },
        
        # Day 7: Claude 对话 - 项目回顾（关键测试）
        "day7_claude": {
            "date": (base_date + timedelta(days=6)).strftime("%d %B %Y"),
            "source": "claude",
            "messages": [
                {"speaker": "User", "text": "我上周规划的项目进展如何？"},  # 关键问题！
                {"speaker": "Claude", "text": "哪个项目？"},  # 应该能关联到 day1
            ]
        }
    }
    
    return test_data


def ingest_all_data(mimir, pipeline, test_data, user_id='test_user'):
    """摄入所有测试数据"""
    
    print("📝 摄入测试数据...")
    
    for key, data in test_data.items():
        if 'messages' in data:
            # 对话数据
            result = pipeline.ingest_conversation(
                messages=data['messages'],
                session_date=data['date'],
                source_type=data['source'],
                user_id=user_id
            )
            print(f"  {key}: {result['memories_created']} memories")
        else:
            # 文章/笔记
            from mimir_native.content_processor import ContentProcessor
            processor = ContentProcessor()
            
            # 简单处理为段落
            memories = processor.process_conversation(
                messages=[{'speaker': 'Author', 'text': data['content']}],
                session_date=data['date'],
                source_type=data['source']
            )
            
            # 存储
            for mem in memories:
                try:
                    from mimir_native.database import MemoryCreate
                    import hashlib
                    content_hash = hashlib.md5(mem['content'].lower().strip().encode()).hexdigest()
                    
                    mem_create = MemoryCreate(
                        user_id=user_id,
                        content=mem['content'],
                        content_hash=content_hash,
                        embedding=mimir.llm.embed(mem['content']),
                        source_type=data['source'],
                        source_metadata=json.dumps({'date': data['date']})
                    )
                    mimir.memory_agent.db.create_memory(mem_create)
                except Exception as e:
                    print(f"    Error: {e}")
            
            print(f"  {key}: {len(memories)} memories")


def run_retrieval_tests(mimir, user_id='test_user'):
    """运行检索测试"""
    
    print("\n🔍 检索测试\n")
    
    test_queries = [
        {
            'query': '我上周规划的项目',  # 应该关联 day1 的 Mimir 项目
            'expected_keywords': ['Mimir', '项目', '存储层', 'SQLite'],
            'test_type': '跨时间关联'
        },
        {
            'query': '向量检索的代码实现',  # 应该关联 day2 文章 + day3 代码
            'expected_keywords': ['sqlite-vec', 'metadata', 'BM25'],
            'test_type': '跨平台关联'
        },
        {
            'query': 'LoCoMo 测试的问题',  # 应该找到 day5 的笔记
            'expected_keywords': ['时序解析', '10%', '日期格式'],
            'test_type': '笔记检索'
        },
        {
            'query': 'Chrome Extension 开发',  # 应该找到 day2 的文章
            'expected_keywords': ['Manifest V3', 'service worker'],
            'test_type': '文章检索'
        },
        {
            'query': '上周一我和 Claude 讨论了什么',  # 精确时间检索
            'expected_keywords': ['Mimir', '模块', 'SQLite'],
            'test_type': '精确时间+平台'
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"Test {i}: {test['test_type']}")
        print(f"  Query: {test['query']}")
        
        try:
            # 检索
            contexts = mimir.query(test['query'], user_id=user_id, top_k=3)
            context_text = "\n".join([str(c.memory.content if hasattr(c, 'memory') else c) for c in contexts])
            
            # 检查关键词
            found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in context_text.lower()]
            recall = len(found_keywords) / len(test['expected_keywords'])
            
            print(f"  Expected: {test['expected_keywords']}")
            print(f"  Found: {found_keywords}")
            print(f"  Recall: {recall:.2%}")
            print()
            
            results.append({
                'test_type': test['test_type'],
                'recall': recall,
                'found': found_keywords,
                'expected': test['expected_keywords']
            })
            
        except Exception as e:
            print(f"  Error: {e}\n")
            results.append({'test_type': test['test_type'], 'recall': 0, 'error': str(e)})
    
    return results


def evaluate_results(results):
    """评估结果"""
    
    print("=" * 60)
    print("📊 评估结果")
    print("=" * 60)
    
    avg_recall = sum(r['recall'] for r in results) / len(results)
    
    print(f"\n平均召回率: {avg_recall:.2%}")
    print(f"测试项: {len(results)}")
    
    print("\n详细结果:")
    for r in results:
        status = "✅" if r['recall'] >= 0.8 else ("⚠️" if r['recall'] >= 0.5 else "❌")
        print(f"  {status} {r['test_type']}: {r['recall']:.2%}")
    
    # 评估标准
    print("\n评估标准:")
    if avg_recall >= 0.8:
        print("  🟢 优秀 - 记忆层可靠，可以开始插件开发")
    elif avg_recall >= 0.6:
        print("  🟡 良好 - 基本可用，但有优化空间")
    else:
        print("  🔴 需改进 - 记忆层不稳定，需要修复")
    
    return avg_recall >= 0.6  # 及格线


def main():
    """主测试函数"""
    
    print("=" * 60)
    print("Mimir-Native 真实能力验证测试")
    print("=" * 60)
    print()
    
    # 初始化
    mimir = MimirMemory(db_path=':memory:')
    llm = BedrockClient()
    pipeline = MimirIngestionPipeline(
        mimir_db=mimir.memory_agent.db,
        llm_client=llm,
        embedder=llm
    )
    
    # 创建测试数据
    test_data = create_test_data()
    
    # 摄入数据
    ingest_all_data(mimir, pipeline, test_data)
    
    # 运行检索测试
    results = run_retrieval_tests(mimir)
    
    # 评估
    passed = evaluate_results(results)
    
    print("\n" + "=" * 60)
    if passed:
        print("✅ 测试通过 - 可以开始 Chrome Extension 开发")
    else:
        print("⚠️ 测试未通过 - 需要先优化记忆层")
    print("=" * 60)


if __name__ == "__main__":
    main()
