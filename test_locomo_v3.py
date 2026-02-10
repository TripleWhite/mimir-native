#!/usr/bin/env python3
"""
使用 IngestionPipeline V3 测试 LoCoMo 10 题
关键改进：
1. LLM 智能提取事实
2. 强制时序标准化（yesterday → 7 May 2023）
3. 正确的 session_date 绑定
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
from mimir_native import MimirMemory
from mimir_native.llm_client import BedrockClient
from mimir_native.ingestion_pipeline import MimirIngestionPipeline

def calculate_f1(prediction, ground_truth) -> float:
    prediction = str(prediction) if prediction else ""
    ground_truth = str(ground_truth) if ground_truth else ""
    pred_tokens = set(prediction.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    common = pred_tokens & truth_tokens
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def test_locomo_v3():
    print("🚀 LoCoMo 10Q Test (Ingestion Pipeline V3)")
    print("=" * 60)
    print("关键改进：LLM 提取 + 强制时序标准化")
    print("=" * 60)
    
    mimir = MimirMemory(db_path=':memory:')
    llm = BedrockClient()
    
    # 初始化 ingestion pipeline
    pipeline = MimirIngestionPipeline(
        mimir_db=mimir.memory_agent.db,
        llm_client=llm,
        embedder=llm
    )
    
    # 加载数据
    data_path = '/Users/Zhuanz/.openclaw/workspace/mimir-locomo-testbed/data/locomo10.json'
    print(f"\n1. Loading LoCoMo data...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    conv = data[0]
    qa_list = conv.get('qa', [])[:10]
    print(f"   ✅ Loaded {len(qa_list)} questions")
    
    # 使用新的 ingestion pipeline
    print("\n2. Ingesting conversation (V3 Pipeline)...")
    conversation = conv.get('conversation', conv)
    
    result = pipeline.ingest_locomo_conversation(
        conversation=conversation,
        user_id='locomo_test'
    )
    
    print(f"   ✅ Ingested {result['total_messages']} messages → {result['total_memories']} memories")
    
    # 验证存储的记忆（检查时序标准化）
    print("\n3. Sample memories (checking temporal normalization):")
    cursor = mimir.memory_agent.db._execute(
        "SELECT content FROM memories WHERE user_id = ? AND content LIKE '%May 2023%' LIMIT 3",
        ('locomo_test',)
    )
    sample_rows = cursor.fetchall()
    for row in sample_rows:
        print(f"   📌 {row['content'][:70]}...")
    
    # 评估
    print("\n4. Evaluating questions...")
    results = []
    total_f1 = 0
    total_em = 0
    
    for i, qa in enumerate(qa_list, 1):
        question = qa['question']
        ground_truth = qa.get('answer') or qa.get('adversarial_answer', '')
        category = qa.get('category', 1)
        
        print(f"\n   Q{i}: {question[:60]}...")
        
        try:
            contexts = mimir.query(question, user_id='locomo_test', top_k=5)
            context_text = "\n".join([str(c.memory.content if hasattr(c, 'memory') else c) for c in contexts])
            
            # 调试：打印上下文
            if i <= 3:  # 只打印前3题
                print(f"\n      📄 Context ({len(contexts)} memories):")
                for j, ctx in enumerate(contexts[:3], 1):
                    content = str(ctx.memory.content if hasattr(ctx, 'memory') else ctx)
                    print(f"         {j}. {content[:80]}...")
            
            # 答案生成 prompt - 修复版：强制简洁回答
            prompt = f"""Answer the question using ONLY the context provided. Maximum 10 words. No explanations. Facts only.

Context:
{context_text[:3000]}

Question: {question}

Answer (max 10 words, no explanations):"""
            
            prediction = llm.invoke_mistral(prompt, max_tokens=100, temperature=0.0)
        except Exception as e:
            print(f"      Error: {e}")
            prediction = ""
        
        f1 = calculate_f1(prediction, ground_truth)
        em = prediction.lower().strip() == str(ground_truth).lower().strip()
        
        total_f1 += f1
        total_em += 1 if em else 0
        
        results.append({
            'question': question,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'f1': f1,
            'em': em,
            'category': category
        })
        
        print(f"      Pred: {str(prediction)[:60]}...")
        print(f"      True: {str(ground_truth)[:60]}...")
        print(f"      F1: {f1:.3f}, EM: {em}")
    
    avg_f1 = total_f1 / len(qa_list)
    avg_em = total_em / len(qa_list)
    
    print("\n" + "=" * 60)
    print("📊 RESULTS (Ingestion Pipeline V3)")
    print("=" * 60)
    print(f"Overall F1:  {avg_f1:.4f} ({avg_f1*100:.2f}%)")
    print(f"Overall EM:  {avg_em:.4f}")
    
    cat_stats = {1: {'f1': 0, 'em': 0, 'count': 0},
                 2: {'f1': 0, 'em': 0, 'count': 0},
                 3: {'f1': 0, 'em': 0, 'count': 0}}
    
    for r in results:
        cat = r['category']
        cat_stats[cat]['f1'] += r['f1']
        cat_stats[cat]['em'] += r['em']
        cat_stats[cat]['count'] += 1
    
    print("\nBy Category:")
    for cat, stats in cat_stats.items():
        if stats['count'] > 0:
            print(f"  Cat {cat}: F1={stats['f1']/stats['count']:.4f}, EM={stats['em']/stats['count']:.4f}, N={stats['count']}")
    
    # 对比之前版本
    print("\n" + "=" * 60)
    print("📈 Comparison")
    print("=" * 60)
    print(f"V1 (Original):  F1=10.33%")
    print(f"V2 (Prompt):    F1=8.86%")
    print(f"V3 (Pipeline):  F1={avg_f1*100:.2f}%")
    
    # 保存结果
    with open('locomo_10q_results_v3.json', 'w') as f:
        json.dump({
            'overall': {'f1': avg_f1, 'em': avg_em},
            'by_category': {k: {'f1': v['f1']/v['count'] if v['count'] else 0, 
                               'em': v['em']/v['count'] if v['count'] else 0,
                               'count': v['count']} for k, v in cat_stats.items()},
            'details': results,
            'improvements': [
                'LLM-based fact extraction',
                'Forced temporal normalization',
                'Correct session_date binding'
            ]
        }, f, indent=2)
    print(f"\n💾 Results saved to locomo_10q_results_v3.json")

if __name__ == "__main__":
    test_locomo_v3()
