#!/usr/bin/env python3
"""
使用 batch_processor_v2 测试 LoCoMo 10 题
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 使用 v2 batch_processor
import mimir_native.batch_processor_v2 as bp_module
import mimir_native.evaluation.locomo_evaluator as eval_module

# 替换原模块
import mimir_native
mimir_native.batch_processor = bp_module

import json
import tempfile
from mimir_native import MimirMemory
from mimir_native.llm_client import BedrockClient
from mimir_native.evaluation.locomo_evaluator import LoCoMoEvaluator

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

def test_locomo_v2():
    """使用 v2 batch processor 测试 LoCoMo"""
    print("🚀 LoCoMo 10Q Test (Batch Processor V2)")
    print("=" * 60)
    
    mimir = MimirMemory(db_path=':memory:')
    llm = BedrockClient()
    evaluator = LoCoMoEvaluator(mimir, llm)
    
    # 加载数据
    data_path = '/Users/Zhuanz/.openclaw/workspace/mimir-locomo-testbed/data/locomo10.json'
    
    print(f"\n2. Loading LoCoMo data...")
    data = evaluator.load_locomo_data(data_path)
    conv = data[0]
    qa_list = conv.get('qa', [])[:10]
    print(f"   ✅ Loaded {len(qa_list)} questions")
    
    # 使用新的 batch_processor_v2
    print("\n3. Ingesting conversation (BATCH PROCESSOR V2)...")
    from mimir_native.batch_processor_v2 import BatchProcessor
    processor = bp_module.BatchProcessor(mimir, llm, max_workers=5)
    
    # 转换数据格式
    conversations = []
    conversation = conv.get('conversation', conv)
    
    for session_key, messages in conversation.items():
        if not (session_key.startswith('session_') and '_date_time' not in session_key):
            continue
        if not isinstance(messages, list):
            continue
        
        date_key = f"{session_key}_date_time"
        session_date = conversation.get(date_key)
        
        if session_date:
            conversations.append({
                'session_date': session_date,
                'messages': messages
            })
    
    result = processor.process_conversations_batch(
        conversations,
        user_id='locomo_test',
        batch_size=10
    )
    
    print(f"   ✅ Ingested {result['memories']} memories from {result['processed']} messages")
    
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
            
            # 更简洁的 prompt
            prompt = f"""Based on the context, answer concisely (max 10 words).

Context:
{context_text[:2000]}

Question: {question}

Answer:"""
            
            prediction = llm.invoke_mistral(prompt, max_tokens=50, temperature=0.0)
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
        
        print(f"      Pred: {str(prediction)[:50]}...")
        print(f"      True: {str(ground_truth)[:50]}...")
        print(f"      F1: {f1:.3f}, EM: {em}")
    
    avg_f1 = total_f1 / len(qa_list)
    avg_em = total_em / len(qa_list)
    
    print("\n" + "=" * 60)
    print("📊 RESULTS (Batch Processor V2)")
    print("=" * 60)
    print(f"Overall F1:  {avg_f1:.4f}")
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
    
    # 保存结果
    with open('locomo_10q_results_v2.json', 'w') as f:
        json.dump({
            'overall': {'f1': avg_f1, 'em': avg_em},
            'by_category': {k: {'f1': v['f1']/v['count'] if v['count'] else 0, 
                               'em': v['em']/v['count'] if v['count'] else 0,
                               'count': v['count']} for k, v in cat_stats.items()},
            'details': results
        }, f, indent=2)
    print(f"\n💾 Results saved to locomo_10q_results_v2.json")

if __name__ == "__main__":
    test_locomo_v2()
