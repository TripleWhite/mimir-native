#!/usr/bin/env python3
"""
使用 LLM-Native 方法测试 LoCoMo 10 题
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
import tempfile
from datetime import datetime
from mimir_native import MimirMemory
from mimir_native.llm_client import BedrockClient
from mimir_native.llm_native_memory import LLMNativeMemoryExtractor, LLMNativeRetriever

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

def test_locomo_llm_native():
    """使用 LLM-Native 方法测试 LoCoMo"""
    print("🚀 LoCoMo 10Q Test (LLM-Native)")
    
    # 初始化
    llm = BedrockClient()
    extractor = LLMNativeMemoryExtractor(llm)
    retriever = LLMNativeRetriever(llm)
    
    # 加载 LoCoMo 数据
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'mimir-locomo-testbed', 'data', 'locomo10.json'
    )
    if not os.path.exists(data_path):
        data_path = '/Users/Zhuanz/.openclaw/workspace/mimir-locomo-testbed/data/locomo10.json'
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    conv = data[0]
    qa_list = conv.get('qa', [])[:10]
    conversation = conv.get('conversation', conv)
    
    print(f"✅ Loaded {len(qa_list)} questions")
    
    # 构建对话列表
    all_messages = []
    for session_key, messages in conversation.items():
        if not (session_key.startswith('session_') and '_date_time' not in session_key):
            continue
        if not isinstance(messages, list):
            continue
        
        date_key = f"{session_key}_date_time"
        session_date = conversation.get(date_key)
        
        for msg in messages:
            all_messages.append({
                'speaker': msg.get('speaker', ''),
                'text': msg.get('text', ''),
                'session_date': session_date
            })
    
    print(f"📥 Processing {len(all_messages)} messages...")
    
    # 按 session 分组提取记忆
    all_memories = []
    sessions = {}
    for msg in all_messages:
        date = msg['session_date']
        if date not in sessions:
            sessions[date] = []
        sessions[date].append(msg)
    
    for date, messages in sessions.items():
        print(f"  Extracting memories for {date}...")
        memories = extractor.extract_memories(messages, date)
        all_memories.extend(memories)
        print(f"    -> {len(memories)} memories")
    
    print(f"\n✅ Total memories: {len(all_memories)}")
    
    # 评估每题
    print("\n🎯 Evaluating questions...")
    results = []
    total_f1 = 0
    
    for i, qa in enumerate(qa_list, 1):
        question = qa['question']
        ground_truth = qa.get('answer') or qa.get('adversarial_answer', '')
        category = qa.get('category', 1)
        
        print(f"\nQ{i}: {question[:60]}...")
        
        # 使用 LLM-Native 检索
        relevant = retriever.retrieve(question, all_memories, top_k=5)
        
        # 构建上下文
        context_text = "\n".join([m.content for m in relevant[:3]])
        
        # 生成答案
        prompt = f"""Answer the question based on the context. Be direct and concise (max 20 words).

Context:
{context_text}

Question: {question}

Answer:"""
        
        try:
            prediction = llm.invoke_mistral(prompt, max_tokens=50, temperature=0.0)
        except Exception as e:
            print(f"    Error: {e}")
            prediction = ""
        
        # 计算 F1
        f1 = calculate_f1(prediction, ground_truth)
        total_f1 += f1
        
        print(f"    Pred: {prediction[:60]}...")
        print(f"    True: {str(ground_truth)[:60]}...")
        print(f"    F1: {f1:.3f}")
        
        results.append({
            'question': question,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'f1': f1,
            'category': category
        })
    
    # 计算总体
    avg_f1 = total_f1 / len(qa_list)
    
    print("\n" + "="*60)
    print("📊 RESULTS (LLM-Native)")
    print("="*60)
    print(f"Overall F1: {avg_f1:.4f}")
    
    # 按类别统计
    cat_stats = {1: {'f1': 0, 'count': 0}, 2: {'f1': 0, 'count': 0}, 3: {'f1': 0, 'count': 0}}
    for r in results:
        cat = r['category']
        cat_stats[cat]['f1'] += r['f1']
        cat_stats[cat]['count'] += 1
    
    print("\nBy Category:")
    for cat, stats in cat_stats.items():
        if stats['count'] > 0:
            print(f"  Cat {cat}: F1={stats['f1']/stats['count']:.4f}, N={stats['count']}")
    
    # 保存结果
    with open('locomo_10q_results_llm_native.json', 'w') as f:
        json.dump({
            'overall': {'f1': avg_f1},
            'by_category': {k: {'f1': v['f1']/v['count'] if v['count'] else 0, 'count': v['count']} 
                          for k, v in cat_stats.items()},
            'details': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to locomo_10q_results_llm_native.json")

if __name__ == "__main__":
    test_locomo_llm_native()
