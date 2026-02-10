#!/usr/bin/env python3
"""
Mimir-Native LoCoMo 10题快速测试 - 批量处理优化版
"""
import os
import sys
import json
import tempfile
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


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


def calculate_em(prediction, ground_truth) -> bool:
    prediction = str(prediction) if prediction else ""
    ground_truth = str(ground_truth) if ground_truth else ""
    return prediction.lower().strip() == ground_truth.lower().strip()


def test_locomo_10q_fast():
    """LoCoMo 10题测试 - 批量处理优化版"""
    from mimir_native import MimirMemory
    from mimir_native.evaluation.locomo_evaluator import LoCoMoEvaluator
    from mimir_native.llm_client import BedrockClient
    
    # 找数据文件
    data_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'mimir-locomo-testbed', 'data', 'locomo10.json'),
        os.path.join(os.path.dirname(__file__), 'data', 'locomo10.json'),
        '/Users/Zhuanz/.openclaw/workspace/mimir-locomo-testbed/data/locomo10.json',
    ]
    
    data_path = None
    for p in data_paths:
        if os.path.exists(p):
            data_path = p
            break
    
    if not data_path:
        logger.error("LoCoMo data not found!")
        return False
    
    logger.info(f"🚀 LoCoMo 10 Question Test (Fast Batch Mode)")
    logger.info(f"   Data: {data_path}")
    
    # 创建临时数据库
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "locomo_10q_fast.db")
    
    start_time = datetime.now()
    
    try:
        # 初始化
        logger.info("\n1. Initializing Mimir-Native...")
        mimir = MimirMemory(db_path=db_path)
        llm = BedrockClient()
        evaluator = LoCoMoEvaluator(mimir, llm)
        logger.info("   ✅ Initialized")
        
        # 加载数据
        logger.info("\n2. Loading LoCoMo data...")
        data = evaluator.load_locomo_data(data_path)
        conv = data[0]
        qa_list = conv.get('qa', [])[:10]
        logger.info(f"   ✅ Loaded {len(qa_list)} questions")
        
        # 使用快速批量摄入
        logger.info("\n3. Ingesting conversation (FAST BATCH MODE)...")
        conversation = conv.get('conversation', conv)
        memory_count = evaluator.ingest_conversation_fast(
            conversation, 
            user_id='locomo_test',
            batch_size=20  # 每批处理 20 条消息
        )
        logger.info(f"   ✅ Ingested {memory_count} memories")
        
        # 评估每题
        logger.info("\n4. Evaluating questions...")
        results = []
        total_f1 = 0
        total_em = 0
        
        for i, qa in enumerate(qa_list, 1):
            question = qa['question']
            ground_truth = qa.get('answer') or qa.get('adversarial_answer', '')
            category = qa.get('category', 1)
            
            logger.info(f"\n   Q{i}: {question[:60]}...")
            
            # 生成答案 - 优化 prompt
            try:
                contexts = mimir.query(question, user_id='locomo_test', top_k=5)
                
                # 按相关性排序（分数高的在前）
                contexts.sort(key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
                
                # 只取最相关的 3 条，避免干扰
                context_text = "\n".join([str(c.memory if hasattr(c, 'memory') else c) for c in contexts[:3]])
                
                # 优化 prompt - 更明确的指令
                prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.

Context (most relevant facts):
{context_text}

Question: {question}

Instructions:
1. If the context contains the answer, provide it directly and concisely
2. If the context implies the answer but doesn't state it explicitly, make a reasonable inference
3. Only say "I don't know" if the context truly contains no relevant information

Answer:"""
                
                prediction = llm.invoke_mistral(prompt, max_tokens=100, temperature=0.1)
            except Exception as e:
                logger.warning(f"      Error: {e}")
                prediction = ""
            
            # 计算指标
            f1 = calculate_f1(prediction, ground_truth)
            em = calculate_em(prediction, ground_truth)
            
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
            
            logger.info(f"      Pred: {str(prediction)[:50]}...")
            logger.info(f"      True: {str(ground_truth)[:50]}...")
            logger.info(f"      F1: {f1:.3f}, EM: {em}")
        
        # 计算总体
        avg_f1 = total_f1 / len(qa_list)
        avg_em = total_em / len(qa_list)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "="*60)
        logger.info("📊 RESULTS")
        logger.info("="*60)
        logger.info(f"Overall F1:  {avg_f1:.4f}")
        logger.info(f"Overall EM:  {avg_em:.4f}")
        logger.info(f"Time:        {elapsed:.1f}s")
        
        # 按类别统计
        cat_stats = {1: {'f1': 0, 'em': 0, 'count': 0},
                     2: {'f1': 0, 'em': 0, 'count': 0},
                     3: {'f1': 0, 'em': 0, 'count': 0}}
        
        for r in results:
            cat = r['category']
            cat_stats[cat]['f1'] += r['f1']
            cat_stats[cat]['em'] += r['em']
            cat_stats[cat]['count'] += 1
        
        logger.info("\nBy Category:")
        for cat, stats in cat_stats.items():
            if stats['count'] > 0:
                logger.info(f"  Cat {cat}: F1={stats['f1']/stats['count']:.4f}, EM={stats['em']/stats['count']:.4f}, N={stats['count']}")
        
        # 保存结果
        result_file = 'locomo_10q_results_fast.json'
        with open(result_file, 'w') as f:
            json.dump({
                'overall': {'f1': avg_f1, 'em': avg_em, 'time_seconds': elapsed},
                'by_category': {k: {'f1': v['f1']/v['count'] if v['count'] else 0, 
                                   'em': v['em']/v['count'] if v['count'] else 0,
                                   'count': v['count']} for k, v in cat_stats.items()},
                'details': results
            }, f, indent=2)
        logger.info(f"\n💾 Results saved to {result_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"\n🧹 Cleaned up")
        except:
            pass


if __name__ == "__main__":
    print("="*60)
    print("Mimir-Native LoCoMo 10Q Test (Fast Batch)")
    print("="*60)
    
    success = test_locomo_10q_fast()
    sys.exit(0 if success else 1)
