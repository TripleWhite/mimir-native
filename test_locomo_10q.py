#!/usr/bin/env python3
"""
Mimir-Native LoCoMo 10题快速测试
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

from mimir_native import MimirMemory
from mimir_native.evaluation.locomo_evaluator import LoCoMoEvaluator
from mimir_native.llm_client import BedrockClient


def calculate_f1(prediction: str, ground_truth) -> float:
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


def test_locomo_10q():
    """LoCoMo 10题测试"""
    
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
    
    logger.info(f"🚀 LoCoMo 10 Question Test")
    logger.info(f"   Data: {data_path}")
    
    # 创建临时数据库
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "locomo_10q.db")
    
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
        conv = data[0]  # 第一个对话
        qa_list = conv.get('qa', [])[:10]  # 只取前10题
        logger.info(f"   ✅ Loaded {len(qa_list)} questions")
        
        # 摄入对话
        logger.info("\n3. Ingesting conversation...")
        # 传入 conversation['conversation'] 而不是整个 conversation
        evaluator.ingest_conversation(conv.get('conversation', conv), user_id='locomo_test')
        logger.info("   ✅ Ingested")
        
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
            
            # 生成答案
            try:
                # 使用检索+生成
                contexts = mimir.query(question, user_id='locomo_test', top_k=5)
                context_text = "\n".join([c.get('content', str(c)) for c in contexts])
                
                prompt = f"""Based on the following context, answer the question concisely.

Context:
{context_text}

Question: {question}

Answer:"""
                prediction = llm.invoke_mistral(prompt)
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
        
        logger.info("\n" + "="*60)
        logger.info("📊 RESULTS")
        logger.info("="*60)
        logger.info(f"Overall F1:  {avg_f1:.4f}")
        logger.info(f"Overall EM:  {avg_em:.4f}")
        
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
        result_file = 'locomo_10q_results.json'
        with open(result_file, 'w') as f:
            json.dump({
                'overall': {'f1': avg_f1, 'em': avg_em},
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
    print("Mimir-Native LoCoMo 10Q Test")
    print("="*60)
    
    success = test_locomo_10q()
    sys.exit(0 if success else 1)
