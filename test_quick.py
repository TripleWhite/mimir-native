#!/usr/bin/env python3
"""
Mimir-Native 快速测试脚本
验证基本功能是否正常
"""
import os
import sys
import tempfile
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """测试基本功能"""
    from mimir_native import MimirMemory
    
    # 创建临时数据库
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_mimir_native.db")
    
    logger.info(f"🚀 Testing Mimir-Native with database: {db_path}")
    
    try:
        # 1. 初始化 MimirMemory
        logger.info("1. Initializing MimirMemory...")
        mimir = MimirMemory(db_path=db_path)
        logger.info("   ✅ MimirMemory initialized")
        
        # 2. 添加简单内容
        logger.info("2. Adding simple text content...")
        memories = mimir.add_content(
            content="Caroline visited the LGBTQ support group on May 7, 2023.",
            content_type="text",
            user_id="test_user"
        )
        logger.info(f"   ✅ Added {len(memories)} memories")
        
        # 3. 查询
        logger.info("3. Testing query...")
        results = mimir.query(
            query="When did Caroline visit the support group?",
            user_id="test_user",
            top_k=5
        )
        logger.info(f"   ✅ Retrieved {len(results)} results")
        
        # 4. 添加更多内容（对话格式）
        logger.info("4. Adding conversation content...")
        conversation = {
            "messages": [
                {"speaker": "Caroline", "text": "I had a great time at the meeting yesterday."},
                {"speaker": "Friend", "text": "That's wonderful! When was it?"},
                {"speaker": "Caroline", "text": "It was on May 7, 2023."}
            ],
            "session_date": "2023-05-07"
        }
        memories2 = mimir.add_content(
            content=conversation,
            content_type="conversation",
            user_id="test_user"
        )
        logger.info(f"   ✅ Added {len(memories2)} memories from conversation")
        
        # 5. 时序查询
        logger.info("5. Testing temporal query...")
        results2 = mimir.query(
            query="What happened on May 7, 2023?",
            user_id="test_user",
            top_k=5
        )
        logger.info(f"   ✅ Retrieved {len(results2)} results for temporal query")
        
        logger.info("\n✅ All basic tests passed!")
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
            logger.info(f"\n🧹 Cleaned up temp directory")
        except:
            pass


def test_locomo_10q():
    """测试 LoCoMo 10 题"""
    from mimir_native import MimirMemory
    from mimir_native.evaluation.locomo_evaluator import LoCoMoEvaluator
    
    # 检查数据文件
    data_path = os.path.join(os.path.dirname(__file__), '..', 'mimir-locomo-testbed', 'data', 'locomo10.json')
    if not os.path.exists(data_path):
        # 尝试其他路径
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'locomo10.json')
    
    if not os.path.exists(data_path):
        logger.warning("LoCoMo data not found, skipping LoCoMo test")
        return False
    
    logger.info(f"🚀 Testing LoCoMo 10 questions with data: {data_path}")
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_locomo.db")
    
    try:
        # 初始化
        mimir = MimirMemory(db_path=db_path)
        
        # 创建评估器（需要 LLM 客户端）
        from mimir_native.llm_client import BedrockClient
        llm = BedrockClient()
        evaluator = LoCoMoEvaluator(mimir, llm)
        
        # 加载数据
        data = evaluator.load_locomo_data(data_path)
        
        # 只测试第一个对话的前3题
        conv = data[0]
        logger.info(f"Testing with conversation: {conv.get('id', 'unknown')}")
        
        # 摄入对话
        evaluator.ingest_conversation(conv, user_id='locomo_test')
        
        # 测试前3题
        qa_pairs = conv.get('qa', [])[:3]
        logger.info(f"Testing {len(qa_pairs)} questions")
        
        for i, qa in enumerate(qa_pairs):
            question = qa['question']
            answer = qa.get('answer') or qa.get('adversarial_answer', '')
            
            logger.info(f"\nQ{i+1}: {question}")
            logger.info(f"Expected: {answer}")
            
            # 生成答案
            prediction = evaluator._answer_question(question, 'locomo_test')
            logger.info(f"Predicted: {prediction}")
        
        logger.info("\n✅ LoCoMo 10q test completed")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ LoCoMo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


if __name__ == "__main__":
    print("=" * 60)
    print("Mimir-Native Quick Test")
    print("=" * 60)
    
    # 运行基本测试
    success = test_basic_functionality()
    
    if success:
        print("\n" + "=" * 60)
        print("All tests passed! ✅")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Tests failed! ❌")
        print("=" * 60)
        sys.exit(1)
