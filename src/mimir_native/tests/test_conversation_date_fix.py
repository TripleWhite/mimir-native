"""
Test for LoCoMo Conversation Date Fix
LoCoMo 对话日期修复验证测试（使用标准库 unittest）

关键测试：
1. 输入：LoCoMo 格式的 conversation（含 session_1_date_time）
2. 输出：每条消息都带有日期前缀
3. 断言：输出文本包含 "7 May 2023" 等具体日期
"""

import sys
import os
import unittest
from datetime import datetime
from pathlib import Path

# 添加项目路径
backend_path = Path(__file__).parent.parent.parent.parent  # 指向 backend 目录
sys.path.insert(0, str(backend_path))

from app.mimir_v2.preprocessors import ConversationProcessor, MultimodalPreprocessor, parse_date


class TestLoCoMoDateFix(unittest.TestCase):
    """
    LoCoMo 日期修复关键验证测试
    
    问题背景：
    - LoCoMo 数据集使用 session_1_date_time 字段存储日期
    - 之前实现可能忽略了这个日期，导致消息没有时间戳
    - 修复后：每条消息必须带上 session_date 作为前缀
    """
    
    def setUp(self):
        """测试前设置"""
        self.processor = ConversationProcessor()
        self.locomo_conversation = {
            "id": "locomo_test_001",
            "user_id": "user_123",
            "session_date": "7 May 2023",
            "session_1_date_time": "7 May 2023",
            "messages": [
                {
                    "speaker": "User",
                    "text": "Hey, I need to book a flight to New York next week.",
                },
                {
                    "speaker": "Assistant",
                    "text": "Sure! What dates are you looking at?",
                },
                {
                    "speaker": "User",
                    "text": "I'm thinking May 15th to May 20th.",
                },
                {
                    "speaker": "Assistant",
                    "text": "Great choice! The weather should be nice then.",
                }
            ]
        }
    
    def test_all_messages_have_date_prefix(self):
        """
        关键测试 1：所有消息都必须有日期前缀
        
        验证：输出文本中每条消息都以 "[7 May 2023]" 开头
        """
        result = self.processor.process(self.locomo_conversation, {})
        
        lines = result.text.strip().split('\n')
        
        # 每条消息行都应该以日期开头
        for line in lines:
            if line.strip():  # 忽略空行
                self.assertIn("[7 May 2023]", line, f"消息缺少日期前缀: {line}")
        
        print(f"✓ 所有 {len(lines)} 条消息都包含日期前缀")
    
    def test_output_contains_specific_date(self):
        """
        关键测试 2：输出文本包含具体日期 "7 May 2023"
        """
        result = self.processor.process(self.locomo_conversation, {})
        
        # 断言：输出文本包含具体日期
        self.assertIn("7 May 2023", result.text, "输出文本必须包含日期 '7 May 2023'")
        
        # 统计日期出现次数（应该等于消息数量）
        date_count = result.text.count("7 May 2023")
        message_count = len(self.locomo_conversation["messages"])
        
        self.assertGreaterEqual(date_count, message_count, 
            f"日期出现次数({date_count})应 >= 消息数量({message_count})")
        
        print(f"✓ 日期 '7 May 2023' 出现了 {date_count} 次，消息数量 {message_count}")
    
    def test_occurred_at_correctly_parsed(self):
        """
        关键测试 3：occurred_at 字段正确解析为 datetime
        """
        result = self.processor.process(self.locomo_conversation, {})
        
        # 验证 occurred_at 不是 None
        self.assertIsNotNone(result.occurred_at, "occurred_at 必须被正确解析")
        
        # 验证 datetime 字段
        self.assertIsInstance(result.occurred_at, datetime, "occurred_at 必须是 datetime 类型")
        self.assertEqual(result.occurred_at.year, 2023, "年份应为 2023")
        self.assertEqual(result.occurred_at.month, 5, "月份应为 5 (May)")
        self.assertEqual(result.occurred_at.day, 7, "日期应为 7")
        
        print(f"✓ occurred_at 正确解析为: {result.occurred_at}")
    
    def test_locomo_format_method(self):
        """
        关键测试 4：LoCoMo 专用格式化方法
        """
        result = self.processor.format_for_locomo(self.locomo_conversation)
        
        lines = result.split('\n')
        
        # 验证格式：每行都应该是 "[7 May 2023] Speaker: Text"
        for line in lines:
            if line.strip():
                self.assertTrue(line.startswith("[7 May 2023]"), f"格式错误: {line}")
        
        # 验证具体内容
        self.assertIn("[7 May 2023] User: Hey, I need to book a flight", result)
        self.assertIn("[7 May 2023] Assistant: Sure! What dates", result)
        
        print(f"✓ LoCoMo 格式化正确，共 {len(lines)} 行")
    
    def test_session_date_in_metadata(self):
        """
        关键测试 5：session_date 正确保存在 metadata 中
        """
        result = self.processor.process(self.locomo_conversation, {})
        
        self.assertIn("session_date", result.metadata, "metadata 必须包含 session_date")
        self.assertEqual(result.metadata["session_date"], "7 May 2023", "session_date 应为 '7 May 2023'")
        
        print(f"✓ metadata 正确包含 session_date: {result.metadata['session_date']}")
    
    def test_multiple_dates_support(self):
        """
        关键测试 6：支持多种日期格式
        """
        test_cases = [
            {"session_date": "7 May 2023", "messages": [{"speaker": "A", "text": "Test"}]},
            {"session_date": "2023-05-07", "messages": [{"speaker": "A", "text": "Test"}]},
            {"session_1_date_time": "7 May 2023", "messages": [{"speaker": "A", "text": "Test"}]},
        ]
        
        for i, conversation in enumerate(test_cases):
            result = self.processor.process(conversation, {})
            self.assertIsNotNone(result.occurred_at, f"测试用例 {i+1} 的日期解析失败")
            self.assertEqual(result.occurred_at.year, 2023, f"测试用例 {i+1} 年份错误")
        
        print(f"✓ 所有 {len(test_cases)} 种日期格式都正确解析")
    
    def test_multimodal_preprocessor_integration(self):
        """
        关键测试 7：MultimodalPreprocessor 集成
        """
        preprocessor = MultimodalPreprocessor()
        
        result = preprocessor.process(
            self.locomo_conversation,
            content_type="conversation",
            metadata={}
        )
        
        # 验证通过主处理器也能正确处理
        self.assertIn("7 May 2023", result.text)
        self.assertIsNotNone(result.occurred_at)
        self.assertEqual(result.occurred_at.year, 2023)
        
        print(f"✓ MultimodalPreprocessor 集成测试通过")


class TestDateExtractionEdgeCases(unittest.TestCase):
    """日期提取边界情况测试"""
    
    def setUp(self):
        """测试前设置"""
        self.processor = ConversationProcessor()
    
    def test_session_1_date_time_priority(self):
        """测试 session_1_date_time 字段优先（LoCoMo 格式）"""
        conversation = {
            "session_1_date_time": "7 May 2023",  # LoCoMo 字段
            "date": "2023-01-01",  # 其他字段
            "messages": [{"speaker": "A", "text": "Test"}]
        }
        
        result = self.processor.process(conversation, {})
        
        # 应该使用 session_1_date_time
        self.assertIn("7 May 2023", result.text)
        self.assertEqual(result.occurred_at.month, 5)
    
    def test_fallback_to_message_timestamp(self):
        """测试回退到消息时间戳"""
        conversation = {
            "messages": [
                {"speaker": "A", "text": "Test", "timestamp": "10 May 2023"},
                {"speaker": "B", "text": "Reply"}  # 无时间戳
            ]
        }
        
        result = self.processor.process(conversation, {})
        
        # 第一条消息应有时间戳
        self.assertIn("10 May 2023", result.text)
    
    def test_no_date_available(self):
        """测试无日期可用的情况"""
        conversation = {
            "messages": [
                {"speaker": "A", "text": "Test"},
                {"speaker": "B", "text": "Reply"}
            ]
        }
        
        result = self.processor.process(conversation, {})
        
        # 应该仍然返回结果，但 occurred_at 为 None
        self.assertIsNotNone(result.text)
        self.assertIsNone(result.occurred_at)


class TestParseDateFunction(unittest.TestCase):
    """日期解析函数测试"""
    
    def test_locomo_date_format(self):
        """测试 LoCoMo 日期格式"""
        dt = parse_date("7 May 2023")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2023)
        self.assertEqual(dt.month, 5)
        self.assertEqual(dt.day, 7)
    
    def test_various_date_formats(self):
        """测试多种日期格式"""
        formats = [
            "7 May 2023",
            "07 May 2023",
            "May 7, 2023",
            "2023-05-07",
            "2023/05/07",
        ]
        
        for fmt in formats:
            dt = parse_date(fmt)
            self.assertIsNotNone(dt, f"无法解析格式: {fmt}")
            self.assertEqual(dt.year, 2023)
            self.assertEqual(dt.month, 5)
            self.assertEqual(dt.day, 7)
    
    def test_invalid_date(self):
        """测试无效日期"""
        dt = parse_date("not a date")
        self.assertIsNone(dt)


# ============================================================================
# Main
# ============================================================================

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("LoCoMo Conversation Date Fix - Verification Tests")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestLoCoMoDateFix))
    suite.addTests(loader.loadTestsFromTestCase(TestDateExtractionEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestParseDateFunction))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 60)
    if result.wasSuccessful():
        print("✓ 所有测试通过！LoCoMo 日期修复验证成功。")
    else:
        print("✗ 部分测试失败，请检查实现。")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 简单的手动测试
    print("\n手动验证测试：")
    print("-" * 40)
    
    processor = ConversationProcessor()
    
    # 测试数据
    conversation = {
        "session_date": "7 May 2023",
        "messages": [
            {"speaker": "User", "text": "Hello!"},
            {"speaker": "Assistant", "text": "Hi there!"}
        ]
    }
    
    result = processor.process(conversation, {})
    
    print("\n输入数据:")
    print(f"  session_date: {conversation['session_date']}")
    print(f"  messages: {len(conversation['messages'])} 条")
    
    print("\n输出文本:")
    for line in result.text.split('\n'):
        print(f"  {line}")
    
    print(f"\noccurred_at: {result.occurred_at}")
    print(f"metadata: {result.metadata}")
    
    # 验证
    success = True
    if "7 May 2023" not in result.text:
        print("\n✗ 失败：输出文本不包含日期 '7 May 2023'")
        success = False
    else:
        print("\n✓ 成功：输出文本包含日期 '7 May 2023'")
    
    if result.occurred_at is None:
        print("✗ 失败：occurred_at 为 None")
        success = False
    else:
        print(f"✓ 成功：occurred_at = {result.occurred_at}")
    
    # 运行 unittest
    print("\n" + "=" * 60)
    success = run_all_tests()
    sys.exit(0 if success else 1)
