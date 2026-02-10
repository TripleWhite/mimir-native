"""
Tests for Multimodal Preprocessors (使用标准库 unittest)
多模态预处理器测试
"""

import sys
import os
import unittest
import tempfile
from datetime import datetime
from pathlib import Path

# 添加项目路径
backend_path = Path(__file__).parent.parent.parent.parent  # 指向 backend 目录
sys.path.insert(0, str(backend_path))

from app.mimir_v2.preprocessors import (
    MultimodalPreprocessor,
    DocumentProcessor,
    ImageProcessor,
    AudioProcessor,
    ConversationProcessor,
    RawContent,
    parse_date,
)


# ============================================================================
# Utility Tests
# ============================================================================

class TestParseDate(unittest.TestCase):
    """测试日期解析函数"""
    
    def test_parse_7_may_2023(self):
        """测试 LoCoMo 格式日期"""
        result = parse_date("7 May 2023")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2023)
        self.assertEqual(result.month, 5)
        self.assertEqual(result.day, 7)
    
    def test_parse_iso_date(self):
        """测试 ISO 格式日期"""
        result = parse_date("2023-05-07")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2023)
        self.assertEqual(result.month, 5)
        self.assertEqual(result.day, 7)
    
    def test_parse_datetime_object(self):
        """测试传入 datetime 对象"""
        dt = datetime(2023, 5, 7, 10, 30, 0)
        result = parse_date(dt)
        self.assertEqual(result, dt)
    
    def test_parse_none(self):
        """测试 None 输入"""
        result = parse_date(None)
        self.assertIsNone(result)
    
    def test_parse_invalid(self):
        """测试无效日期"""
        result = parse_date("not a date")
        self.assertIsNone(result)


# ============================================================================
# Base Preprocessor Tests
# ============================================================================

class TestRawContent(unittest.TestCase):
    """测试 RawContent 数据类"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        content = RawContent(
            text="Hello world",
            summary="A greeting",
            chunks=["Hello", "world"],
            metadata={"key": "value"},
            occurred_at=datetime(2023, 5, 7)
        )
        self.assertEqual(content.text, "Hello world")
        self.assertEqual(content.summary, "A greeting")
        self.assertEqual(len(content.chunks), 2)
        self.assertEqual(content.metadata["key"], "value")
    
    def test_default_values(self):
        """测试默认值"""
        content = RawContent(text="Test")
        self.assertEqual(content.chunks, [])
        self.assertEqual(content.metadata, {})
        self.assertIsNone(content.summary)
        self.assertIsNone(content.occurred_at)


# ============================================================================
# Document Processor Tests
# ============================================================================

class TestDocumentProcessor(unittest.TestCase):
    """测试文档预处理器"""
    
    def setUp(self):
        """测试前设置"""
        self.processor = DocumentProcessor()
    
    def test_supports_document_types(self):
        """测试支持的内容类型"""
        self.assertTrue(self.processor.supports("document"))
        self.assertTrue(self.processor.supports("pdf"))
        self.assertTrue(self.processor.supports("docx"))
        self.assertTrue(self.processor.supports("txt"))
        self.assertFalse(self.processor.supports("image"))
    
    def test_process_text_content(self):
        """测试处理文本内容"""
        content = "This is a test document.\n\nIt has multiple paragraphs."
        result = self.processor.process(content, {"file_name": "test.txt"})
        
        self.assertIsInstance(result, RawContent)
        self.assertIn("test document", result.text)
        self.assertEqual(result.metadata["file_name"], "test.txt")
        self.assertGreater(len(result.chunks), 0)
    
    def test_process_txt_file(self):
        """测试处理 TXT 文件"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Line 1\nLine 2\nLine 3")
            test_file = f.name
        
        try:
            result = self.processor.process(test_file, {})
            
            self.assertIsInstance(result, RawContent)
            self.assertIn("Line 1", result.text)
            self.assertEqual(result.metadata["file_type"], ".txt")
            self.assertEqual(result.metadata["line_count"], 3)
        finally:
            os.unlink(test_file)
    
    def test_chunk_by_structure(self):
        """测试按结构分块"""
        text = """# Heading 1
This is the first section.

# Heading 2
This is the second section."""
        
        result = self.processor._chunk_by_structure(text)
        
        # 应该分成多个块
        self.assertGreaterEqual(len(result), 1)
    
    def test_extract_metadata(self):
        """测试元数据提取"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello world")
            test_file = f.name
        
        try:
            metadata = self.processor._extract_metadata(test_file, "Hello world")
            
            self.assertEqual(metadata["char_count"], 11)
            self.assertEqual(metadata["word_count"], 2)
            self.assertIn("file_size", metadata)
        finally:
            os.unlink(test_file)


# ============================================================================
# Image Processor Tests
# ============================================================================

class TestImageProcessor(unittest.TestCase):
    """测试图像预处理器"""
    
    def setUp(self):
        """测试前设置"""
        self.processor = ImageProcessor(vlm_client=None)
    
    def test_supports_image_types(self):
        """测试支持的内容类型"""
        self.assertTrue(self.processor.supports("image"))
        self.assertTrue(self.processor.supports("img"))
        self.assertTrue(self.processor.supports("picture"))
        self.assertFalse(self.processor.supports("document"))
    
    def test_process_image_metadata(self):
        """测试处理图像元数据"""
        # 创建一个简单的测试数据
        result = self.processor._generate_basic_description({
            "file_name": "test.jpg",
            "created_date": "2023-05-07"
        })
        
        self.assertIn("test.jpg", result)
        self.assertIn("2023-05-07", result)
    
    def test_invalid_content_type(self):
        """测试无效内容类型"""
        with self.assertRaises(ValueError):
            self.processor.process(12345, {})


# ============================================================================
# Audio Processor Tests
# ============================================================================

class TestAudioProcessor(unittest.TestCase):
    """测试音频预处理器"""
    
    def setUp(self):
        """测试前设置"""
        self.processor = AudioProcessor()
    
    def test_supports_audio_types(self):
        """测试支持的内容类型"""
        self.assertTrue(self.processor.supports("audio"))
        self.assertTrue(self.processor.supports("voice"))
        self.assertTrue(self.processor.supports("speech"))
        self.assertFalse(self.processor.supports("document"))
    
    def test_format_timestamp(self):
        """测试时间戳格式化"""
        self.assertEqual(self.processor._format_timestamp(65), "01:05")
        self.assertEqual(self.processor._format_timestamp(0), "00:00")
        self.assertEqual(self.processor._format_timestamp(3661), "61:01")
    
    def test_format_transcription(self):
        """测试转录格式化"""
        transcription = {
            "segments": [
                {"speaker": "Speaker A", "start": 0, "text": "Hello"},
                {"speaker": "Speaker B", "start": 5, "text": "Hi there"}
            ]
        }
        
        result = self.processor._format_transcription(transcription)
        
        self.assertIn("[Speaker A] 00:00 Hello", result)
        self.assertIn("[Speaker B] 00:05 Hi there", result)


# ============================================================================
# Conversation Processor Tests
# ============================================================================

class TestConversationProcessor(unittest.TestCase):
    """测试对话预处理器（关键 LoCoMo 修复）"""
    
    def setUp(self):
        """测试前设置"""
        self.processor = ConversationProcessor()
    
    def test_supports_conversation_types(self):
        """测试支持的内容类型"""
        self.assertTrue(self.processor.supports("conversation"))
        self.assertTrue(self.processor.supports("chat"))
        self.assertTrue(self.processor.supports("dialogue"))
        self.assertFalse(self.processor.supports("audio"))
    
    def test_process_locomo_format(self):
        """测试 LoCoMo 格式处理（关键测试）"""
        conversation = {
            "session_date": "7 May 2023",
            "messages": [
                {"speaker": "User", "text": "Hello!"},
                {"speaker": "Assistant", "text": "Hi there!"}
            ]
        }
        
        result = self.processor.process(conversation, {})
        
        # 验证输出包含日期前缀
        self.assertIn("[7 May 2023]", result.text)
        self.assertIn("User: Hello!", result.text)
        self.assertIn("Assistant: Hi there!", result.text)
        
        # 验证 occurred_at 正确解析
        self.assertIsNotNone(result.occurred_at)
        self.assertEqual(result.occurred_at.year, 2023)
        self.assertEqual(result.occurred_at.month, 5)
        self.assertEqual(result.occurred_at.day, 7)
    
    def test_extract_session_date(self):
        """测试 session 日期提取"""
        # LoCoMo 格式
        conv_locomo = {"session_1_date_time": "7 May 2023"}
        self.assertEqual(self.processor._extract_session_date(conv_locomo, {}), "7 May 2023")
        
        # 标准格式
        conv_std = {"session_date": "2023-05-07"}
        self.assertEqual(self.processor._extract_session_date(conv_std, {}), "2023-05-07")
        
        # Metadata 优先
        self.assertEqual(self.processor._extract_session_date({}, {"session_date": "2023-06-01"}), "2023-06-01")
    
    def test_format_for_locomo(self):
        """测试 LoCoMo 专用格式化"""
        conversation = {
            "session_date": "7 May 2023",
            "messages": [
                {"speaker": "User", "text": "Question?"},
                {"speaker": "Assistant", "text": "Answer!"}
            ]
        }
        
        result = self.processor.format_for_locomo(conversation)
        
        lines = result.split('\n')
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "[7 May 2023] User: Question?")
        self.assertEqual(lines[1], "[7 May 2023] Assistant: Answer!")
    
    def test_parse_plain_text(self):
        """测试纯文本解析"""
        text = """User: Hello
Assistant: Hi there
User: How are you?"""
        
        result = self.processor._parse_plain_text(text)
        
        self.assertEqual(len(result["messages"]), 3)
        self.assertEqual(result["messages"][0]["speaker"], "User")
        self.assertEqual(result["messages"][0]["text"], "Hello")
    
    def test_conversation_summary(self):
        """测试对话摘要生成"""
        messages = [
            {"speaker": "User", "text": "Hello"},
            {"speaker": "Assistant", "text": "Hi"}
        ]
        
        summary = self.processor._generate_conversation_summary(messages, "7 May 2023")
        
        self.assertIn("7 May 2023", summary)
        self.assertIn("参与人数: 2", summary)
        self.assertIn("消息数量: 2", summary)


# ============================================================================
# Multimodal Preprocessor Integration Tests
# ============================================================================

class TestMultimodalPreprocessor(unittest.TestCase):
    """测试多模态预处理器主类"""
    
    def setUp(self):
        """测试前设置"""
        self.preprocessor = MultimodalPreprocessor()
    
    def test_initialization(self):
        """测试初始化"""
        mp = MultimodalPreprocessor()
        self.assertIn('document', mp.processors)
        self.assertIn('image', mp.processors)
        self.assertIn('audio', mp.processors)
        self.assertIn('conversation', mp.processors)
    
    def test_normalize_type(self):
        """测试类型标准化"""
        self.assertEqual(self.preprocessor._normalize_type("PDF"), "document")
        self.assertEqual(self.preprocessor._normalize_type("Chat"), "conversation")
        self.assertEqual(self.preprocessor._normalize_type("voice"), "audio")
        self.assertEqual(self.preprocessor._normalize_type("PHOTO"), "image")
    
    def test_supports(self):
        """测试支持检查"""
        self.assertTrue(self.preprocessor.supports("document"))
        self.assertTrue(self.preprocessor.supports("pdf"))
        self.assertTrue(self.preprocessor.supports("conversation"))
        self.assertTrue(self.preprocessor.supports("chat"))
        self.assertFalse(self.preprocessor.supports("unknown_type"))
    
    def test_list_supported_types(self):
        """测试列出支持类型"""
        types = self.preprocessor.list_supported_types()
        self.assertIn("document", types)
        self.assertIn("image", types)
        self.assertIn("audio", types)
        self.assertIn("conversation", types)
    
    def test_process_conversation_integration(self):
        """测试对话处理集成"""
        conversation = {
            "session_date": "7 May 2023",
            "messages": [
                {"speaker": "A", "text": "Hello"},
                {"speaker": "B", "text": "World"}
            ]
        }
        
        result = self.preprocessor.process(conversation, "conversation", {})
        
        self.assertIsInstance(result, RawContent)
        self.assertIn("[7 May 2023]", result.text)
        self.assertIsNotNone(result.occurred_at)
    
    def test_process_text_document(self):
        """测试文本文档处理"""
        result = self.preprocessor.process("Hello world", "document", {})
        
        self.assertIsInstance(result, RawContent)
        self.assertIn("Hello world", result.text)
    
    def test_get_processor(self):
        """测试获取处理器"""
        doc_processor = self.preprocessor.get_processor("document")
        self.assertIsInstance(doc_processor, DocumentProcessor)
        
        conv_processor = self.preprocessor.get_processor("chat")
        self.assertIsInstance(conv_processor, ConversationProcessor)
    
    def test_unknown_content_type(self):
        """测试未知内容类型"""
        with self.assertRaises(ValueError) as context:
            self.preprocessor.process("content", "unknown_type", {})
        
        self.assertIn("Unknown content type", str(context.exception))


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling(unittest.TestCase):
    """测试错误处理"""
    
    def test_document_not_found(self):
        """测试文件不存在处理"""
        processor = DocumentProcessor()
        result = processor.process("/nonexistent/file.pdf", {})
        
        self.assertTrue("错误" in result.text or "error" in result.text.lower())
        self.assertIn("error", result.metadata)
    
    def test_conversation_invalid_format(self):
        """测试无效对话格式"""
        processor = ConversationProcessor()
        result = processor.process(12345, {})
        
        self.assertTrue("错误" in result.text or "error" in result.metadata)


# ============================================================================
# Run Tests
# ============================================================================

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Multimodal Preprocessors Test Suite")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestParseDate))
    suite.addTests(loader.loadTestsFromTestCase(TestRawContent))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestImageProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestConversationProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestMultimodalPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"测试结果: {result.testsRun} 个测试")
    if result.wasSuccessful():
        print("✓ 所有测试通过！")
    else:
        print(f"✗ 失败: {len(result.failures)} 个")
        print(f"✗ 错误: {len(result.errors)} 个")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
