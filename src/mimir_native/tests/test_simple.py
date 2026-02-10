"""
Simple test for LoCoMo Conversation Date Fix (No pytest dependency)
LoCoMo 对话日期修复验证测试（无 pytest 依赖版本）
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
backend_path = Path(__file__).parent.parent.parent.parent  # 指向 backend 目录
sys.path.insert(0, str(backend_path))
from app.mimir_v2.preprocessors.base import BasePreprocessor, RawContent, parse_date
from app.mimir_v2.preprocessors.document import DocumentProcessor
from app.mimir_v2.preprocessors.image import ImageProcessor
from app.mimir_v2.preprocessors.audio import AudioProcessor
from app.mimir_v2.preprocessors.conversation import ConversationProcessor
from app.mimir_v2.preprocessors import MultimodalPreprocessor


def test_parse_date():
    """测试日期解析"""
    print("\n测试 1: 日期解析")
    print("-" * 40)
    
    # 测试 LoCoMo 格式
    dt = parse_date("7 May 2023")
    assert dt is not None, "解析 '7 May 2023' 失败"
    assert dt.year == 2023 and dt.month == 5 and dt.day == 7
    print("✓ '7 May 2023' ->", dt)
    
    # 测试 ISO 格式
    dt = parse_date("2023-05-07")
    assert dt is not None
    assert dt.year == 2023
    print("✓ '2023-05-07' ->", dt)
    
    return True


def test_locomo_conversation():
    """测试 LoCoMo 对话处理"""
    print("\n测试 2: LoCoMo 对话处理（关键修复）")
    print("-" * 40)
    
    processor = ConversationProcessor()
    
    # LoCoMo 格式对话
    conversation = {
        "id": "locomo_test_001",
        "session_date": "7 May 2023",
        "session_1_date_time": "7 May 2023",
        "messages": [
            {"speaker": "User", "text": "Hey, I need to book a flight to New York."},
            {"speaker": "Assistant", "text": "Sure! What dates are you looking at?"},
            {"speaker": "User", "text": "I'm thinking May 15th to May 20th."},
            {"speaker": "Assistant", "text": "Great choice!"}
        ]
    }
    
    result = processor.process(conversation, {})
    
    # 验证 1: 输出包含日期
    assert "7 May 2023" in result.text, "输出文本必须包含日期 '7 May 2023'"
    print("✓ 输出文本包含日期 '7 May 2023'")
    
    # 验证 2: 每条消息都有日期前缀
    lines = result.text.strip().split('\n')
    for line in lines:
        if line.strip():
            assert "[7 May 2023]" in line, f"消息缺少日期前缀: {line}"
    print(f"✓ 所有 {len(lines)} 条消息都包含日期前缀")
    
    # 验证 3: occurred_at 正确解析
    assert result.occurred_at is not None, "occurred_at 必须被正确解析"
    assert isinstance(result.occurred_at, datetime), "occurred_at 必须是 datetime 类型"
    assert result.occurred_at.year == 2023
    assert result.occurred_at.month == 5
    assert result.occurred_at.day == 7
    print(f"✓ occurred_at 正确解析为: {result.occurred_at}")
    
    # 验证 4: metadata 包含 session_date
    assert "session_date" in result.metadata
    assert result.metadata["session_date"] == "7 May 2023"
    print(f"✓ metadata 正确包含 session_date")
    
    # 打印输出
    print("\n输出文本:")
    print(result.text)
    
    return True


def test_locomo_format_method():
    """测试 LoCoMo 专用格式化"""
    print("\n测试 3: LoCoMo 专用格式化")
    print("-" * 40)
    
    processor = ConversationProcessor()
    
    conversation = {
        "session_date": "7 May 2023",
        "messages": [
            {"speaker": "User", "text": "Question?"},
            {"speaker": "Assistant", "text": "Answer!"}
        ]
    }
    
    result = processor.format_for_locomo(conversation)
    
    lines = result.split('\n')
    assert len(lines) == 2
    assert lines[0] == "[7 May 2023] User: Question?"
    assert lines[1] == "[7 May 2023] Assistant: Answer!"
    
    print("✓ LoCoMo 格式化正确")
    print("输出:")
    print(result)
    
    return True


def test_multimodal_preprocessor():
    """测试多模态预处理器集成"""
    print("\n测试 4: MultimodalPreprocessor 集成")
    print("-" * 40)
    
    preprocessor = MultimodalPreprocessor()
    
    conversation = {
        "session_date": "7 May 2023",
        "messages": [
            {"speaker": "A", "text": "Hello"},
            {"speaker": "B", "text": "World"}
        ]
    }
    
    result = preprocessor.process(conversation, "conversation", {})
    
    assert "7 May 2023" in result.text
    assert result.occurred_at is not None
    assert result.occurred_at.year == 2023
    
    print("✓ MultimodalPreprocessor 集成测试通过")
    print(f"  occurred_at: {result.occurred_at}")
    
    return True


def test_document_processor():
    """测试文档处理器"""
    print("\n测试 5: DocumentProcessor")
    print("-" * 40)
    
    processor = DocumentProcessor()
    
    # 测试文本处理
    content = "This is a test document.\n\nIt has multiple paragraphs."
    result = processor.process(content, {"file_name": "test.txt"})
    
    assert "test document" in result.text
    assert len(result.chunks) > 0
    print("✓ 文档处理正常")
    
    return True


def test_all_processors():
    """测试所有处理器类型"""
    print("\n测试 6: 所有处理器类型")
    print("-" * 40)
    
    preprocessor = MultimodalPreprocessor()
    
    # 测试 supports
    assert preprocessor.supports("document")
    assert preprocessor.supports("image")
    assert preprocessor.supports("audio")
    assert preprocessor.supports("conversation")
    assert preprocessor.supports("chat")
    assert preprocessor.supports("pdf")
    print("✓ 所有类型支持检测正常")
    
    # 测试处理器获取
    doc_proc = preprocessor.get_processor("document")
    assert doc_proc is not None
    print("✓ 处理器获取正常")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Mimir V2 - Multimodal Preprocessors Test Suite")
    print("=" * 60)
    
    tests = [
        ("日期解析", test_parse_date),
        ("LoCoMo 对话处理", test_locomo_conversation),
        ("LoCoMo 格式化", test_locomo_format_method),
        ("多模态预处理器集成", test_multimodal_preprocessor),
        ("文档处理器", test_document_processor),
        ("所有处理器类型", test_all_processors),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {name} 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name} 错误: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
