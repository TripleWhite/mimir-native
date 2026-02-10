"""
Standalone test for LoCoMo Conversation Date Fix
独立测试脚本 - 不依赖项目其他模块
"""

import sys
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

print("=" * 60)
print("Mimir V2 - Preprocessor Core Functions Test")
print("=" * 60)


# ============================================================================
# Copy of core functions for testing
# ============================================================================

@dataclass
class RawContent:
    text: str
    summary: Optional[str] = None
    chunks: Optional[List[str]] = None
    metadata: Dict[str, Any] = None
    occurred_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
        if self.metadata is None:
            self.metadata = {}


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """解析日期字符串为 datetime 对象"""
    if not date_str:
        return None
    
    if isinstance(date_str, datetime):
        return date_str
    
    formats = [
        "%d %B %Y",      # 7 May 2023
        "%d %b %Y",      # 7 May 2023
        "%B %d, %Y",     # May 7, 2023
        "%b %d, %Y",     # May 7, 2023
        "%Y-%m-%d",      # 2023-05-07
        "%Y/%m/%d",      # 2023/05/07
        "%d-%m-%Y",      # 07-05-2023
        "%d/%m/%Y",      # 07/05/2023
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    return None


class ConversationProcessor:
    """简化版对话处理器"""
    
    def process(self, conversation: dict, metadata: dict) -> RawContent:
        # 提取 session 日期（LoCoMo 修复关键）
        session_date = self._extract_session_date(conversation, metadata)
        
        formatted_lines = []
        message_chunks = []
        
        messages = conversation.get('messages', conversation.get('turns', []))
        
        for turn in messages:
            speaker = turn.get('speaker') or turn.get('user') or turn.get('from', 'Unknown')
            text = turn.get('text') or turn.get('message') or turn.get('content', '')
            
            # 关键修复：获取时间戳
            timestamp = turn.get('timestamp') or turn.get('time') or turn.get('date')
            if not timestamp and session_date:
                timestamp = session_date
            
            formatted_line = f"[{timestamp}] {speaker}: {text}"
            formatted_lines.append(formatted_line)
            message_chunks.append(f"[{timestamp}] {speaker}: {text}")
        
        full_text = "\n".join(formatted_lines)
        
        # 生成摘要
        summary = self._generate_summary(messages, session_date)
        
        # 解析 occurred_at
        occurred_at = parse_date(session_date)
        
        return RawContent(
            text=full_text,
            summary=summary,
            chunks=message_chunks,
            metadata={
                'session_date': session_date,
                'message_count': len(messages),
                'speakers': list(set(
                    (m.get('speaker') or m.get('user') or m.get('from', 'Unknown'))
                    for m in messages
                )),
            },
            occurred_at=occurred_at
        )
    
    def _extract_session_date(self, conversation: dict, metadata: dict) -> Optional[str]:
        """提取 session 日期"""
        for key in ['session_date', 'session_1_date_time', 'date', 'session_time']:
            if key in conversation and conversation[key]:
                return str(conversation[key])
        
        for key in ['session_date', 'timestamp', 'date', 'occurred_at']:
            if key in metadata and metadata[key]:
                return str(metadata[key])
        
        return None
    
    def _generate_summary(self, messages: list, session_date: Optional[str]) -> str:
        speakers = set()
        for msg in messages:
            speaker = msg.get('speaker') or msg.get('user') or 'Unknown'
            speakers.add(speaker)
        
        parts = []
        if session_date:
            parts.append(f"会话日期: {session_date}")
        parts.append(f"参与人数: {len(speakers)}")
        parts.append(f"消息数量: {len(messages)}")
        return " | ".join(parts)
    
    def format_for_locomo(self, conversation: dict) -> str:
        """LoCoMo 专用格式化"""
        session_date = conversation.get('session_date') or conversation.get('session_1_date_time', '')
        messages = conversation.get('messages', [])
        
        lines = []
        for turn in messages:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            lines.append(f"[{session_date}] {speaker}: {text}")
        
        return "\n".join(lines)


# ============================================================================
# Tests
# ============================================================================

def test_parse_date():
    """测试日期解析"""
    print("\n[测试 1] 日期解析")
    print("-" * 40)
    
    # LoCoMo 格式
    dt = parse_date("7 May 2023")
    assert dt is not None, "解析 '7 May 2023' 失败"
    assert dt.year == 2023 and dt.month == 5 and dt.day == 7
    print("✓ '7 May 2023' ->", dt)
    
    # ISO 格式
    dt = parse_date("2023-05-07")
    assert dt is not None
    assert dt.year == 2023
    print("✓ '2023-05-07' ->", dt)
    
    # 其他格式
    dt = parse_date("May 7, 2023")
    assert dt is not None
    print("✓ 'May 7, 2023' ->", dt)
    
    return True


def test_locomo_conversation():
    """测试 LoCoMo 对话处理（关键修复）"""
    print("\n[测试 2] LoCoMo 对话处理（关键修复）")
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
    print(f"✓ 所有 {len(lines)} 条消息都包含日期前缀 [7 May 2023]")
    
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
    print(f"✓ metadata 正确包含 session_date: {result.metadata['session_date']}")
    
    # 打印输出
    print("\n输出文本:")
    for line in lines:
        print(f"  {line}")
    
    return True


def test_locomo_format_method():
    """测试 LoCoMo 专用格式化"""
    print("\n[测试 3] LoCoMo 专用格式化")
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


def test_session_1_date_time_field():
    """测试 session_1_date_time 字段（LoCoMo 特有）"""
    print("\n[测试 4] session_1_date_time 字段支持")
    print("-" * 40)
    
    processor = ConversationProcessor()
    
    # 只有 session_1_date_time 字段（LoCoMo 格式）
    conversation = {
        "session_1_date_time": "15 June 2023",
        "messages": [
            {"speaker": "User", "text": "Hello"}
        ]
    }
    
    result = processor.process(conversation, {})
    
    assert "15 June 2023" in result.text
    assert result.occurred_at.year == 2023
    assert result.occurred_at.month == 6
    assert result.occurred_at.day == 15
    
    print("✓ session_1_date_time 字段正确解析")
    print(f"  occurred_at: {result.occurred_at}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    tests = [
        ("日期解析", test_parse_date),
        ("LoCoMo 对话处理", test_locomo_conversation),
        ("LoCoMo 格式化", test_locomo_format_method),
        ("session_1_date_time 字段", test_session_1_date_time_field),
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
    if failed == 0:
        print("✓ LoCoMo 日期修复验证成功！")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
