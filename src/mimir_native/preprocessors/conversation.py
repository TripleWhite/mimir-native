"""
Conversation Preprocessor
对话处理 - 关键时序修复

关键修复：正确处理 LoCoMo 数据集的 session 日期
确保每条消息都带上绝对日期前缀
"""

import logging
from typing import Any, Optional, List, Dict
from datetime import datetime
import json
import re

from .base import BasePreprocessor, RawContent, parse_date

logger = logging.getLogger(__name__)


class ConversationProcessor(BasePreprocessor):
    """
    对话预处理器
    
    关键修复（LoCoMo 数据集）：
    - 提取 session 日期（如 "7 May 2023"）
    - 每条消息都带上绝对日期前缀
    - 正确处理 occurred_at 时序信息
    
    支持格式：
    - LoCoMo 格式: {"session_date": "7 May 2023", "messages": [...]}
    - 标准格式: {"messages": [{"speaker": "A", "text": "...", "timestamp": "..."}]}
    """
    
    def supports(self, content_type: str) -> bool:
        """检查是否支持指定的内容类型"""
        return content_type.lower() in {'conversation', 'chat', 'dialogue', 'message'}
    
    def process(self, content: Any, metadata: dict) -> RawContent:
        """
        处理对话内容
        
        Args:
            content: 对话字典或 JSON 字符串
            metadata: 元数据字典
            
        Returns:
            RawContent: 标准化的内容对象
                text 格式：[timestamp] speaker: text
        """
        try:
            # 解析对话数据
            conversation = self._parse_conversation(content)
            
            # 提取 session 日期（LoCoMo 修复关键）
            session_date = self._extract_session_date(conversation, metadata)
            
            # 处理消息
            formatted_lines = []
            message_chunks = []
            
            messages = conversation.get('messages', conversation.get('turns', []))
            
            for i, turn in enumerate(messages):
                # 提取消息信息
                speaker = turn.get('speaker') or turn.get('user') or turn.get('from', 'Unknown')
                text = turn.get('text') or turn.get('message') or turn.get('content', '')
                
                # 关键修复：获取时间戳，优先使用消息自带的时间戳，否则使用 session_date
                timestamp = turn.get('timestamp') or turn.get('time') or turn.get('date')
                if not timestamp and session_date:
                    # 如果没有具体时间，使用 session_date
                    timestamp = session_date
                
                # 格式化行
                formatted_line = f"[{timestamp}] {speaker}: {text}"
                formatted_lines.append(formatted_line)
                
                # 分块：每条消息作为一个块
                message_chunks.append(f"[{timestamp}] {speaker}: {text}")
            
            full_text = "\n".join(formatted_lines)
            
            # 生成摘要
            summary = self._generate_conversation_summary(messages, session_date)
            
            # 解析 occurred_at（关键时序信息）
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
                    'conversation_id': conversation.get('id') or conversation.get('session_id'),
                    **metadata
                },
                occurred_at=occurred_at
            )
            
        except Exception as e:
            logger.error(f"对话处理失败: {e}")
            return RawContent(
                text=f"[对话处理错误] {str(e)}",
                summary="对话处理失败",
                chunks=[],
                metadata={'error': str(e), **metadata},
                occurred_at=None
            )
    
    def _parse_conversation(self, content: Any) -> Dict:
        """解析对话数据为字典"""
        if isinstance(content, dict):
            return content
        
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 尝试作为简单文本解析
                return self._parse_plain_text(content)
        
        raise ValueError(f"不支持的对话内容类型: {type(content)}")
    
    def _parse_plain_text(self, text: str) -> Dict:
        """从纯文本解析对话"""
        messages = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 尝试匹配 "Speaker: Text" 格式
            match = re.match(r'^(?P<speaker>[^:]+):\s*(?P<text>.+)$', line)
            if match:
                messages.append({
                    'speaker': match.group('speaker').strip(),
                    'text': match.group('text').strip()
                })
            else:
                # 无 speaker 前缀的行
                messages.append({
                    'speaker': 'Unknown',
                    'text': line
                })
        
        return {'messages': messages}
    
    def _extract_session_date(self, conversation: Dict, metadata: Dict) -> Optional[str]:
        """
        提取 session 日期（LoCoMo 关键修复）
        
        优先级：
        1. conversation['session_date']
        2. conversation['session_1_date_time'] (LoCoMo 格式)
        3. conversation['date']
        4. metadata['session_date']
        5. metadata['timestamp']
        """
        # 从 conversation 提取
        for key in ['session_date', 'session_1_date_time', 'date', 'session_time']:
            if key in conversation and conversation[key]:
                return str(conversation[key])
        
        # 从 metadata 提取
        for key in ['session_date', 'timestamp', 'date', 'occurred_at']:
            if key in metadata and metadata[key]:
                return str(metadata[key])
        
        # 尝试从消息中提取
        messages = conversation.get('messages', [])
        for msg in messages:
            for key in ['timestamp', 'date', 'time']:
                if key in msg and msg[key]:
                    return str(msg[key])
        
        return None
    
    def _generate_conversation_summary(self, messages: List[Dict], session_date: Optional[str]) -> str:
        """生成对话摘要"""
        summary_parts = []
        
        if session_date:
            summary_parts.append(f"会话日期: {session_date}")
        
        # 统计说话人
        speakers = set()
        for msg in messages:
            speaker = msg.get('speaker') or msg.get('user') or msg.get('from', 'Unknown')
            speakers.add(speaker)
        
        summary_parts.append(f"参与人数: {len(speakers)}")
        summary_parts.append(f"消息数量: {len(messages)}")
        
        # 添加前几条消息预览
        if messages:
            preview_msgs = messages[:2]
            preview_texts = []
            for msg in preview_msgs:
                speaker = msg.get('speaker') or msg.get('user') or 'Unknown'
                text = msg.get('text') or msg.get('message') or ''
                preview_texts.append(f"{speaker}: {text[:50]}{'...' if len(text) > 50 else ''}")
            summary_parts.append(f"预览: {' | '.join(preview_texts)}")
        
        return " | ".join(summary_parts)
    
    def format_for_locomo(self, conversation: Dict) -> str:
        """
        专门为 LoCoMo 数据集格式化对话
        
        确保输出格式与论文要求一致：
        [7 May 2023] User: Hello
        [7 May 2023] Assistant: Hi there
        """
        session_date = conversation.get('session_date') or conversation.get('session_1_date_time', '')
        messages = conversation.get('messages', [])
        
        lines = []
        for turn in messages:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            
            # 关键：每条消息使用 session_date
            lines.append(f"[{session_date}] {speaker}: {text}")
        
        return "\n".join(lines)
