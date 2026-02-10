"""
Mimir Memory V2 - Multimodal Preprocessors
多模态预处理层 - 将 PDF/图片/音频/对话转换为结构化文本
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
import logging

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class RawContent:
    """
    预处理后的统一内容格式
    
    Attributes:
        text: 提取的文本内容
        summary: 内容摘要（可选）
        chunks: 语义分块列表
        metadata: 元数据字典
        occurred_at: 关键发生时间（用于时序处理）
    """
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


class BasePreprocessor(ABC):
    """
    多模态预处理器基类
    
    所有具体预处理器必须继承此类并实现 process 和 supports 方法
    """
    
    @abstractmethod
    def process(self, content: Any, metadata: dict) -> RawContent:
        """
        处理内容并返回标准化的 RawContent
        
        Args:
            content: 输入内容（类型取决于具体处理器）
            metadata: 元数据字典
            
        Returns:
            RawContent: 标准化的内容对象
        """
        pass
    
    @abstractmethod
    def supports(self, content_type: str) -> bool:
        """
        检查此处理器是否支持指定的内容类型
        
        Args:
            content_type: 内容类型标识符
            
        Returns:
            bool: 是否支持
        """
        pass
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        将文本分割成语义块
        
        Args:
            text: 输入文本
            chunk_size: 每块的最大字符数
            overlap: 块之间的重叠字符数
            
        Returns:
            List[str]: 文本块列表
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # 尝试在句子边界处分割
            if end < text_len:
                # 查找最近的句子结束符
                for sep in ['.\n', '。\n', '!\n', '?\n', '. ', '。', '!', '?', '\n\n', '\n']:
                    pos = text.rfind(sep, start, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break
            
            chunks.append(text[start:end].strip())
            start = end - overlap if end < text_len else end
        
        return chunks
    
    def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        生成文本摘要（简单实现，可扩展为使用 LLM）
        
        Args:
            text: 输入文本
            max_length: 摘要最大长度
            
        Returns:
            str: 摘要文本
        """
        if not text:
            return ""
        
        # 简单摘要：取前 max_length 个字符
        if len(text) <= max_length:
            return text
        
        # 尝试在句子边界截断
        truncated = text[:max_length]
        for sep in ['. ', '。', '!', '?', '\n']:
            pos = truncated.rfind(sep)
            if pos > max_length * 0.5:  # 至少保留一半
                return truncated[:pos + 1].strip() + "..."
        
        return truncated.strip() + "..."


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    解析日期字符串为 datetime 对象
    
    支持多种常见格式：
    - "7 May 2023"
    - "2023-05-07"
    - "2023/05/07"
    - "May 7, 2023"
    - ISO 格式
    
    Args:
        date_str: 日期字符串
        
    Returns:
        Optional[datetime]: 解析后的 datetime 或 None
    """
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
        "%Y-%m-%dT%H:%M:%S",  # ISO 格式
        "%Y-%m-%d %H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    logger.warning(f"无法解析日期: {date_str}")
    return None
