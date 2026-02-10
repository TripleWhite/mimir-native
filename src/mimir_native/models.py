"""
Mimir Memory V2 - Models

数据模型定义，包含 Fact 等 Memory Agent 专用模型
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Fact:
    """事实数据模型 - 从文本中提取的结构化事实"""
    fact: str  # 事实陈述
    temporal_info: Dict[str, Any] = field(default_factory=dict)  # 时间信息
    entities: List[str] = field(default_factory=list)  # 涉及的实体
    fact_type: str = "event"  # event | preference | relationship | work
    confidence: float = 1.0  # 置信度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'fact': self.fact,
            'temporal_info': self.temporal_info,
            'entities': self.entities,
            'fact_type': self.fact_type,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fact':
        """从字典创建"""
        return cls(
            fact=data.get('fact', ''),
            temporal_info=data.get('temporal_info', {}),
            entities=data.get('entities', []),
            fact_type=data.get('fact_type', 'event'),
            confidence=data.get('confidence', 1.0)
        )
    
    def get_content_hash_input(self) -> str:
        """获取用于计算内容哈希的输入字符串"""
        # 标准化事实文本，用于相似度比较
        return self.fact.lower().strip()


@dataclass 
class ConflictResolutionResult:
    """冲突解决结果"""
    is_conflict: bool  # 是否检测到冲突
    resolution: str  # "keep_existing" | "update" | "merge" | "new"
    reason: str  # 冲突原因说明
    confidence: float = 1.0  # 解决置信度


@dataclass
class DeduplicationResult:
    """去重结果"""
    is_duplicate: bool  # 是否是重复
    similar_memories: List[Dict[str, Any]]  # 相似的记忆
    similarity_scores: Dict[str, float]  # 相似度分数
    best_match_id: Optional[str] = None  # 最佳匹配的ID
    best_match_score: float = 0.0  # 最佳匹配分数
