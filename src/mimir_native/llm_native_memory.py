"""
LLM-Native Memory Extraction - 深度依赖大模型的记忆提取

核心思想：
1. 不再提取原子化事实，而是让 LLM 直接生成结构化记忆
2. LLM 自行处理时序推理、冲突解决
3. 存储更丰富的上下文，而非碎片化事实
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RichMemory:
    """
    富记忆结构 - 由 LLM 直接生成
    """
    content: str  # 自然语言描述的记忆
    memory_type: str  # event | attribute | plan | preference | relationship
    entities: List[str]  # 相关实体
    temporal_info: Dict[str, Any]  # 时间信息（由 LLM 解析）
    confidence: float  # LLM 置信度
    source_text: str  # 原始文本
    source_date: Optional[str] = None  # 参考日期
    
    def to_storage_format(self) -> Dict:
        """转换为数据库存储格式"""
        return {
            'content': self.content,
            'memory_type': self.memory_type,
            'entities': json.dumps(self.entities),
            'temporal_info': json.dumps(self.temporal_info),
            'confidence': self.confidence,
            'source_text': self.source_text,
            'source_date': self.source_date,
        }


class LLMNativeMemoryExtractor:
    """
    LLM-Native 记忆提取器
    
    不再尝试提取原子事实，而是让 LLM：
    1. 理解整个对话上下文
    2. 直接生成结构化记忆
    3. 自行处理时间推理
    4. 识别实体关系
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def extract_memories(
        self, 
        conversation: List[Dict], 
        reference_date: Optional[str] = None
    ) -> List[RichMemory]:
        """
        从对话中提取富记忆
        
        Args:
            conversation: 对话列表，每项包含 speaker 和 text
            reference_date: 对话发生的参考日期
            
        Returns:
            List[RichMemory]: 结构化记忆列表
        """
        # 构建对话文本
        dialogue_text = "\n".join([
            f"{msg['speaker']}: {msg['text']}"
            for msg in conversation
        ])
        
        prompt = f"""你是一个专业的记忆提取助手。请从以下对话中提取结构化记忆。

对话内容：
{dialogue_text}

参考日期：{reference_date or '未知'}

任务：
1. 分析整个对话的上下文
2. 提取所有重要信息（事件、人物属性、计划、偏好等）
3. **时间转换**（关键）：
   - 如果参考日期是 "8 May 2023"
   - "yesterday" → absolute_date: "2023-05-07"
   - "last year" → absolute_date: "2022-01-01"
   - "last Saturday" → 基于参考日期计算具体日期
4. 识别实体之间的关系

 temporal_info 填写规则：
- 如果有明确日期（如 "7 May 2023"）→ absolute_date: "2023-05-07"
- 如果只有相对时间（如 "last year"）→ 基于参考日期计算并填写 absolute_date

输出格式（JSON 数组）：
[
  {{
    "content": "记忆的自然语言描述，包含所有关键信息",
    "memory_type": "event|attribute|plan|preference|relationship",
    "entities": ["实体1", "实体2"],
    "temporal_info": {{
      "absolute_date": "2023-05-07",
      "relative_time": "yesterday",
      "time_range": null
    }},
    "confidence": 0.95
  }}
]

要求：
- content 必须自包含，无需上下文就能理解
- **必须将相对时间转换为绝对日期**：如果提到 "yesterday" 且参考日期是 "8 May 2023"，则 absolute_date 必须是 "2023-05-07"
- temporal_info.absolute_date 必须填写转换后的绝对日期（YYYY-MM-DD 格式）
- 不要遗漏任何重要信息
- 置信度低于 0.7 的记忆不要输出

请提取所有记忆："""

        try:
            response = self.llm.invoke_mistral(prompt, max_tokens=2000, temperature=0.0)
            memories_data = json.loads(response)
            
            memories = []
            for m in memories_data:
                # 处理可能缺失的字段
                temporal_info = m.get('temporal_info') or {}
                if temporal_info is None:
                    temporal_info = {}
                
                memory = RichMemory(
                    content=m.get('content', ''),
                    memory_type=m.get('memory_type', 'event'),
                    entities=m.get('entities', []),
                    temporal_info=temporal_info,
                    confidence=m.get('confidence', 0.5),
                    source_text=dialogue_text,
                    source_date=reference_date
                )
                if memory.confidence >= 0.7:
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"LLM 记忆提取失败: {e}")
            return []
    
    def consolidate_memories(
        self, 
        existing_memories: List[RichMemory],
        new_memories: List[RichMemory]
    ) -> List[RichMemory]:
        """
        记忆整合 - 让 LLM 处理冲突和重复
        
        不再使用相似度阈值，而是让 LLM 判断：
        - 是否是同一事件的不同描述
        - 是否需要更新（时间更近的信息）
        - 是否需要合并（互补信息）
        """
        if not existing_memories:
            return new_memories
        
        existing_text = "\n\n".join([
            f"[{i}] {m.content} (Date: {m.temporal_info.get('absolute_date', 'unknown')})"
            for i, m in enumerate(existing_memories)
        ])
        
        new_text = "\n\n".join([
            f"[NEW {i}] {m.content} (Date: {m.temporal_info.get('absolute_date', 'unknown')})"
            for i, m in enumerate(new_memories)
        ])
        
        prompt = f"""你是一个记忆管理助手。请整合新记忆与已有记忆。

已有记忆：
{existing_text}

新记忆：
{new_text}

任务：
1. 识别重复或冲突的记忆
2. 决定如何处理每条新记忆：
   - "add": 作为新记忆添加
   - "update": 更新已有记忆（如果新信息更准确或时间更近）
   - "merge": 与已有记忆合并（互补信息）
   - "skip": 跳过（重复或低置信度）

输出格式（JSON 数组）：
[
  {{
    "new_memory_idx": 0,
    "action": "add|update|merge|skip",
    "existing_memory_idx": 1,  // update/merge 时指定
    "reason": "决策理由"
  }}
]

请给出整合方案："""

        try:
            response = self.llm.invoke_mistral(prompt, max_tokens=1500, temperature=0.0)
            decisions = json.loads(response)
            
            # 根据 LLM 决策处理记忆
            result = list(existing_memories)  # 复制已有记忆
            
            for decision in decisions:
                action = decision['action']
                new_idx = decision['new_memory_idx']
                
                if action == 'add':
                    result.append(new_memories[new_idx])
                elif action == 'update' and 'existing_memory_idx' in decision:
                    old_idx = decision['existing_memory_idx']
                    if old_idx < len(result):
                        result[old_idx] = new_memories[new_idx]
                elif action == 'merge' and 'existing_memory_idx' in decision:
                    old_idx = decision['existing_memory_idx']
                    if old_idx < len(result):
                        # 合并内容
                        merged = self._merge_two_memories(
                            result[old_idx], 
                            new_memories[new_idx]
                        )
                        result[old_idx] = merged
                # skip 不做任何操作
            
            return result
            
        except Exception as e:
            logger.error(f"记忆整合失败: {e}")
            # 失败时直接追加新记忆
            return existing_memories + new_memories
    
    def _merge_two_memories(
        self, 
        old: RichMemory, 
        new: RichMemory
    ) -> RichMemory:
        """合并两个记忆的内容"""
        prompt = f"""合并以下两条记忆的信息：

记忆1：{old.content}
记忆2：{new.content}

输出合并后的自然语言描述："""
        
        try:
            merged_content = self.llm.invoke_mistral(prompt, max_tokens=200, temperature=0.0)
            
            # 使用更详细的时间信息
            temporal_info = old.temporal_info
            if new.temporal_info.get('absolute_date'):
                temporal_info = new.temporal_info
            
            return RichMemory(
                content=merged_content.strip(),
                memory_type=old.memory_type,
                entities=list(set(old.entities + new.entities)),
                temporal_info=temporal_info,
                confidence=max(old.confidence, new.confidence),
                source_text=old.source_text + "\n" + new.source_text,
                source_date=old.source_date or new.source_date
            )
        except:
            # 合并失败，返回新记忆
            return new


class LLMNativeRetriever:
    """
    LLM-Native 检索器
    
    不再依赖向量相似度，而是让 LLM 理解查询意图，
    从记忆中找出最相关的信息。
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def retrieve(
        self, 
        query: str, 
        memories: List[RichMemory],
        top_k: int = 5
    ) -> List[RichMemory]:
        """
        使用 LLM 进行相关性检索
        
        不再使用向量相似度，而是让 LLM 判断每条记忆
        与查询的相关性。
        """
        if not memories:
            return []
        
        # 限制记忆数量，避免超出 token 限制
        max_memories = min(len(memories), 50)
        selected_memories = memories[:max_memories]
        
        memories_text = "\n\n".join([
            f"[{i}] {m.content} (Date: {m.temporal_info.get('absolute_date', 'unknown')})"
            for i, m in enumerate(selected_memories)
        ])
        
        prompt = f"""你是一个检索助手。请找出与查询最相关的记忆。

查询：{query}

候选记忆：
{memories_text}

任务：
1. 分析查询的意图（找什么信息）
2. 评估每条记忆的相关性（0-1 分数）
3. 返回最相关的 {top_k} 条记忆的索引

输出格式（JSON）：
{{
  "relevant_indices": [3, 7, 1],
  "reasoning": "选择这些记忆的理由"
}}

请给出检索结果："""

        try:
            response = self.llm.invoke_mistral(prompt, max_tokens=500, temperature=0.0)
            result = json.loads(response)
            
            indices = result.get('relevant_indices', [])
            
            # 获取相关记忆
            relevant = []
            for idx in indices[:top_k]:
                if 0 <= idx < len(selected_memories):
                    relevant.append(selected_memories[idx])
            
            return relevant
            
        except Exception as e:
            logger.error(f"LLM 检索失败: {e}")
            # 失败时返回前 top_k 条
            return memories[:top_k]


# 使用示例
if __name__ == "__main__":
    from mimir_native.llm_client import BedrockClient
    
    llm = BedrockClient()
    extractor = LLMNativeMemoryExtractor(llm)
    
    # 测试对话
    conversation = [
        {"speaker": "Caroline", "text": "I visited the LGBTQ support group yesterday."},
        {"speaker": "Friend", "text": "How was it?"},
        {"speaker": "Caroline", "text": "It was great, very supportive."},
    ]
    
    memories = extractor.extract_memories(conversation, "8 May 2023")
    
    for m in memories:
        print(f"\n记忆: {m.content}")
        print(f"类型: {m.memory_type}")
        print(f"实体: {m.entities}")
        print(f"时间: {m.temporal_info}")
        print(f"置信度: {m.confidence}")
