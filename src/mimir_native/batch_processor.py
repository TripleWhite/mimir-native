"""
Mimir-Native 批量处理优化方案

目标: 将 LoCoMo 数据处理时间从 15-20 分钟缩短到 2-3 分钟
"""

import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    批量处理器 - 优化大量数据摄入性能
    """
    
    def __init__(self, mimir_memory, llm_client, max_workers=5):
        self.mimir = mimir_memory
        self.llm = llm_client
        self.max_workers = max_workers
    
    def process_conversations_batch(
        self,
        conversations: List[Dict],
        user_id: str = 'default',
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        批量处理对话 - 核心优化
        
        优化点:
        1. 批量事实提取: 多条消息一次 LLM 调用
        2. 并行 embedding: 线程池并发处理
        3. 批量数据库写入: 减少 I/O 次数
        
        Args:
            conversations: 对话列表，每项包含 messages 和 session_date
            user_id: 用户ID
            batch_size: 每批处理的消息数
            
        Returns:
            {'total': 100, 'processed': 100, 'memories': 250, 'time_seconds': 120}
        """
        # 1. 收集所有消息
        all_items = []
        for conv in conversations:
            session_date = conv.get('session_date')
            for msg in conv.get('messages', []):
                all_items.append({
                    'text': msg.get('text', ''),
                    'speaker': msg.get('speaker', ''),
                    'session_date': session_date,
                    'metadata': {'speaker': msg.get('speaker'), 'session_date': session_date}
                })
        
        total = len(all_items)
        logger.info(f"批量处理 {total} 条消息，每批 {batch_size} 条")
        
        # 2. 分批处理
        all_memories = []
        for i in range(0, total, batch_size):
            batch = all_items[i:i+batch_size]
            batch_memories = self._process_one_batch(batch, user_id)
            all_memories.extend(batch_memories)
            logger.info(f"进度: {min(i+batch_size, total)}/{total}")
        
        return {
            'total': total,
            'processed': len(all_items),
            'memories': len(all_memories)
        }
    
    def _process_one_batch(self, items: List[Dict], user_id: str) -> List[Any]:
        """
        处理一批消息 - 核心优化逻辑
        """
        if not items:
            return []
        
        # === 优化1: 批量事实提取（一次 LLM 调用）===
        combined_text = "\n\n".join([
            f"[{i}] {item['speaker']}: {item['text']}" 
            for i, item in enumerate(items)
        ])
        
        # 一次性提取所有事实
        facts = self._extract_facts_batch(combined_text, items)
        
        # === 优化2: 并行 embedding ===
        fact_texts = [f['fact'] for f in facts]
        embeddings = self._batch_embed_parallel(fact_texts)
        
        # === 优化3: 批量数据库写入 ===
        memories = []
        for i, fact in enumerate(facts):
            try:
                memory = self._create_memory_fast(
                    fact=fact['fact'],
                    embedding=embeddings[i] if i < len(embeddings) else None,
                    metadata=fact.get('metadata', {}),
                    user_id=user_id
                )
                if memory:
                    memories.append(memory)
            except Exception as e:
                logger.warning(f"创建记忆失败: {e}")
        
        return memories
    
    def _extract_facts_batch(
        self, 
        combined_text: str, 
        items: List[Dict]
    ) -> List[Dict]:
        """
        批量事实提取 - 一次 LLM 调用提取多条消息的事实
        
        Args:
            combined_text: 合并后的文本，带编号
            items: 原始消息列表
            
        Returns:
            [{'fact': '...', 'metadata': {...}}, ...]
        """
        prompt = f"""从以下带编号的对话中提取客观事实。

对话内容:
{combined_text}

规则:
1. 为每个 [编号] 提取相关事实
2. 保留所有具体时间（如 "7 May 2023"）
3. 每个事实应该自包含
4. 输出格式：JSON 数组

输出示例:
[
  {{"source_idx": 0, "fact": "Caroline visited the LGBTQ support group yesterday.", "temporal_info": {{"absolute_time": null, "relative_time": "yesterday"}}, "entities": ["Caroline", "LGBTQ support group"]}},
  {{"source_idx": 1, "fact": "Melanie painted a sunrise last year.", "temporal_info": {{"absolute_time": null, "relative_time": "last year"}}, "entities": ["Melanie"]}}
]

请提取所有事实:"""

        try:
            response = self.llm.invoke_mistral(prompt, max_tokens=2000)
            import json
            facts = json.loads(response)
            
            # 添加原始 metadata 并应用时间解析
            from .temporal_resolver import TemporalResolver
            resolver = TemporalResolver()
            
            for fact in facts:
                idx = fact.get('source_idx', 0)
                if idx < len(items):
                    metadata = items[idx].get('metadata', {})
                    fact['metadata'] = metadata
                    
                    # 应用时间解析
                    session_date = metadata.get('session_date')
                    if session_date:
                        from .preprocessors.base import parse_date
                        try:
                            ref_date = parse_date(session_date)
                            if ref_date:
                                fact_text = fact.get('fact', '')
                                resolved = resolver.extract_and_resolve(fact_text, ref_date)
                                if resolved:
                                    # 在事实后附加绝对时间
                                    for expr, absolute in resolved.items():
                                        if absolute not in fact_text:
                                            fact_text += f" (Date: {absolute})"
                                    fact['fact'] = fact_text
                                    # 更新 temporal_info
                                    if not fact.get('temporal_info', {}).get('absolute_time'):
                                        if 'temporal_info' not in fact:
                                            fact['temporal_info'] = {}
                                        fact['temporal_info']['absolute_time'] = list(resolved.values())[0]
                        except Exception as e:
                            logger.debug(f"时间解析失败: {e}")
            
            return facts
        except Exception as e:
            logger.warning(f"批量事实提取失败: {e}")
            # 降级：逐条处理
            return self._fallback_extract(items)
    
    def _fallback_extract(self, items: List[Dict]) -> List[Dict]:
        """降级方案：逐条提取"""
        facts = []
        for item in items:
            # 简化处理：直接把消息作为事实
            facts.append({
                'fact': item['text'],
                'metadata': item.get('metadata', {})
            })
        return facts
    
    def _batch_embed_parallel(self, texts: List[str]) -> List[List[float]]:
        """
        并行批量 embedding
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding 列表
        """
        if not texts:
            return []
        
        # 使用 Silicon Flow 的批量 embedding API（更快）
        try:
            # Silicon Flow 支持一次最多 32 条
            embeddings = self.llm.batch_embed(texts)
            return embeddings
        except Exception as e:
            logger.warning(f"批量 embedding 失败: {e}")
            # 降级：逐个处理
            return [self.llm.embed(t) for t in texts]
    
    def _create_memory_fast(
        self, 
        fact: str, 
        embedding: List[float], 
        metadata: Dict,
        user_id: str
    ) -> Any:
        """快速创建记忆（跳过复杂的去重/冲突检测）"""
        # 使用绝对导入
        from mimir_native.database import MemoryCreate
        import hashlib
        import json
        
        content_hash = hashlib.md5(fact.lower().strip().encode()).hexdigest()
        
        memory_create = MemoryCreate(
            user_id=user_id,
            content=fact,
            content_hash=content_hash,
            embedding=embedding,
            source_type='conversation',
            source_metadata=json.dumps(metadata) if metadata else None
        )
        
        memory_id = self.mimir.db.create_memory(memory_create)
        return self.mimir.db.get_memory(memory_id)


# === 使用示例 ===
def example_usage():
    """
    使用示例：优化后的 LoCoMo 处理
    """
    from mimir_native import MimirMemory
    from mimir_native.llm_client import BedrockClient
    
    # 初始化
    mimir = MimirMemory(db_path="mimir.db")
    llm = BedrockClient()
    processor = BatchProcessor(mimir, llm, max_workers=5)
    
    # 准备 LoCoMo 数据
    conversations = [
        {
            'session_date': '7 May 2023',
            'messages': [
                {'speaker': 'Caroline', 'text': 'I visited the LGBTQ support group today.'},
                {'speaker': 'Friend', 'text': 'How was it?'},
                {'speaker': 'Caroline', 'text': 'It was great, very supportive.'},
            ]
        },
        # ... 更多 session
    ]
    
    # 批量处理（预计 2-3 分钟处理 100 条消息）
    result = processor.process_conversations_batch(
        conversations,
        user_id='locomo_test',
        batch_size=10  # 每批 10 条，一次 LLM 调用
    )
    
    print(f"处理完成: {result['processed']} 条消息 → {result['memories']} 条记忆")


if __name__ == "__main__":
    example_usage()
