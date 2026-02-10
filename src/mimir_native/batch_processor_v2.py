"""
Mimir-Native 批量处理优化方案 - 修复版

修复：
1. 强制时间转换（yesterday -> 7 May 2023）
2. 确保简单属性提取（single, identity等）
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
        """处理一批消息"""
        if not items:
            return []
        
        # 批量事实提取
        combined_text = "\n\n".join([
            f"[{i}] {item['speaker']}: {item['text']}" 
            for i, item in enumerate(items)
        ])
        
        facts = self._extract_facts_batch(combined_text, items)
        
        # 并行 embedding
        fact_texts = [f['fact'] for f in facts]
        embeddings = self._batch_embed_parallel(fact_texts)
        
        # 批量数据库写入
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
    
    def _extract_facts_batch(self, combined_text: str, items: List[Dict]) -> List[Dict]:
        """
        批量事实提取 - 强制时间转换版本
        """
        reference_date = items[0].get('session_date', 'unknown') if items else 'unknown'
        
        prompt = f"""从以下带编号的对话中提取客观事实和人物属性。

对话内容:
{combined_text}

参考日期：{reference_date}

规则：
1. 为每个 [编号] 提取相关事实
2. **提取所有人物属性**：身份、职业、关系状态、兴趣爱好等（不要遗漏！）
3. 保留原始时间表达式（如 "yesterday", "last year"），后处理会转换
4. 不要遗漏简单属性陈述
5. 输出格式：JSON 数组

输出示例：
[
  {{"source_idx": 0, "fact": "Caroline visited the LGBTQ support group yesterday.", "entities": ["Caroline"], "fact_type": "event"}},
  {{"source_idx": 1, "fact": "Caroline is a transgender woman.", "entities": ["Caroline"], "fact_type": "personal_info"}},
  {{"source_idx": 2, "fact": "Melanie is single.", "entities": ["Melanie"], "fact_type": "personal_info"}},
  {{"source_idx": 3, "fact": "Melanie painted a sunrise last year.", "entities": ["Melanie"], "fact_type": "event"}}
]

请提取所有事实："""

        try:
            response = self.llm.invoke_mistral(prompt, max_tokens=2000, temperature=0.0)
            import json
            facts = json.loads(response)
            
            # 初始化时间后处理器
            from .temporal_post_processor import TemporalPostProcessor
            temporal_pp = TemporalPostProcessor()
            
            # 添加 metadata 并应用时间后处理
            for fact in facts:
                idx = fact.get('source_idx', 0)
                if idx < len(items):
                    metadata = items[idx].get('metadata', {})
                    fact['metadata'] = metadata
                    
                    # 强制时间转换
                    session_date = metadata.get('session_date')
                    if session_date:
                        original_fact = fact.get('fact', '')
                        processed_fact = temporal_pp.process_fact(original_fact, session_date)
                        if processed_fact != original_fact:
                            logger.debug(f"时间转换: '{original_fact}' -> '{processed_fact}'")
                            fact['fact'] = processed_fact
            
            return facts
            
        except Exception as e:
            logger.warning(f"批量事实提取失败: {e}")
            return self._fallback_extract(items)
    
    def _fallback_extract(self, items: List[Dict]) -> List[Dict]:
        """降级方案：逐条提取 + 时间后处理"""
        from .temporal_post_processor import TemporalPostProcessor
        temporal_pp = TemporalPostProcessor()
        
        facts = []
        for item in items:
            text = item['text']
            metadata = item.get('metadata', {})
            session_date = metadata.get('session_date')
            
            # 应用时间后处理
            if session_date:
                text = temporal_pp.process_fact(text, session_date)
            
            facts.append({
                'fact': text,
                'metadata': metadata
            })
        return facts
    
    def _batch_embed_parallel(self, texts: List[str]) -> List[List[float]]:
        """并行批量 embedding"""
        if not texts:
            return []
        
        try:
            embeddings = self.llm.batch_embed(texts)
            return embeddings
        except Exception as e:
            logger.warning(f"批量 embedding 失败: {e}")
            return [self.llm.embed(t) for t in texts]
    
    def _create_memory_fast(self, fact: str, embedding: List[float], metadata: Dict, user_id: str) -> Any:
        """快速创建记忆"""
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
