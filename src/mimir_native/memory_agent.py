"""
Mimir Memory V2 - Memory Agent

核心智能层，负责从预处理后的内容中提取结构化事实，
保持记忆库干净、无重复、无冲突。
"""

import json
import hashlib
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from difflib import SequenceMatcher

import numpy as np

from .database import MimirDatabase, MemoryCreate, Memory
from .models import Fact, ConflictResolutionResult, DeduplicationResult
from .llm_client import BedrockClient, create_llm_client

logger = logging.getLogger(__name__)


class MemoryAgent:
    """
    Memory Agent - 核心智能层
    
    职责：
    1. 从原始内容中提取事实
    2. 去重检测
    3. 冲突解决
    4. 存储原子化记忆
    """
    
    def __init__(self, db: MimirDatabase, llm_client: BedrockClient = None,
                 similarity_threshold: float = 0.75):
        """
        初始化 Memory Agent
        
        Args:
            db: MimirDatabase 实例
            llm_client: BedrockClient 实例（可选，会自动创建）
            similarity_threshold: 相似度阈值，默认 0.75（降低以减少误判）
        """
        self.db = db
        self.llm = llm_client or create_llm_client()
        self.similarity_threshold = similarity_threshold
        
        # 检查 LLM 客户端可用性
        if not self.llm.is_available():
            logger.warning("LLM 客户端不可用，将使用降级模式")
    
    def process_raw_content(self, raw_content: 'RawContent') -> List[Memory]:
        """
        主流程: 原始内容 → 原子化记忆

        优化：使用批量 embedding 减少 API 调用次数

        Args:
            raw_content: 原始内容对象

        Returns:
            List[Memory]: 创建/更新的记忆列表
        """
        # 获取文本内容 (RawContent.text 或 raw_text)
        text_content = getattr(raw_content, 'text', None) or getattr(raw_content, 'raw_text', '')
        if not text_content:
            logger.warning("原始内容为空，跳过处理")
            return []

        # 1. 分块处理（如果内容过长）
        chunks = self._chunk_content(text_content)
        all_memories = []
        all_facts = []  # 收集所有事实用于批量处理

        # 解析元数据
        metadata = {}
        if raw_content.metadata:
            if isinstance(raw_content.metadata, dict):
                metadata = raw_content.metadata
            elif isinstance(raw_content.metadata, str):
                try:
                    metadata = json.loads(raw_content.metadata)
                except json.JSONDecodeError:
                    logger.warning(f"无法解析元数据: {raw_content.metadata}")

        # 2. 对每个块提取事实
        for chunk in chunks:
            facts = self._extract_facts(chunk, metadata)
            all_facts.extend(facts)

        # 3. 批量获取所有事实的 embedding（关键优化！）
        fact_embeddings = {}
        if all_facts and self.llm.is_available():
            try:
                fact_texts = [f.fact[:500] for f in all_facts]
                embeddings = self.llm.batch_embed(fact_texts)
                for i, fact in enumerate(all_facts):
                    if i < len(embeddings) and embeddings[i]:
                        fact_embeddings[id(fact)] = embeddings[i]
                logger.debug(f"批量获取 {len(all_facts)} 个事实的 embedding")
            except Exception as e:
                logger.warning(f"批量 embedding 失败: {e}")

        # 4. 处理每个事实（使用预计算的 embedding）
        for fact in all_facts:
            embedding = fact_embeddings.get(id(fact))
            memory = self._process_fact(fact, raw_content, embedding)
            if memory:
                all_memories.append(memory)

        # 5. 标记原始内容为已处理
        memory_ids = [m.id for m in all_memories]
        self.db.update_raw_content(
            raw_content.id,
            {
                'processed': True,
                'extracted_memory_ids': json.dumps(memory_ids)
            }
        )

        logger.info(f"处理完成: {len(chunks)} 个块 → {len(all_facts)} 个事实 → {len(all_memories)} 条记忆")
        return all_memories
    
    def process_raw_content_batch(self, raw_contents: List['RawContent']) -> Dict[str, List[Memory]]:
        """
        批量处理原始内容
        
        Args:
            raw_contents: 原始内容列表
            
        Returns:
            Dict[str, List[Memory]]: 每个 raw_content_id 对应的记忆列表
        """
        results = {}
        for content in raw_contents:
            try:
                memories = self.process_raw_content(content)
                results[content.id] = memories
            except Exception as e:
                logger.error(f"批量处理失败 for {content.id}: {e}")
                results[content.id] = []
        return results
    
    def _chunk_content(self, text: str, max_chunk_size: int = 2000, 
                       overlap: int = 200) -> List[str]:
        """
        将长文本分块
        
        Args:
            text: 原始文本
            max_chunk_size: 每块最大字符数
            overlap: 块间重叠字符数
            
        Returns:
            List[str]: 文本块列表
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # 尝试在句子边界分割
            if end < len(text):
                # 查找最近的句子结束位置
                for sep in ['.\n', '。', '. ', '? ', '! ', '\n\n']:
                    pos = text.rfind(sep, start, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return chunks
    
    def _extract_facts(self, text: str, metadata: dict = None) -> List[Fact]:
        """
        使用 Bedrock Mistral 提取事实（中国 IP 兼容）
        
        关键: 保留所有时间信息！
        
        Args:
            text: 文本内容
            metadata: 元数据（可能包含日期信息）
            
        Returns:
            List[Fact]: 提取的事实列表
        """
        if not self.llm.is_available():
            logger.warning("LLM 不可用，使用简单提取模式")
            return self._fallback_extract_facts(text, metadata)
        
        try:
            fact_dicts = self.llm.extract_facts(text, metadata or {})
            facts = []
            
            for fd in fact_dicts:
                try:
                    fact = Fact(
                        fact=fd.get('fact', ''),
                        temporal_info=fd.get('temporal_info', {}),
                        entities=fd.get('entities', []),
                        fact_type=fd.get('fact_type', 'event'),
                        confidence=fd.get('confidence', 1.0)
                    )
                    if fact.fact:  # 过滤空事实
                        facts.append(fact)
                except Exception as e:
                    logger.warning(f"解析事实失败: {e}, data: {fd}")
            
            return facts
            
        except Exception as e:
            logger.error(f"事实提取失败: {e}")
            return self._fallback_extract_facts(text, metadata)
    
    def _fallback_extract_facts(self, text: str, metadata: dict = None) -> List[Fact]:
        """
        降级模式：简单事实提取
        
        当 LLM 不可用时使用
        """
        facts = []
        
        # 简单的句子分割
        sentences = [s.strip() for s in text.replace('。', '.').split('.') if len(s.strip()) > 10]
        
        for sentence in sentences[:5]:  # 限制数量
            # 尝试提取时间信息
            temporal_info = self._extract_temporal_info(sentence)
            
            fact = Fact(
                fact=sentence,
                temporal_info=temporal_info,
                entities=[],  # 降级模式不提取实体
                fact_type='other',
                confidence=0.5
            )
            facts.append(fact)
        
        return facts
    
    def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """
        简单的时间信息提取
        
        Args:
            text: 文本
            
        Returns:
            Dict: 时间信息
        """
        import re
        
        temporal_info = {
            'absolute_time': None,
            'relative_time': None,
            'time_mentions': []
        }
        
        # 匹配常见日期格式
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',  # DD Month YYYY
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    temporal_info['time_mentions'].extend(matches[0])
                else:
                    temporal_info['time_mentions'].extend(matches)
        
        # 相对时间
        relative_patterns = [
            r'\b(yesterday|today|tomorrow|last week|next week|last month|next month)\b',
            r'\b(last\s+\w+|next\s+\w+ ago)\b',
        ]
        
        for pattern in relative_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                temporal_info['relative_time'] = matches[0] if isinstance(matches[0], str) else matches[0][0]
                temporal_info['time_mentions'].extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
        
        return temporal_info
    
    def _process_fact(self, fact: Fact, raw_content: 'RawContent',
                      precomputed_embedding: List[float] = None) -> Optional[Memory]:
        """
        处理单个事实：去重和冲突解决

        Args:
            fact: 事实对象
            raw_content: 原始内容
            precomputed_embedding: 预计算的 embedding（可选，用于批量优化）

        Returns:
            Optional[Memory]: 创建或更新的记忆
        """
        # 1. 计算内容哈希
        content_hash = self._compute_content_hash(fact)

        # 2. 查找相似记忆（向量+关键词）
        similar = self._find_similar_memories(fact, precomputed_embedding)

        # 3. 冲突检测和解决
        for match in similar:
            if match['similarity'] > self.similarity_threshold:
                existing = match['memory']

                # 检查是否真正冲突
                conflict_result = self._check_conflict(existing, fact)

                if conflict_result.is_conflict:
                    if conflict_result.resolution == 'update':
                        # 更新时间更近的事实
                        return self._update_memory(existing, fact, raw_content, precomputed_embedding)
                    elif conflict_result.resolution == 'merge':
                        # 合并事实
                        return self._merge_memory(existing, fact, raw_content, precomputed_embedding)
                    else:
                        # 保留已有事实
                        logger.debug(f"保留已有记忆: {existing.id}")
                        return existing
                else:
                    # 相似但不是冲突，视为重复
                    logger.debug(f"发现重复: {existing.id}")
                    return None

        # 4. 没有相似记忆，创建新记忆
        return self._create_memory(fact, raw_content, content_hash, precomputed_embedding)
    
    def _compute_content_hash(self, fact: Fact) -> str:
        """
        计算事实的内容哈希
        
        Args:
            fact: 事实对象
            
        Returns:
            str: 哈希值
        """
        content = fact.get_content_hash_input()
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
    
    def _find_similar_memories(self, fact: Fact, precomputed_embedding: List[float] = None,
                               threshold: float = None) -> List[Dict[str, Any]]:
        """
        查找相似记忆（混合检索: 向量 + FTS5）

        优化：使用预计算的 embedding 避免重复 API 调用

        Args:
            fact: 事实对象
            precomputed_embedding: 预计算的 embedding（可选）
            threshold: 相似度阈值

        Returns:
            List[Dict]: 相似的记忆及相似度
        """
        threshold = threshold or self.similarity_threshold
        results = []
        seen_ids = set()

        # 1. 向量检索（如果 LLM 可用）
        if self.llm.is_available():
            try:
                # 使用预计算的 embedding 或重新计算
                embedding = precomputed_embedding
                if embedding is None:
                    embedding = self.llm.embed(fact.fact[:500])

                if embedding:
                    vec_results = self.db.vector_search(embedding, top_k=10)
                    for r in vec_results:
                        if r['memory_id'] not in seen_ids:
                            # 转换距离为相似度（余弦相似度）
                            similarity = 1 - r['distance']
                            if similarity >= threshold - 0.1:  # 稍低的阈值以获取更多候选
                                results.append({
                                    'memory': r['memory'],
                                    'similarity': similarity,
                                    'method': 'vector'
                                })
                                seen_ids.add(r['memory_id'])
            except Exception as e:
                logger.warning(f"向量检索失败: {e}")

        # 2. 全文检索
        try:
            # 提取关键词（简单实现：取前 3 个实体或前 10 个字）
            keywords = ' '.join(fact.entities[:3]) if fact.entities else fact.fact[:30]
            fts_results = self.db.fts_search(keywords, top_k=10)

            for r in fts_results:
                if r['memory_id'] not in seen_ids:
                    # 计算文本相似度
                    similarity = self._text_similarity(fact.fact, r['memory'].content)
                    if similarity >= threshold - 0.1:
                        results.append({
                            'memory': r['memory'],
                            'similarity': similarity,
                            'method': 'fts'
                        })
                        seen_ids.add(r['memory_id'])
        except Exception as e:
            logger.warning(f"全文检索失败: {e}")

        # 3. 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:5]  # 最多返回 5 个
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            float: 相似度分数 (0-1)
        """
        # 使用 SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _check_conflict(self, existing: Memory, new_fact: Fact) -> ConflictResolutionResult:
        """
        使用 LLM 判断事实是否冲突
        
        Args:
            existing: 已有记忆
            new_fact: 新事实
            
        Returns:
            ConflictResolutionResult: 冲突检测结果
        """
        # 简单规则优先
        similarity = self._text_similarity(existing.content, new_fact.fact)
        
        # 如果相似度不够高，可能不是同一主题
        if similarity < 0.6:
            return ConflictResolutionResult(
                is_conflict=False,
                resolution="new",
                reason="相似度较低，视为不同事实",
                confidence=similarity
            )
        
        # 使用 LLM 检测冲突
        if self.llm.is_available():
            try:
                result = self.llm.check_conflict(
                    existing.content,
                    new_fact.fact,
                    existing.valid_at.isoformat() if existing.valid_at else None,
                    new_fact.temporal_info.get('absolute_time') if new_fact.temporal_info else None
                )
                
                return ConflictResolutionResult(
                    is_conflict=result.get('is_conflict', False),
                    resolution=result.get('resolution', 'new'),
                    reason=result.get('reason', ''),
                    confidence=result.get('confidence', 0.5)
                )
            except Exception as e:
                logger.warning(f"LLM 冲突检测失败: {e}")
        
        # 降级模式：基于时间判断
        return self._fallback_conflict_check(existing, new_fact)
    
    def _fallback_conflict_check(self, existing: Memory, new_fact: Fact) -> ConflictResolutionResult:
        """
        降级模式：基于规则的冲突检测
        """
        # 如果新事实有明确的时间且更近，优先更新
        new_time = new_fact.temporal_info.get('absolute_time') if new_fact.temporal_info else None
        
        if new_time and existing.valid_at:
            try:
                new_dt = datetime.fromisoformat(new_time.replace('Z', '+00:00'))
                if new_dt > existing.valid_at:
                    return ConflictResolutionResult(
                        is_conflict=True,
                        resolution="update",
                        reason="新事实时间更近（降级模式判断）",
                        confidence=0.6
                    )
            except:
                pass
        
        # 默认视为新事实
        return ConflictResolutionResult(
            is_conflict=False,
            resolution="new",
            reason="降级模式：默认作为新事实",
            confidence=0.5
        )
    
    def _create_memory(self, fact: Fact, raw_content: 'RawContent',
                       content_hash: str, precomputed_embedding: List[float] = None) -> Memory:
        """
        创建新记忆

        优化：使用预计算的 embedding 避免重复 API 调用

        Args:
            fact: 事实对象
            raw_content: 原始内容
            content_hash: 内容哈希
            precomputed_embedding: 预计算的 embedding（可选）

        Returns:
            Memory: 创建的记忆
        """
        # 使用预计算的 embedding 或生成新的
        embedding = precomputed_embedding
        if embedding is None and self.llm.is_available():
            try:
                embedding = self.llm.embed(fact.fact[:500])
            except Exception as e:
                logger.warning(f"嵌入生成失败: {e}")

        # 解析时间
        valid_at = self._parse_time(fact.temporal_info.get('absolute_time')) if fact.temporal_info else None

        memory_create = MemoryCreate(
            user_id=raw_content.user_id,
            content=fact.fact,
            content_hash=content_hash,
            embedding=embedding,
            valid_at=valid_at,
            valid_at_confidence=fact.confidence,
            temporal_tags=json.dumps(fact.temporal_info),
            source_type=raw_content.content_type,
            source_id=raw_content.id,
            source_metadata=json.dumps({
                'entities': fact.entities,
                'fact_type': fact.fact_type,
                'raw_metadata': raw_content.metadata
            })
        )

        memory_id = self.db.create_memory(memory_create)
        logger.info(f"创建新记忆: {memory_id}")

        return self.db.get_memory(memory_id)
    
    def _update_memory(self, existing: Memory, new_fact: Fact,
                       raw_content: 'RawContent', precomputed_embedding: List[float] = None) -> Memory:
        """
        更新已有记忆

        优化：使用预计算的 embedding 避免重复 API 调用

        Args:
            existing: 已有记忆
            new_fact: 新事实
            raw_content: 原始内容
            precomputed_embedding: 预计算的 embedding（可选）

        Returns:
            Memory: 更新后的记忆
        """
        # 使用预计算的 embedding 或生成新的
        embedding = precomputed_embedding
        if embedding is None and self.llm.is_available():
            try:
                embedding = self.llm.embed(new_fact.fact[:500])
            except Exception as e:
                logger.warning(f"嵌入生成失败: {e}")

        # 解析时间
        valid_at = self._parse_time(new_fact.temporal_info.get('absolute_time')) if new_fact.temporal_info else None

        # 更新记忆
        self.db.update_memory(existing.id, {
            'content': new_fact.fact,
            'content_hash': self._compute_content_hash(new_fact),
            'embedding': embedding,
            'valid_at': valid_at,
            'valid_at_confidence': new_fact.confidence,
            'temporal_tags': json.dumps(new_fact.temporal_info),
            'source_id': raw_content.id,
            'source_metadata': json.dumps({
                'entities': new_fact.entities,
                'fact_type': new_fact.fact_type,
                'previous_content': existing.content,
                'update_reason': 'conflict_resolution'
            }),
            'version': existing.version + 1
        })

        logger.info(f"更新记忆: {existing.id} (v{existing.version} → v{existing.version + 1})")
        return self.db.get_memory(existing.id)
    
    def _merge_memory(self, existing: Memory, new_fact: Fact,
                      raw_content: 'RawContent', precomputed_embedding: List[float] = None) -> Memory:
        """
        合并新旧事实

        优化：使用预计算的 embedding 避免重复 API 调用

        Args:
            existing: 已有记忆
            new_fact: 新事实
            raw_content: 原始内容
            precomputed_embedding: 预计算的 embedding（可选）

        Returns:
            Memory: 合并后的记忆
        """
        # 简单合并：将两个事实连接起来
        merged_content = f"{existing.content}; {new_fact.fact}"

        # 合并时间信息
        try:
            existing_temporal = json.loads(existing.temporal_tags) if existing.temporal_tags else {}
        except:
            existing_temporal = {}

        merged_temporal = {
            'merged_from': [existing_temporal, new_fact.temporal_info],
            'absolute_time': (new_fact.temporal_info.get('absolute_time') if new_fact.temporal_info else None) or existing_temporal.get('absolute_time')
        }

        # 使用预计算的 embedding（如果它是合并内容的 embedding）或生成新的
        embedding = precomputed_embedding
        if embedding is None and self.llm.is_available():
            try:
                embedding = self.llm.embed(merged_content[:500])
            except Exception as e:
                logger.warning(f"嵌入生成失败: {e}")

        self.db.update_memory(existing.id, {
            'content': merged_content,
            'embedding': embedding,
            'temporal_tags': json.dumps(merged_temporal),
            'version': existing.version + 1
        })

        logger.info(f"合并记忆: {existing.id}")
        return self.db.get_memory(existing.id)
    
    def _parse_time(self, time_str: Optional[str]) -> Optional[datetime]:
        """
        解析时间字符串
        
        Args:
            time_str: 时间字符串
            
        Returns:
            Optional[datetime]: 解析后的时间
        """
        if not time_str:
            return None
        
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d',
            '%d %B %Y',
            '%B %d, %Y',
            '%B %d %Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        return None
