"""
Mimir Memory V2 - Hybrid Retriever

混合检索器 - 整合多种检索信号，为下游 LLM 提供最相关的上下文

融合信号：
- 向量相似度 (语义)
- FTS5 全文检索 (关键词)
- 时序相关性 (时间 proximity)
- 图谱关系 (实体关联)
- 访问频率 (热度)
"""

import re
import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型枚举"""
    FACTUAL = "factual"      # 事实查询
    TEMPORAL = "temporal"    # 时序查询
    MULTI_HOP = "multi_hop"  # 多跳推理
    HYBRID = "hybrid"        # 混合


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    memory: Any  # Memory 对象
    final_score: float
    vector_score: float = 0.0
    fts_score: float = 0.0
    temporal_score: float = 0.0
    graph_score: float = 0.0
    recency_score: float = 0.0
    source: str = "unknown"


class HybridRetriever:
    """
    混合检索器
    
    融合信号：
    - 向量相似度 (语义)
    - FTS5 全文检索 (关键词)
    - 时序相关性 (时间 proximity)
    - 图谱关系 (实体关联)
    - 访问频率 (热度)
    """
    
    def __init__(
        self,
        db: Any,  # MimirDatabase
        kg: Any,  # TemporalKnowledgeGraph
        llm_client: Any = None,
        weights: Dict[str, Dict[str, float]] = None
    ):
        """
        初始化混合检索器
        
        Args:
            db: MimirDatabase 实例
            kg: TemporalKnowledgeGraph 实例
            llm_client: 可选，用于生成 embedding
            weights: 自定义权重配置
        """
        self.db = db
        self.kg = kg
        self.llm_client = llm_client
        
        # 默认权重配置 - 优化 for LoCoMo
        self.weights = weights or {
            QueryType.FACTUAL: {
                'vector': 0.6,  # 提高向量权重
                'fts': 0.3,     # 提高 FTS 权重
                'temporal': 0.0,
                'graph': 0.0,
                'recency': 0.1
            },
            QueryType.TEMPORAL: {
                'vector': 0.4,
                'fts': 0.3,
                'temporal': 0.2,
                'graph': 0.0,
                'recency': 0.1
            },
            QueryType.MULTI_HOP: {
                'vector': 0.4,
                'fts': 0.3,
                'temporal': 0.0,
                'graph': 0.2,
                'recency': 0.1
            }
        }
    
    def retrieve(
        self,
        query: str,
        user_id: str,
        query_type: QueryType = QueryType.HYBRID,
        top_k: int = 10,
        filters: Dict = None
    ) -> List[RetrievalResult]:
        """
        主检索接口
        
        Args:
            query: 查询文本
            user_id: 用户 ID
            query_type: 查询类型（自动检测或指定）
            top_k: 返回结果数
            filters: 可选过滤条件
        
        Returns:
            融合排序后的结果列表
        """
        # 1. 自动检测查询类型
        if query_type == QueryType.HYBRID:
            query_type = self._classify_query(query)
            logger.info(f"查询类型自动检测: {query_type.value}")
        
        # 2. 并行检索（各信号独立召回）
        vector_results = self._vector_search(query, user_id, top_k * 3, filters)  # 增加召回数量
        fts_results = self._fts_search(query, user_id, top_k * 3, filters)  # 增加召回数量
        
        temporal_results = []
        graph_results = []
        
        if query_type == QueryType.TEMPORAL:
            # 解析时序查询
            entity, time_constraint = self._parse_temporal_query(query)
            if entity:
                temporal_results = self._temporal_search(
                    entity, time_constraint, user_id, top_k * 2
                )
        
        elif query_type == QueryType.MULTI_HOP:
            # 解析多跳查询
            entities = self._extract_entities(query)
            if len(entities) >= 2:
                paths = self.kg.multi_hop_query(entities[0], entities[1], max_hops=3)
                graph_results = self._paths_to_memories(paths, user_id)
        
        # 3. 融合排序
        merged = self._merge_results(
            vector_results, fts_results, temporal_results, graph_results
        )
        
        # 4. 加权评分
        weights = self.weights.get(query_type, self.weights[QueryType.FACTUAL])
        ranked = self._rank_with_weights(merged, weights, query_type)
        
        # 5. 过滤低相关性结果（避免干扰）
        ranked = [r for r in ranked if r.final_score > 0.05]  # 降低阈值到 0.05
        
        # 6. 应用过滤条件
        if filters:
            ranked = self._apply_filters(ranked, filters)
        
        # 7. 返回 top_k
        return ranked[:top_k]
    
    def _classify_query(self, query: str) -> QueryType:
        """
        自动分类查询类型
        
        Args:
            query: 查询文本
            
        Returns:
            QueryType 枚举值
        """
        query_lower = query.lower()
        
        # 时序关键词
        temporal_indicators = [
            'when', 'before', 'after', 'yesterday', 'last', 'ago',
            'what time', 'which day', 'date', 'recently', 'earlier',
            'previously', 'then', 'at that time', 'during'
        ]
        temporal_score = sum(1 for w in temporal_indicators if w in query_lower)
        
        # 多跳关键词
        multi_hop_indicators = [
            "'s", 'where', 'who', 'company', 'work', 'and then',
            'what did', 'before the', 'after the', 'through',
            'connection', 'related to', 'associated with', 'linked'
        ]
        multi_hop_score = sum(1 for w in multi_hop_indicators if w in query_lower)
        
        # 判定逻辑
        if temporal_score > 0 and temporal_score >= multi_hop_score:
            return QueryType.TEMPORAL
        elif multi_hop_score > 0 and multi_hop_score > temporal_score:
            return QueryType.MULTI_HOP
        
        return QueryType.FACTUAL
    
    def _parse_temporal_query(self, query: str) -> Tuple[Optional[str], Optional[Dict]]:
        """
        解析时序查询，提取实体和时间约束
        
        Args:
            query: 查询文本
            
        Returns:
            (entity_name, time_constraint_dict) 或 (None, None)
        """
        query_lower = query.lower()
        
        # 简单的时间约束识别
        time_constraint = None
        
        # 识别时间类型
        if 'before' in query_lower:
            time_type = 'before'
        elif 'after' in query_lower:
            time_type = 'after'
        elif 'when' in query_lower or 'what time' in query_lower:
            time_type = 'at'
        else:
            time_type = 'at'
        
        # 尝试从图谱中查找提及的实体
        # 简单实现：查找图谱中存在的实体名称
        entity_name = None
        for node_id, node_data in self.kg.graph.nodes(data=True):
            name = node_data.get('name', '').lower()
            if name and name in query_lower:
                entity_name = node_id
                break
        
        if entity_name:
            time_constraint = {
                'type': time_type,
                'time': datetime.now()  # 默认使用当前时间，实际应解析具体时间
            }
        
        return entity_name, time_constraint
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        从查询中提取实体 ID
        
        Args:
            query: 查询文本
            
        Returns:
            实体 ID 列表
        """
        query_lower = query.lower()
        entities = []
        
        # 简单实现：查找图谱中匹配的实体
        for node_id, node_data in self.kg.graph.nodes(data=True):
            name = node_data.get('name', '').lower()
            if name and len(name) > 2 and name in query_lower:
                entities.append(node_id)
        
        return entities
    
    def _vector_search(
        self, 
        query: str, 
        user_id: str, 
        top_k: int,
        filters: Dict = None
    ) -> List[Dict]:
        """
        向量相似度搜索
        
        Args:
            query: 查询文本
            user_id: 用户 ID
            top_k: 返回数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        # 生成查询向量
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            logger.warning("无法生成查询向量，向量搜索返回空结果")
            return []
        
        # 调用数据库向量搜索
        try:
            results = self.db.vector_search(query_embedding, top_k, user_id)
            return [
                {
                    'memory_id': r['memory_id'],
                    'memory': r['memory'],
                    'score': 1 - min(r['distance'], 1.0),  # 距离转相似度，限制在 0-1
                    'source': 'vector'
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def _fts_search(
        self, 
        query: str, 
        user_id: str, 
        top_k: int,
        filters: Dict = None
    ) -> List[Dict]:
        """
        全文检索搜索
        
        Args:
            query: 查询文本
            user_id: 用户 ID
            top_k: 返回数量
            filters: 过滤条件
            
        Returns:
            搜索结果列表
        """
        try:
            # 清理查询：移除 FTS5 特殊字符
            # FTS5 特殊字符: " * : ^ ( ) { } [ ] - + ~ < > = & | / ? ' 
            import re
            clean_query = re.sub(r'[^\w\s]', ' ', query)  # 只保留字母数字和空格
            clean_query = re.sub(r'\s+', ' ', clean_query).strip()  # 合并多个空格
            
            if not clean_query:
                return []
            
            results = self.db.fts_search(clean_query, top_k, user_id)
            return [
                {
                    'memory_id': r['memory_id'],
                    'memory': r['memory'],
                    'score': 1.0 / (1 + abs(r['rank'])),  # rank 越小越好
                    'source': 'fts'
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"全文检索失败: {e}")
            return []
    
    def _temporal_search(
        self,
        entity_id: str,
        time_constraint: Dict,
        user_id: str,
        top_k: int
    ) -> List[Dict]:
        """
        时序检索
        
        Args:
            entity_id: 实体 ID
            time_constraint: 时间约束
            user_id: 用户 ID
            top_k: 返回数量
            
        Returns:
            搜索结果列表
        """
        try:
            # 调用知识图谱时序查询
            events = self.kg.query_temporal(
                entity_id, 
                time_constraint['type'], 
                time_constraint.get('time')
            )
            
            results = []
            for event in events[:top_k]:
                # 从事件中获取相关记忆
                memory_id = event.get('memory_id')
                if memory_id:
                    memory = self.db.get_memory(memory_id)
                    if memory and memory.user_id == user_id:
                        results.append({
                            'memory_id': memory_id,
                            'memory': memory,
                            'score': event.get('confidence', 0.5),
                            'source': 'temporal',
                            'event': event
                        })
            
            return results
        except Exception as e:
            logger.error(f"时序检索失败: {e}")
            return []
    
    def _paths_to_memories(
        self, 
        paths: List[List[Dict]], 
        user_id: str
    ) -> List[Dict]:
        """
        将多跳路径转换为记忆结果
        
        Args:
            paths: 路径列表
            user_id: 用户 ID
            
        Returns:
            记忆结果列表
        """
        results = []
        seen_memories = set()
        
        for path in paths:
            # 计算路径得分（基于跳数和置信度）
            path_confidence = sum(step.get('confidence', 0.5) for step in path) / len(path) if path else 0
            
            # 从路径的 evidence 中提取记忆
            for step in path:
                evidence = step.get('evidence')
                if evidence:
                    try:
                        import json
                        memory_ids = json.loads(evidence) if isinstance(evidence, str) else evidence
                        if isinstance(memory_ids, list):
                            for memory_id in memory_ids:
                                if memory_id not in seen_memories:
                                    memory = self.db.get_memory(memory_id)
                                    if memory and memory.user_id == user_id:
                                        results.append({
                                            'memory_id': memory_id,
                                            'memory': memory,
                                            'score': path_confidence,
                                            'source': 'graph',
                                            'path': path
                                        })
                                        seen_memories.add(memory_id)
                    except Exception:
                        pass
        
        return results
    
    def _merge_results(
        self,
        vector_results: List[Dict],
        fts_results: List[Dict],
        temporal_results: List[Dict],
        graph_results: List[Dict]
    ) -> Dict[str, Dict]:
        """
        合并各信号检索结果
        
        Args:
            vector_results: 向量搜索结果
            fts_results: 全文搜索结果
            temporal_results: 时序搜索结果
            graph_results: 图谱搜索结果
            
        Returns:
            合并后的结果字典 {memory_id: merged_data}
        """
        merged = defaultdict(lambda: {
            'memory': None,
            'vector_score': 0.0,
            'fts_score': 0.0,
            'temporal_score': 0.0,
            'graph_score': 0.0,
            'sources': []
        })
        
        # 合并向量结果
        for r in vector_results:
            mid = r['memory_id']
            merged[mid]['memory'] = r['memory']
            merged[mid]['vector_score'] = r['score']
            merged[mid]['sources'].append('vector')
        
        # 合并 FTS 结果
        for r in fts_results:
            mid = r['memory_id']
            merged[mid]['memory'] = r['memory']
            merged[mid]['fts_score'] = r['score']
            merged[mid]['sources'].append('fts')
        
        # 合并时序结果
        for r in temporal_results:
            mid = r['memory_id']
            merged[mid]['memory'] = r['memory']
            merged[mid]['temporal_score'] = r['score']
            merged[mid]['sources'].append('temporal')
        
        # 合并图谱结果
        for r in graph_results:
            mid = r['memory_id']
            merged[mid]['memory'] = r['memory']
            merged[mid]['graph_score'] = max(merged[mid]['graph_score'], r['score'])
            if 'graph' not in merged[mid]['sources']:
                merged[mid]['sources'].append('graph')
        
        return dict(merged)
    
    def _rank_with_weights(
        self,
        merged: Dict[str, Dict],
        weights: Dict[str, float],
        query_type: QueryType
    ) -> List[RetrievalResult]:
        """
        根据权重计算最终分数
        
        Score = w1*vector + w2*fts + w3*temporal + w4*graph + w5*recency
        
        Args:
            merged: 合并后的结果字典
            weights: 权重配置
            query_type: 查询类型
            
        Returns:
            排序后的检索结果列表
        """
        results = []
        
        for memory_id, data in merged.items():
            memory = data['memory']
            if memory is None:
                continue
            
            # 计算各信号分数（归一化到 0-1）
            vector_score = data.get('vector_score', 0)
            fts_score = data.get('fts_score', 0)
            temporal_score = data.get('temporal_score', 0)
            graph_score = data.get('graph_score', 0)
            
            # 访问频率分数（热度）
            recency_score = self._calculate_recency_score(memory)
            
            # 加权求和
            final_score = (
                weights.get('vector', 0) * vector_score +
                weights.get('fts', 0) * fts_score +
                weights.get('temporal', 0) * temporal_score +
                weights.get('graph', 0) * graph_score +
                weights.get('recency', 0) * recency_score
            )
            
            # 提升多源命中的结果
            sources = data.get('sources', [])
            if len(sources) > 1:
                final_score *= (1 + 0.1 * (len(sources) - 1))  # 每多一个源增加 10%
            
            results.append(RetrievalResult(
                memory=memory,
                final_score=final_score,
                vector_score=vector_score,
                fts_score=fts_score,
                temporal_score=temporal_score,
                graph_score=graph_score,
                recency_score=recency_score,
                source='+'.join(sources)
            ))
        
        # 按最终分数排序
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
    
    def _calculate_recency_score(self, memory) -> float:
        """
        计算访问频率/时效分数
        
        - 最近访问的分数高
        - 访问次数多的分数高
        
        Args:
            memory: Memory 对象
            
        Returns:
            0-1 之间的分数
        """
        # 基础分数
        score = 0.5
        
        # 访问次数加成
        if hasattr(memory, 'access_count') and memory.access_count > 0:
            score += min(0.3, math.log(memory.access_count + 1) / 10)
        
        # 最近访问加成
        if hasattr(memory, 'last_accessed') and memory.last_accessed:
            if isinstance(memory.last_accessed, datetime):
                days_since = (datetime.now() - memory.last_accessed).days
                score += max(0, 0.2 - days_since * 0.01)  # 每天衰减
        
        # 创建时间加成（越新的记忆分数越高）
        if hasattr(memory, 'created_at') and memory.created_at:
            if isinstance(memory.created_at, datetime):
                days_since = (datetime.now() - memory.created_at).days
                score += max(0, 0.1 - days_since * 0.005)  # 每天衰减，最多加 0.1
        
        return min(1.0, score)
    
    def _apply_filters(
        self, 
        results: List[RetrievalResult], 
        filters: Dict
    ) -> List[RetrievalResult]:
        """
        应用过滤条件
        
        Args:
            results: 检索结果列表
            filters: 过滤条件字典
            
        Returns:
            过滤后的结果列表
        """
        filtered = results
        
        # 按时间范围过滤
        if 'start_date' in filters and 'end_date' in filters:
            start = filters['start_date']
            end = filters['end_date']
            filtered = [
                r for r in filtered 
                if hasattr(r.memory, 'valid_at') and r.memory.valid_at
                and start <= r.memory.valid_at <= end
            ]
        
        # 按来源类型过滤
        if 'source_type' in filters:
            source_type = filters['source_type']
            filtered = [
                r for r in filtered 
                if hasattr(r.memory, 'source_type') 
                and r.memory.source_type == source_type
            ]
        
        # 按最小分数过滤
        if 'min_score' in filters:
            min_score = filters['min_score']
            filtered = [r for r in filtered if r.final_score >= min_score]
        
        return filtered
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        获取文本的 embedding
        
        Args:
            text: 输入文本
            
        Returns:
            向量或 None
        """
        if self.llm_client and hasattr(self.llm_client, 'embed'):
            try:
                return self.llm_client.embed(text)
            except Exception as e:
                logger.error(f"Embedding 生成失败: {e}")
                return None
        
        # 如果没有 LLM 客户端，返回 None
        logger.warning("未配置 LLM 客户端，无法生成 embedding")
        return None
    
    # ========================================================================
    # 高级检索功能
    # ========================================================================
    
    def retrieve_with_explanation(
        self,
        query: str,
        user_id: str,
        query_type: QueryType = QueryType.HYBRID,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        带解释的高级检索
        
        Args:
            query: 查询文本
            user_id: 用户 ID
            query_type: 查询类型
            top_k: 返回数量
            
        Returns:
            包含结果和解释的字典
        """
        start_time = datetime.now()
        
        # 执行检索
        results = self.retrieve(query, user_id, query_type, top_k)
        
        # 计算统计数据
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # 按来源统计
        source_counts = defaultdict(int)
        for r in results:
            for src in r.source.split('+'):
                source_counts[src] += 1
        
        return {
            'query': query,
            'query_type': query_type.value if query_type != QueryType.HYBRID else self._classify_query(query).value,
            'results': results,
            'stats': {
                'total_results': len(results),
                'elapsed_ms': round(elapsed_ms, 2),
                'source_distribution': dict(source_counts)
            }
        }
    
    def rerank_with_llm(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        使用 LLM 重新排序结果
        
        Args:
            query: 查询文本
            results: 初步检索结果
            top_k: 返回数量
            
        Returns:
            重新排序后的结果
        """
        if not self.llm_client or len(results) == 0:
            return results[:top_k]
        
        # 构建提示
        memories_text = "\n\n".join([
            f"[{i+1}] {r.memory.content[:200]}"
            for i, r in enumerate(results[:20])  # 只取前 20 个
        ])
        
        prompt = f"""根据以下查询，对检索到的记忆进行相关性排序。

查询: {query}

候选记忆:
{memories_text}

请输出最相关的 {top_k} 个记忆的序号（从 1 开始），按相关性从高到低排列。
输出格式: 数字序号列表，如 [3, 1, 5, 2, 4]
只输出 JSON 数组，不要其他内容。"""

        try:
            # 使用 Mistral 替代 Claude（中国 IP 兼容）
            response = self.llm_client.invoke_mistral(prompt, max_tokens=500, temperature=0.0)
            import json
            
            # 解析 JSON 数组
            ranks = json.loads(response.strip())
            if isinstance(ranks, list):
                reranked = []
                for idx in ranks[:top_k]:
                    if 1 <= idx <= len(results):
                        reranked.append(results[idx - 1])
                return reranked if reranked else results[:top_k]
        except Exception as e:
            logger.error(f"LLM 重排序失败: {e}")
        
        return results[:top_k]
