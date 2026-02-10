#!/usr/bin/env python3
"""
Mimir 检索优化方案

问题：
1. 跨时间关联 25% - "上周"没解析为日期范围
2. 精确时间+平台 33% - 没利用 source_type 过滤  
3. 跨平台关联 66% - 相关性算法不够强

优化策略：
1. 增强 temporal 查询解析
2. 混合检索加入平台过滤
3. 改进跨内容关联算法
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class QueryEnhancer:
    """
    查询增强器 - 解析用户查询中的时间、平台等隐含信息
    """
    
    def __init__(self, reference_date: datetime = None):
        self.reference_date = reference_date or datetime.now()
        
    def enhance(self, query: str) -> Dict:
        """
        增强查询，提取时间、平台等过滤条件
        
        Returns:
            {
                'original_query': str,
                'enhanced_query': str,
                'time_range': {'start': date, 'end': date} or None,
                'platform': str or None,
                'keywords': List[str]
            }
        """
        result = {
            'original_query': query,
            'enhanced_query': query,
            'time_range': None,
            'platform': None,
            'keywords': []
        }
        
        # 1. 解析时间表达
        time_range = self._parse_time_expression(query)
        if time_range:
            result['time_range'] = time_range
            # 从查询中移除时间表达，避免干扰语义检索
            result['enhanced_query'] = self._remove_time_expression(query)
        
        # 2. 解析平台来源
        platform = self._parse_platform(query)
        if platform:
            result['platform'] = platform
            result['enhanced_query'] = self._remove_platform_expression(result['enhanced_query'])
        
        # 3. 提取关键词
        result['keywords'] = self._extract_keywords(result['enhanced_query'])
        
        return result
    
    def _parse_time_expression(self, query: str) -> Optional[Dict]:
        """解析查询中的时间表达"""
        query_lower = query.lower()
        
        # "上周" → 上一周的时间范围
        if '上周' in query_lower or 'last week' in query_lower:
            end = self.reference_date
            start = end - timedelta(days=7)
            return {'start': start, 'end': end, 'type': 'last_week'}
        
        # "周一" / "Monday" → 本周一或上周一（根据上下文）
        weekday_match = re.search(r'(上周?)?(周一|monday)', query_lower)
        if weekday_match:
            # 默认上周一（因为用户通常问过去的事）
            days_since_monday = (self.reference_date.weekday() - 0) % 7
            if days_since_monday == 0:
                days_since_monday = 7
            last_monday = self.reference_date - timedelta(days=days_since_monday)
            return {
                'start': last_monday,
                'end': last_monday + timedelta(days=1),
                'type': 'last_monday'
            }
        
        # "昨天" → 昨天
        if '昨天' in query_lower or 'yesterday' in query_lower:
            yesterday = self.reference_date - timedelta(days=1)
            return {
                'start': yesterday,
                'end': yesterday + timedelta(days=1),
                'type': 'yesterday'
            }
        
        return None
    
    def _parse_platform(self, query: str) -> Optional[str]:
        """解析查询中的平台来源"""
        query_lower = query.lower()
        
        platforms = {
            'claude': ['claude', '克劳德'],
            'chatgpt': ['chatgpt', 'chatgpt', 'gpt'],
            'article': ['文章', 'article', '收藏'],
            'note': ['笔记', 'note']
        }
        
        for platform, keywords in platforms.items():
            if any(kw in query_lower for kw in keywords):
                return platform
        
        return None
    
    def _remove_time_expression(self, query: str) -> str:
        """从查询中移除时间表达"""
        # 简单移除常见时间词
        time_patterns = [
            r'上周?', r'last week',
            r'周一|monday',
            r'昨天|yesterday',
            r'上?个?月'
        ]
        
        result = query
        for pattern in time_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        return result.strip()
    
    def _remove_platform_expression(self, query: str) -> str:
        """从查询中移除平台表达"""
        platform_patterns = [
            r'claude', r'克劳德',
            r'chatgpt', r'gpt',
            r'和\s*\w+\s*讨论',
            r'在\s*\w+\s*上'
        ]
        
        result = query
        for pattern in platform_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        return result.strip()
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 简单实现：取名词和动词
        # 实际可以用 jieba 或 LLM
        words = query.split()
        # 过滤停用词
        stopwords = {'我', '你', '的', '了', '在', '和', '与', '帮', '写', '做', '是'}
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        return keywords[:5]  # 取前5个


class EnhancedRetriever:
    """
    增强检索器 - 支持时间过滤、平台过滤的混合检索
    """
    
    def __init__(self, mimir_db, llm_client):
        self.db = mimir_db
        self.llm = llm_client
        self.query_enhancer = QueryEnhancer()
    
    def retrieve(self, query: str, user_id: str = 'default', top_k: int = 5) -> List[Dict]:
        """
        增强检索
        
        流程：
        1. 解析查询（时间、平台、关键词）
        2. 语义检索
        3. 应用时间过滤
        4. 应用平台过滤
        5. 重新排序
        """
        # 1. 解析查询
        enhanced = self.query_enhancer.enhance(query)
        
        # 2. 语义检索（扩大范围）
        candidates = self._semantic_search(
            enhanced['enhanced_query'], 
            user_id, 
            top_k=top_k * 3  # 扩大候选池
        )
        
        # 3. 应用过滤
        filtered = self._apply_filters(candidates, enhanced)
        
        # 4. 重新排序（考虑多源关联）
        ranked = self._rerank_with_cross_source(filtered, enhanced)
        
        return ranked[:top_k]
    
    def _semantic_search(self, query: str, user_id: str, top_k: int) -> List[Dict]:
        """语义检索"""
        # 使用现有的检索逻辑
        # 这里简化实现
        return []
    
    def _apply_filters(self, candidates: List[Dict], enhanced: Dict) -> List[Dict]:
        """应用时间和平台过滤"""
        filtered = candidates
        
        # 时间过滤
        if enhanced['time_range']:
            start = enhanced['time_range']['start']
            end = enhanced['time_range']['end']
            
            filtered = [
                c for c in filtered
                if self._is_in_time_range(c, start, end)
            ]
        
        # 平台过滤（软过滤，降低权重而非完全排除）
        if enhanced['platform']:
            # 提升目标平台的权重
            for c in filtered:
                if c.get('source_type') == enhanced['platform']:
                    c['score'] = c.get('score', 1.0) * 1.2  # 提升 20%
        
        return filtered
    
    def _is_in_time_range(self, candidate: Dict, start: datetime, end: datetime) -> bool:
        """检查候选是否在时间范围内"""
        # 从 metadata 中提取日期
        try:
            metadata = candidate.get('metadata', {})
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata)
            
            session_date = metadata.get('session_date', '')
            if session_date:
                # 解析日期
                candidate_date = self._parse_date(session_date)
                if candidate_date:
                    return start <= candidate_date <= end
        except:
            pass
        
        return True  # 如果无法解析，默认保留
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """解析日期字符串"""
        formats = ['%d %B %Y', '%Y-%m-%d', '%B %d, %Y']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        return None
    
    def _rerank_with_cross_source(self, candidates: List[Dict], enhanced: Dict) -> List[Dict]:
        """
        跨源关联重排序
        
        策略：
        - 如果有多来源（claude + article），提升排名
        - 优先展示能提供完整 context 的组合
        """
        # 按来源分组
        by_source = {}
        for c in candidates:
            source = c.get('source_type', 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(c)
        
        # 如果有多于一个来源，交错排列
        if len(by_source) > 1:
            result = []
            sources = list(by_source.keys())
            indices = {s: 0 for s in sources}
            
            while len(result) < len(candidates):
                for source in sources:
                    if indices[source] < len(by_source[source]):
                        result.append(by_source[source][indices[source]])
                        indices[source] += 1
            
            return result
        
        return candidates


# 测试
if __name__ == "__main__":
    enhancer = QueryEnhancer(reference_date=datetime(2026, 2, 10))
    
    test_queries = [
        "我上周规划的项目",
        "上周一我和 Claude 讨论了什么",
        "向量检索的代码实现",
        "Chrome Extension 开发",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = enhancer.enhance(query)
        print(f"  Enhanced: {result['enhanced_query']}")
        print(f"  Time range: {result['time_range']}")
        print(f"  Platform: {result['platform']}")
        print(f"  Keywords: {result['keywords']}")
