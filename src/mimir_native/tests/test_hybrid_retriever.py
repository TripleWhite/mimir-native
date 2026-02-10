"""
Test script for Mimir Memory V2 - Hybrid Retriever
验证混合检索、查询分类、融合排序功能
"""

import os
import sys
import json
import tempfile
import unittest
import math
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# 确保能导入 mimir_v2
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.mimir_v2.retrieval.hybrid_retriever import (
    HybridRetriever, QueryType, RetrievalResult
)
from app.mimir_v2.database import (
    MimirDatabase, MemoryCreate, Memory, init_database
)


class MockMemory:
    """Mock Memory 对象用于测试"""
    
    def __init__(self, memory_id, content, user_id="test_user", **kwargs):
        self.id = memory_id
        self.memory_id = memory_id
        self.content = content
        self.user_id = user_id
        self.access_count = kwargs.get('access_count', 0)
        self.last_accessed = kwargs.get('last_accessed', None)
        self.created_at = kwargs.get('created_at', datetime.now())
        self.valid_at = kwargs.get('valid_at', None)
        self.source_type = kwargs.get('source_type', 'chat')


class MockBedrockClient:
    """Mock Bedrock 客户端用于测试"""
    
    def __init__(self):
        self.available = True
        self.embed_call_count = 0
        self.embedding_dim = 1536
    
    def is_available(self) -> bool:
        return self.available
    
    def embed(self, text: str, dimensions: int = 1536):
        """Mock embedding generation"""
        self.embed_call_count += 1
        # 返回固定模式的向量（基于文本哈希）
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # 生成归一化的随机向量
        vec = [(hash_val >> (i % 32)) % 100 / 100.0 for i in range(dimensions)]
        # 归一化
        norm = math.sqrt(sum(x*x for x in vec))
        if norm > 0:
            vec = [x/norm for x in vec]
        return vec


class MockMimirDatabase:
    """Mock MimirDatabase 用于测试"""
    
    def __init__(self):
        self.memories = {}
        self.vector_search_results = []
        self.fts_search_results = []
    
    def add_memory(self, memory: MockMemory):
        self.memories[memory.memory_id] = memory
    
    def get_memory(self, memory_id: str):
        return self.memories.get(memory_id)
    
    def vector_search(self, query_embedding, top_k: int, user_id: str):
        """Mock vector search"""
        return self.vector_search_results[:top_k]
    
    def fts_search(self, query: str, top_k: int, user_id: str):
        """Mock FTS search"""
        return self.fts_search_results[:top_k]


class MockTemporalKnowledgeGraph:
    """Mock TemporalKnowledgeGraph 用于测试"""
    
    def __init__(self):
        self.graph = MagicMock()
        self.graph.nodes = MagicMock(return_value=[])
        self.temporal_results = []
        self.multi_hop_paths = []
    
    def query_temporal(self, entity_id: str, time_type: str, time_value=None):
        """Mock temporal query"""
        return self.temporal_results
    
    def multi_hop_query(self, entity_a: str, entity_b: str, max_hops: int = 3):
        """Mock multi-hop query"""
        return self.multi_hop_paths


class TestHybridRetriever(unittest.TestCase):
    """混合检索器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.mock_db = MockMimirDatabase()
        self.mock_kg = MockTemporalKnowledgeGraph()
        self.mock_llm = MockBedrockClient()
        
        # 创建 HybridRetriever 实例
        self.retriever = HybridRetriever(
            db=self.mock_db,
            kg=self.mock_kg,
            llm_client=self.mock_llm
        )
        
        # 准备测试数据
        self._setup_test_memories()
    
    def _setup_test_memories(self):
        """设置测试记忆数据"""
        # 创建测试记忆
        self.memories = {
            'mem_1': MockMemory(
                'mem_1', 
                'John works at Google as a software engineer',
                access_count=10,
                last_accessed=datetime.now() - timedelta(days=1),
                created_at=datetime.now() - timedelta(days=30)
            ),
            'mem_2': MockMemory(
                'mem_2',
                'Alice likes to eat pizza on weekends',
                access_count=5,
                last_accessed=datetime.now() - timedelta(days=5),
                created_at=datetime.now() - timedelta(days=60)
            ),
            'mem_3': MockMemory(
                'mem_3',
                'Team meeting scheduled for next Tuesday at 3pm',
                access_count=2,
                last_accessed=datetime.now() - timedelta(days=10),
                created_at=datetime.now() - timedelta(days=15)
            ),
            'mem_4': MockMemory(
                'mem_4',
                'Bob moved to New York last year',
                access_count=0,
                last_accessed=None,
                created_at=datetime.now() - timedelta(days=365)
            ),
        }
        
        for mem in self.memories.values():
            self.mock_db.add_memory(mem)
    
    def test_factual_query(self):
        """测试事实查询 - vector + fts + recency 权重"""
        print("\n=== Test: Factual Query ===")
        
        # 设置 mock 搜索结果
        self.mock_db.vector_search_results = [
            {'memory_id': 'mem_1', 'memory': self.memories['mem_1'], 'distance': 0.1},
            {'memory_id': 'mem_2', 'memory': self.memories['mem_2'], 'distance': 0.3},
        ]
        self.mock_db.fts_search_results = [
            {'memory_id': 'mem_1', 'memory': self.memories['mem_1'], 'rank': 1},
            {'memory_id': 'mem_3', 'memory': self.memories['mem_3'], 'rank': 2},
        ]
        
        # 执行检索 - 事实查询
        results = self.retriever.retrieve(
            query="What does John do at Google?",
            user_id="test_user",
            query_type=QueryType.FACTUAL,
            top_k=5
        )
        
        # 验证结果
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # 验证结果类型
        for r in results:
            self.assertIsInstance(r, RetrievalResult)
            self.assertIsNotNone(r.memory)
            self.assertGreaterEqual(r.final_score, 0)
            self.assertLessEqual(r.final_score, 1.5)  # 考虑多源加成
        
        # 验证事实查询的权重配置
        weights = self.retriever.weights[QueryType.FACTUAL]
        self.assertEqual(weights['vector'], 0.5)
        self.assertEqual(weights['fts'], 0.3)
        self.assertEqual(weights['temporal'], 0.0)
        self.assertEqual(weights['graph'], 0.0)
        self.assertEqual(weights['recency'], 0.2)
        
        print(f"✓ 事实查询返回 {len(results)} 个结果")
        print(f"✓ 权重配置正确: {weights}")
    
    def test_temporal_query(self):
        """测试时序查询 - temporal 权重优先"""
        print("\n=== Test: Temporal Query ===")
        
        # 设置 mock 搜索结果
        self.mock_db.vector_search_results = [
            {'memory_id': 'mem_3', 'memory': self.memories['mem_3'], 'distance': 0.2},
        ]
        self.mock_db.fts_search_results = [
            {'memory_id': 'mem_3', 'memory': self.memories['mem_3'], 'rank': 1},
        ]
        
        # Mock temporal 搜索结果
        self.mock_kg.temporal_results = [
            {'memory_id': 'mem_3', 'confidence': 0.9, 'event_type': 'meeting'}
        ]
        
        # 执行检索 - 时序查询
        results = self.retriever.retrieve(
            query="When was the team meeting scheduled?",
            user_id="test_user",
            query_type=QueryType.TEMPORAL,
            top_k=5
        )
        
        # 验证时序查询的权重配置
        weights = self.retriever.weights[QueryType.TEMPORAL]
        self.assertEqual(weights['temporal'], 0.4)  # temporal 权重最高
        self.assertEqual(weights['vector'], 0.3)
        self.assertEqual(weights['fts'], 0.2)
        
        print(f"✓ 时序查询返回 {len(results)} 个结果")
        print(f"✓ 时序权重配置正确: temporal={weights['temporal']}")
    
    def test_multi_hop_query(self):
        """测试多跳查询 - graph 权重优先"""
        print("\n=== Test: Multi-hop Query ===")
        
        # 设置 mock 搜索结果
        self.mock_db.vector_search_results = [
            {'memory_id': 'mem_1', 'memory': self.memories['mem_1'], 'distance': 0.2},
            {'memory_id': 'mem_4', 'memory': self.memories['mem_4'], 'distance': 0.3},
        ]
        self.mock_db.fts_search_results = [
            {'memory_id': 'mem_1', 'memory': self.memories['mem_1'], 'rank': 1},
        ]
        
        # Mock multi-hop 路径
        self.mock_kg.multi_hop_paths = [
            [
                {'entity': 'John', 'relation': 'works_at', 'target': 'Google', 'confidence': 0.9, 'evidence': '["mem_1"]'},
                {'entity': 'Google', 'relation': 'located_in', 'target': 'New York', 'confidence': 0.8, 'evidence': '["mem_4"]'},
            ]
        ]
        
        # 执行检索 - 多跳查询
        results = self.retriever.retrieve(
            query="Where does John's company locate?",
            user_id="test_user",
            query_type=QueryType.MULTI_HOP,
            top_k=5
        )
        
        # 验证多跳查询的权重配置
        weights = self.retriever.weights[QueryType.MULTI_HOP]
        self.assertEqual(weights['graph'], 0.4)  # graph 权重最高
        self.assertEqual(weights['vector'], 0.3)
        self.assertEqual(weights['fts'], 0.2)
        
        print(f"✓ 多跳查询返回 {len(results)} 个结果")
        print(f"✓ 图谱权重配置正确: graph={weights['graph']}")
    
    def test_query_classification(self):
        """测试自动查询类型分类"""
        print("\n=== Test: Query Classification ===")
        
        # 测试事实查询分类
        factual_queries = [
            "What does John do?",
            "Tell me about Alice preferences",
            "The office is located in downtown",
            "What is the weather today?",
            "How tall is the building?",
        ]
        
        for query in factual_queries:
            query_type = self.retriever._classify_query(query)
            self.assertEqual(query_type, QueryType.FACTUAL, 
                           f"Query '{query}' should be classified as FACTUAL")
        
        # 测试时序查询分类
        temporal_queries = [
            "When did John join the company?",
            "What happened yesterday?",
            "The meeting started before lunch",
            "What time is the appointment?",
        ]
        
        for query in temporal_queries:
            query_type = self.retriever._classify_query(query)
            self.assertEqual(query_type, QueryType.TEMPORAL,
                           f"Query '{query}' should be classified as TEMPORAL")
        
        # 测试多跳查询分类
        multi_hop_queries = [
            "Who works at John's company?",
            "What did Alice eat at Bob's party?",
            "Where does John's wife work?",
        ]
        
        for query in multi_hop_queries:
            query_type = self.retriever._classify_query(query)
            self.assertEqual(query_type, QueryType.MULTI_HOP,
                           f"Query '{query}' should be classified as MULTI_HOP")
        
        print(f"✓ 查询分类测试通过")
        print(f"  - 事实查询: {len(factual_queries)} 个")
        print(f"  - 时序查询: {len(temporal_queries)} 个")
        print(f"  - 多跳查询: {len(multi_hop_queries)} 个")
    
    def test_fusion_ranking(self):
        """测试融合排序正确性"""
        print("\n=== Test: Fusion Ranking ===")
        
        # 准备合并结果
        merged = {
            'mem_1': {
                'memory': self.memories['mem_1'],
                'vector_score': 0.9,
                'fts_score': 0.8,
                'temporal_score': 0.0,
                'graph_score': 0.0,
                'sources': ['vector', 'fts']
            },
            'mem_2': {
                'memory': self.memories['mem_2'],
                'vector_score': 0.7,
                'fts_score': 0.0,
                'temporal_score': 0.0,
                'graph_score': 0.0,
                'sources': ['vector']
            },
            'mem_3': {
                'memory': self.memories['mem_3'],
                'vector_score': 0.0,
                'fts_score': 0.9,
                'temporal_score': 0.8,
                'graph_score': 0.0,
                'sources': ['fts', 'temporal']
            },
        }
        
        weights = {
            'vector': 0.5,
            'fts': 0.3,
            'temporal': 0.2,
            'graph': 0.0,
            'recency': 0.0
        }
        
        # 执行排序
        results = self.retriever._rank_with_weights(merged, weights, QueryType.FACTUAL)
        
        # 验证结果按分数降序排列
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].final_score, results[i+1].final_score)
        
        # 验证 mem_1 分数最高（vector + fts 双源命中）
        self.assertEqual(results[0].memory.id, 'mem_1')
        
        # 验证多源加成
        # mem_1: 0.9*0.5 + 0.8*0.3 = 0.45 + 0.24 = 0.69, +10% = 0.759
        # mem_3: 0.9*0.3 + 0.8*0.2 = 0.27 + 0.16 = 0.43, +10% = 0.473
        # mem_2: 0.7*0.5 = 0.35 (单源无加成)
        
        self.assertGreater(results[0].final_score, results[1].final_score)
        
        print(f"✓ 融合排序测试通过")
        print(f"  - 结果数量: {len(results)}")
        print(f"  - 最高分: {results[0].final_score:.3f}")
        print(f"  - 最低分: {results[-1].final_score:.3f}")
    
    def test_recency_score(self):
        """测试访问频率分数计算"""
        print("\n=== Test: Recency Score ===")
        
        # 创建不同访问模式的记忆
        now = datetime.now()
        
        # 高频访问记忆
        high_freq_mem = MockMemory(
            'high_freq',
            'High frequency memory',
            access_count=100,
            last_accessed=now - timedelta(hours=1),
            created_at=now - timedelta(days=1)
        )
        
        # 低频访问记忆
        low_freq_mem = MockMemory(
            'low_freq',
            'Low frequency memory',
            access_count=1,
            last_accessed=now - timedelta(days=30),
            created_at=now - timedelta(days=60)
        )
        
        # 新创建记忆
        new_mem = MockMemory(
            'new_mem',
            'New memory',
            access_count=0,
            last_accessed=None,
            created_at=now - timedelta(hours=1)
        )
        
        # 计算分数
        high_score = self.retriever._calculate_recency_score(high_freq_mem)
        low_score = self.retriever._calculate_recency_score(low_freq_mem)
        new_score = self.retriever._calculate_recency_score(new_mem)
        
        # 验证分数范围
        self.assertGreaterEqual(high_score, 0)
        self.assertLessEqual(high_score, 1.0)
        self.assertGreaterEqual(low_score, 0)
        self.assertLessEqual(low_score, 1.0)
        
        # 高频访问记忆分数应更高
        self.assertGreater(high_score, low_score, 
                          "High frequency memory should have higher recency score")
        
        # 新记忆分数应高于旧记忆
        self.assertGreater(new_score, low_score,
                          "New memory should have higher recency score than old memory")
        
        print(f"✓ 热度分数计算测试通过")
        print(f"  - 高频记忆分数: {high_score:.3f}")
        print(f"  - 低频记忆分数: {low_score:.3f}")
        print(f"  - 新记忆分数: {new_score:.3f}")
    
    def test_integration_e2e(self):
        """端到端集成测试"""
        print("\n=== Test: End-to-End Integration ===")
        
        # 设置完整的 mock 数据
        self.mock_db.vector_search_results = [
            {'memory_id': 'mem_1', 'memory': self.memories['mem_1'], 'distance': 0.1},
            {'memory_id': 'mem_2', 'memory': self.memories['mem_2'], 'distance': 0.2},
            {'memory_id': 'mem_3', 'memory': self.memories['mem_3'], 'distance': 0.3},
        ]
        self.mock_db.fts_search_results = [
            {'memory_id': 'mem_1', 'memory': self.memories['mem_1'], 'rank': 1},
            {'memory_id': 'mem_4', 'memory': self.memories['mem_4'], 'rank': 2},
        ]
        
        # 执行 HYBRID 自动分类检索
        results = self.retriever.retrieve(
            query="What does John do at work?",
            user_id="test_user",
            query_type=QueryType.HYBRID,  # 自动分类
            top_k=3
        )
        
        # 验证结果
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        
        # 验证每个结果的完整性
        for r in results:
            self.assertIsInstance(r, RetrievalResult)
            self.assertIsNotNone(r.memory)
            self.assertIsNotNone(r.memory.content)
            self.assertGreaterEqual(r.final_score, 0)
            
            # 验证分数组成
            self.assertIsInstance(r.vector_score, float)
            self.assertIsInstance(r.fts_score, float)
            self.assertIsInstance(r.recency_score, float)
        
        # 测试 retrieve_with_explanation
        explanation = self.retriever.retrieve_with_explanation(
            query="What does John do?",
            user_id="test_user",
            top_k=3
        )
        
        self.assertIn('query', explanation)
        self.assertIn('query_type', explanation)
        self.assertIn('results', explanation)
        self.assertIn('stats', explanation)
        self.assertIn('elapsed_ms', explanation['stats'])
        
        print(f"✓ 端到端集成测试通过")
        print(f"  - 检索结果: {len(results)} 个")
        print(f"  - 解释包含字段: {list(explanation.keys())}")
        print(f"  - 统计信息: {explanation['stats']}")
    
    def test_get_embedding(self):
        """测试 embedding 获取"""
        print("\n=== Test: Get Embedding ===")
        
        # 测试正常情况
        embedding = self.retriever._get_embedding("test text")
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 1536)  # 默认维度 (Titan Embeddings)
        
        # 测试 LLM 客户端不可用时
        retriever_no_llm = HybridRetriever(
            db=self.mock_db,
            kg=self.mock_kg,
            llm_client=None
        )
        embedding = retriever_no_llm._get_embedding("test text")
        self.assertIsNone(embedding)
        
        print(f"✓ Embedding 获取测试通过")
        print(f"  - 向量维度: {len(self.retriever._get_embedding('test'))}")
    
    def test_empty_results(self):
        """测试空结果处理"""
        print("\n=== Test: Empty Results ===")
        
        # 清空搜索结果
        self.mock_db.vector_search_results = []
        self.mock_db.fts_search_results = []
        
        results = self.retriever.retrieve(
            query="nonexistent query",
            user_id="test_user",
            top_k=5
        )
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)
        
        print(f"✓ 空结果处理测试通过")
    
    def test_filter_application(self):
        """测试过滤条件应用"""
        print("\n=== Test: Filter Application ===")
        
        # 创建带不同 source_type 的记忆
        mem_chat = MockMemory('mem_chat', 'Chat memory', source_type='chat')
        mem_doc = MockMemory('mem_doc', 'Document memory', source_type='document')
        
        results = [
            RetrievalResult(memory=mem_chat, final_score=0.9),
            RetrievalResult(memory=mem_doc, final_score=0.8),
        ]
        
        # 应用 source_type 过滤
        filtered = self.retriever._apply_filters(results, {'source_type': 'chat'})
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].memory.source_type, 'chat')
        
        # 应用 min_score 过滤
        filtered = self.retriever._apply_filters(results, {'min_score': 0.85})
        self.assertEqual(len(filtered), 1)
        self.assertGreaterEqual(filtered[0].final_score, 0.85)
        
        print(f"✓ 过滤条件应用测试通过")


class TestRetrievalResult(unittest.TestCase):
    """RetrievalResult 数据类测试"""
    
    def test_dataclass_creation(self):
        """测试 RetrievalResult 创建"""
        mem = MockMemory('test', 'test content')
        result = RetrievalResult(
            memory=mem,
            final_score=0.95,
            vector_score=0.9,
            fts_score=0.8,
            source="vector+fts"
        )
        
        self.assertEqual(result.memory.id, 'test')
        self.assertEqual(result.final_score, 0.95)
        self.assertEqual(result.source, "vector+fts")


class TestQueryType(unittest.TestCase):
    """QueryType 枚举测试"""
    
    def test_enum_values(self):
        """测试 QueryType 枚举值"""
        self.assertEqual(QueryType.FACTUAL.value, "factual")
        self.assertEqual(QueryType.TEMPORAL.value, "temporal")
        self.assertEqual(QueryType.MULTI_HOP.value, "multi_hop")
        self.assertEqual(QueryType.HYBRID.value, "hybrid")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestHybridRetriever))
    suite.addTests(loader.loadTestsFromTestCase(TestRetrievalResult))
    suite.addTests(loader.loadTestsFromTestCase(TestQueryType))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回结果
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
