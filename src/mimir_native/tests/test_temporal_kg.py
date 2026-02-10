"""
Tests for Temporal Knowledge Graph

测试用例：
1. 从数据库构建图谱
2. 添加实体和关系
3. 时序查询（before/after/at/between）
4. 多跳路径查询
5. 相关实体查找
6. LLM 实体/关系提取
7. 持久化到数据库
"""

import os
import sys
import json
import uuid
import tempfile
import unittest
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

# 添加项目路径（确保 app.mimir_v2 可导入）
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # 指向 backend 目录

from app.mimir_v2.temporal_kg import TemporalKnowledgeGraph
from app.mimir_v2.database import (
    MimirDatabase, MemoryCreate, EntityCreate, Entity, 
    RelationCreate, Relation, init_database
)


class MockLLMClient:
    """模拟 LLM 客户端用于测试"""
    
    def invoke_mistral(self, prompt: str, max_tokens: int = 1000) -> str:
        """模拟 Mistral 调用"""
        # 根据 prompt 内容返回不同的模拟响应
        if "Caroline" in prompt or "adoption" in prompt:
            return json.dumps({
                "entities": [
                    {"name": "Caroline", "type": "person", "aliases": []},
                    {"name": "adoption agencies", "type": "organization", "aliases": []},
                    {"name": "New York", "type": "location", "aliases": []}
                ],
                "relations": [
                    {
                        "source": "Caroline",
                        "target": "adoption agencies",
                        "relation_type": "WORKED_ON",
                        "time": "2023-05-15"
                    },
                    {
                        "source": "Caroline",
                        "target": "New York",
                        "relation_type": "LOCATED_IN",
                        "time": None
                    }
                ]
            })
        elif "John" in prompt or "Alice" in prompt:
            return json.dumps({
                "entities": [
                    {"name": "John", "type": "person", "aliases": []},
                    {"name": "Alice", "type": "person", "aliases": []},
                    {"name": "Acme Corp", "type": "organization", "aliases": []}
                ],
                "relations": [
                    {
                        "source": "John",
                        "target": "Acme Corp",
                        "relation_type": "WORKS_AT",
                        "time": "2023-01-01"
                    },
                    {
                        "source": "Alice",
                        "target": "Acme Corp",
                        "relation_type": "WORKS_AT",
                        "time": "2023-06-01"
                    }
                ]
            })
        else:
            return json.dumps({
                "entities": [
                    {"name": "TestEntity", "type": "concept", "aliases": []}
                ],
                "relations": []
            })


class TestTemporalKnowledgeGraph(unittest.TestCase):
    """时序知识图谱测试类"""
    
    def setUp(self):
        """测试前设置"""
        # 创建临时数据库
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_kg.db")
        
        # 初始化数据库
        init_database(self.db_path)
        self.db = MimirDatabase(self.db_path)
        
        # 创建图谱
        self.kg = TemporalKnowledgeGraph(self.db)
        self.user_id = "test_user"
        
        # 创建模拟 LLM 客户端
        self.mock_llm = MockLLMClient()
        
        # 设置测试数据
        self._setup_test_graph()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _setup_test_graph(self):
        """设置测试图谱数据（直接添加到内存图谱）"""
        # 创建测试 ID
        self.alice_id = str(uuid.uuid4())
        self.bob_id = str(uuid.uuid4())
        self.carol_id = str(uuid.uuid4())
        self.acme_id = str(uuid.uuid4())
        self.techcorp_id = str(uuid.uuid4())
        self.event1_id = str(uuid.uuid4())
        self.event2_id = str(uuid.uuid4())
        self.event3_id = str(uuid.uuid4())
        
        # 添加节点
        self.kg.graph.add_node(self.alice_id, name="Alice", type="person", mention_count=1)
        self.kg.graph.add_node(self.bob_id, name="Bob", type="person", mention_count=1)
        self.kg.graph.add_node(self.carol_id, name="Carol", type="person", mention_count=1)
        self.kg.graph.add_node(self.acme_id, name="Acme Corp", type="organization", mention_count=1)
        self.kg.graph.add_node(self.techcorp_id, name="TechCorp", type="organization", mention_count=1)
        self.kg.graph.add_node(self.event1_id, name="Conference", type="event", mention_count=1)
        self.kg.graph.add_node(self.event2_id, name="Meeting", type="event", mention_count=1)
        self.kg.graph.add_node(self.event3_id, name="Workshop", type="event", mention_count=1)
        
        # 添加边（带时间）
        self.kg.graph.add_edge(self.alice_id, self.event1_id, 
                              relation_type="PARTICIPATED_IN",
                              valid_from=datetime(2023, 3, 15),
                              confidence=1.0)
        self.kg.graph.add_edge(self.alice_id, self.event2_id,
                              relation_type="PARTICIPATED_IN",
                              valid_from=datetime(2023, 1, 10),
                              confidence=1.0)
        self.kg.graph.add_edge(self.alice_id, self.event3_id,
                              relation_type="PARTICIPATED_IN",
                              valid_from=datetime(2023, 6, 20),
                              confidence=1.0)
        self.kg.graph.add_edge(self.alice_id, self.bob_id,
                              relation_type="FRIEND_OF",
                              valid_from=datetime(2022, 1, 1),
                              confidence=1.0)
        self.kg.graph.add_edge(self.alice_id, self.carol_id,
                              relation_type="FRIEND_OF",
                              valid_from=datetime(2022, 6, 1),
                              confidence=1.0)
        self.kg.graph.add_edge(self.alice_id, self.acme_id,
                              relation_type="WORKS_AT",
                              valid_from=datetime(2023, 1, 1),
                              confidence=1.0)
        self.kg.graph.add_edge(self.bob_id, self.acme_id,
                              relation_type="WORKS_AT",
                              valid_from=datetime(2023, 2, 1),
                              confidence=1.0)
    
    # ========================================================================
    # Test 1: 空图谱构建
    # ========================================================================
    
    def test_empty_graph_build(self):
        """测试空图谱构建"""
        empty_kg = TemporalKnowledgeGraph(self.db)
        empty_kg.build_from_memories(self.user_id)
        
        self.assertEqual(len(empty_kg.graph.nodes), 0)
        self.assertEqual(len(empty_kg.graph.edges), 0)
    
    # ========================================================================
    # Test 2: 添加实体和关系
    # ========================================================================
    
    def test_add_fact(self):
        """测试添加事实到图谱"""
        from app.mimir_v2.database import Memory
        
        # 构建空图谱
        kg = TemporalKnowledgeGraph(self.db)
        
        # 创建模拟 Memory
        memory = Memory(
            id="memory_test",
            user_id=self.user_id,
            content="Dave works at TechCorp",
            content_hash="hash123",
            embedding=None,
            valid_at=datetime(2023, 8, 15),
            valid_at_confidence=1.0,
            temporal_tags=None,
            source_type="chat",
            source_id=None,
            source_metadata=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            access_count=0,
            last_accessed=None,
            version=1,
            superseded_by=None,
            fts_docid=None
        )
        
        # 创建实体
        dave_id = str(uuid.uuid4())
        entity1 = Entity(
            id=dave_id,
            name="Dave",
            type="person",
            aliases=None,
            first_seen=datetime(2023, 8, 15),
            last_seen=datetime(2023, 8, 15),
            mention_count=1
        )
        entity2 = Entity(
            id=self.techcorp_id,
            name="TechCorp",
            type="organization",
            aliases=None,
            first_seen=datetime(2023, 8, 15),
            last_seen=datetime(2023, 8, 15),
            mention_count=1
        )
        
        # 创建关系
        relation = Relation(
            id=str(uuid.uuid4()),
            source_entity=entity1.id,
            target_entity=entity2.id,
            relation_type="WORKS_AT",
            valid_from=datetime(2023, 8, 15),
            valid_until=None,
            evidence_memory_ids=json.dumps([memory.id]),
            confidence=1.0,
            created_at=datetime.now()
        )
        
        # 添加事实
        kg.add_fact(memory, [entity1, entity2], [relation])
        
        # 验证
        self.assertEqual(len(kg.graph.nodes), 2)
        self.assertEqual(len(kg.graph.edges), 1)
        self.assertIn(entity1.id, kg.graph.nodes)
        self.assertIn(entity2.id, kg.graph.nodes)
    
    # ========================================================================
    # Test 3: 时序查询（before/after/at/between）
    # ========================================================================
    
    def test_query_temporal_before(self):
        """测试时序查询 - before"""
        # 查询 2023-04-01 之前的事件
        results = self.kg.query_temporal(self.alice_id, "before", 
                                         datetime(2023, 4, 1))
        
        # 应该包含 Meeting (1/10) 和 Conference (3/15)，以及 FRIEND_OF 关系
        self.assertGreaterEqual(len(results), 2)
        # 验证所有结果都在指定时间之前
        self.assertTrue(all(r['time'] < datetime(2023, 4, 1) for r in results if r['time']))
    
    def test_query_temporal_after(self):
        """测试时序查询 - after"""
        # 查询 2023-03-01 之后的事件
        results = self.kg.query_temporal(self.alice_id, "after",
                                         datetime(2023, 3, 1))
        
        self.assertEqual(len(results), 2)  # Conference (3/15) 和 Workshop (6/20)
        self.assertTrue(all(r['time'] > datetime(2023, 3, 1) for r in results))
    
    def test_query_between(self):
        """测试时序查询 - between"""
        # 查询 2023-02-01 到 2023-08-01 之间的事件
        results = self.kg.query_between(self.alice_id,
                                        datetime(2023, 2, 1),
                                        datetime(2023, 8, 1))
        
        self.assertEqual(len(results), 2)  # Conference (3/15) 和 Meeting (1/10 在范围外)
        self.assertTrue(all(datetime(2023, 2, 1) <= r['time'] <= datetime(2023, 8, 1) 
                           for r in results))
    
    # ========================================================================
    # Test 4: 多跳路径查询
    # ========================================================================
    
    def test_multi_hop_query(self):
        """测试多跳路径查找"""
        # 添加更多节点和边形成路径
        dave_id = str(uuid.uuid4())
        self.kg.graph.add_node(dave_id, name="Dave", type="person")
        self.kg.graph.add_edge(self.carol_id, dave_id, 
                              relation_type="FRIEND_OF",
                              valid_from=datetime(2022, 3, 1),
                              confidence=1.0)
        self.kg.graph.add_edge(dave_id, self.techcorp_id,
                              relation_type="WORKS_AT",
                              valid_from=datetime(2023, 4, 1),
                              confidence=1.0)
        
        # 查询 Alice 到 TechCorp 的路径（应该有多条）
        paths = self.kg.multi_hop_query(self.alice_id, self.techcorp_id, max_hops=3)
        
        # 应该至少有一条路径
        self.assertGreaterEqual(len(paths), 1)
        
        # 验证路径结构
        for path in paths:
            self.assertEqual(path[0]['from_id'], self.alice_id)
            self.assertEqual(path[-1]['to_id'], self.techcorp_id)
    
    def test_multi_hop_no_path(self):
        """测试无路径情况"""
        orphan_id = str(uuid.uuid4())
        self.kg.graph.add_node(orphan_id, name="Orphan", type="person")
        
        paths = self.kg.multi_hop_query(self.alice_id, orphan_id)
        self.assertEqual(len(paths), 0)
    
    # ========================================================================
    # Test 5: 相关实体查找
    # ========================================================================
    
    def test_find_related_entities(self):
        """测试查找相关实体"""
        # 查找 Alice 的相关实体
        related = self.kg.find_related_entities(self.alice_id, max_depth=2)
        
        # 验证
        self.assertIn("FRIEND_OF", related)
        self.assertIn("WORKS_AT", related)
        
        # 应该找到 Bob 和 Carol 作为朋友
        friend_ids = [e['id'] for e in related.get("FRIEND_OF", [])]
        self.assertIn(self.bob_id, friend_ids)
        self.assertIn(self.carol_id, friend_ids)
    
    def test_find_related_entities_by_type(self):
        """测试按关系类型查找相关实体"""
        # 只查找 FRIEND_OF 关系
        related = self.kg.find_related_entities(self.alice_id, 
                                                 relation_type="FRIEND_OF",
                                                 max_depth=2)
        
        self.assertEqual(len(related), 1)
        self.assertIn("FRIEND_OF", related)
        self.assertEqual(len(related["FRIEND_OF"]), 2)  # Bob 和 Carol
    
    # ========================================================================
    # Test 6: 实体时间线
    # ========================================================================
    
    def test_get_entity_timeline(self):
        """测试获取实体时间线"""
        timeline = self.kg.get_entity_timeline(self.alice_id)
        
        # Alice 有 3 个事件 + 2 个朋友 + 1 个工作关系 = 6 个
        self.assertEqual(len(timeline), 6)
        # 验证按时间排序
        times = [e['time'] for e in timeline if e['time']]
        self.assertEqual(times, sorted(times))
    
    # ========================================================================
    # Test 7: LLM 实体/关系提取
    # ========================================================================
    
    def test_extract_entities_and_relations(self):
        """测试 LLM 实体/关系提取"""
        self.kg._user_id = self.user_id
        
        text = "Caroline worked on adoption agencies project in New York on May 15, 2023."
        
        entities, relations = self.kg.extract_entities_and_relations(text, self.mock_llm)
        
        # 验证实体
        self.assertEqual(len(entities), 3)
        entity_names = [e.name for e in entities]
        self.assertIn("Caroline", entity_names)
        self.assertIn("adoption agencies", entity_names)
        self.assertIn("New York", entity_names)
        
        # 验证关系
        self.assertEqual(len(relations), 2)
        relation_types = [r.relation_type for r in relations]
        self.assertIn("WORKED_ON", relation_types)
        self.assertIn("LOCATED_IN", relation_types)
    
    # ========================================================================
    # Test 8: 统计信息
    # ========================================================================
    
    def test_get_statistics(self):
        """测试获取图谱统计信息"""
        stats = self.kg.get_statistics()
        
        self.assertEqual(stats['node_count'], 8)
        self.assertGreaterEqual(stats['edge_count'], 7)
        self.assertIn('person', stats['entity_types'])
        self.assertIn('organization', stats['entity_types'])
        self.assertIn('event', stats['entity_types'])
        self.assertIn('FRIEND_OF', stats['relation_types'])
        self.assertIn('WORKS_AT', stats['relation_types'])
    
    def test_find_central_entities(self):
        """测试查找中心实体"""
        central = self.kg.find_central_entities(top_k=3)
        
        # Alice 应该有最高的中心性（连接最多）
        self.assertEqual(central[0]['id'], self.alice_id)
        self.assertGreaterEqual(central[0]['degree'], 5)
    
    # ========================================================================
    # Test 9: 导出导入
    # ========================================================================
    
    def test_export_import(self):
        """测试图谱导出导入"""
        # 导出
        data = self.kg.export_to_dict()
        
        self.assertEqual(len(data['nodes']), 8)
        self.assertGreaterEqual(len(data['edges']), 7)
        
        # 导入到新图谱
        new_kg = TemporalKnowledgeGraph(self.db)
        new_kg.import_from_dict(data, self.user_id)
        
        self.assertEqual(len(new_kg.graph.nodes), 8)
        self.assertGreaterEqual(len(new_kg.graph.edges), 7)


class TestTemporalKnowledgeGraphEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_kg_edge.db")
        init_database(self.db_path)
        self.db = MimirDatabase(self.db_path)
        self.kg = TemporalKnowledgeGraph(self.db)
        self.user_id = "test_user"
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_graph_queries(self):
        """测试空图谱查询"""
        # 各种查询都应该返回空结果
        results = self.kg.query_temporal("nonexistent", "before", datetime.now())
        self.assertEqual(len(results), 0)
        
        paths = self.kg.multi_hop_query("a", "b")
        self.assertEqual(len(paths), 0)
        
        related = self.kg.find_related_entities("nonexistent")
        self.assertEqual(len(related), 0)
    
    def test_query_nonexistent_entity(self):
        """测试查询不存在的实体"""
        timeline = self.kg.get_entity_timeline("nonexistent")
        self.assertEqual(len(timeline), 0)
    
    def test_statistics_empty_graph(self):
        """测试空图谱统计"""
        stats = self.kg.get_statistics()
        self.assertEqual(stats['node_count'], 0)
        self.assertEqual(stats['edge_count'], 0)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalKnowledgeGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalKnowledgeGraphEdgeCases))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
