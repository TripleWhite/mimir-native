"""
Test script for Mimir Memory V2 Database Layer
验证 SQLite + sqlite-vec 基础功能
"""

import os
import sys
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

# 确保能导入 mimir_v2
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.mimir_v2.database import (
    MimirDatabase,
    MemoryCreate,
    Memory,
    EntityCreate,
    Entity,
    RelationCreate,
    Relation,
    RawContentCreate,
    RawContent,
    init_database,
    serialize_embedding,
    deserialize_embedding,
)


class TestMimirDatabase(unittest.TestCase):
    """Mimir Database 测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试前准备：创建临时数据库"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "test_mimir_v2.db")
        cls.db = init_database(cls.db_path)
    
    @classmethod
    def tearDownClass(cls):
        """测试后清理：关闭并删除数据库"""
        cls.db.close()
        try:
            os.remove(cls.db_path)
            os.rmdir(cls.temp_dir)
        except:
            pass
    
    def setUp(self):
        """每个测试前清理数据（不直接删除 FTS 虚拟表）"""
        # 清理数据 - 从实体表开始（外键约束）
        self.db._execute("DELETE FROM entity_memories")
        self.db._execute("DELETE FROM relations")
        self.db._execute("DELETE FROM vec_memories")
        self.db._execute("DELETE FROM memories")
        self.db._execute("DELETE FROM entities")
        self.db._execute("DELETE FROM raw_contents")
        self.db._commit()
    
    # ========================================================================
    # 基础连接测试
    # ========================================================================
    
    def test_01_connection(self):
        """测试数据库连接是否正常"""
        self.assertIsNotNone(self.db.conn)
        cursor = self.db._execute("SELECT 1 as value")
        result = cursor.fetchone()
        # 兼容 apsw（字典）和 sqlite3（元组）
        if isinstance(result, dict):
            self.assertEqual(result['value'], 1)
        else:
            self.assertEqual(result[0], 1)
        print("✓ 数据库连接正常")
    
    def test_02_schema_exists(self):
        """测试所有表是否已创建"""
        tables = [
            'memories', 'memories_fts', 'entities', 'relations',
            'raw_contents', 'entity_memories', 'vec_memories'
        ]
        
        for table in tables:
            cursor = self.db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            )
            self.assertIsNotNone(cursor.fetchone(), f"表 {table} 不存在")
        
        print(f"✓ 所有 {len(tables)} 个表已创建")
    
    def test_03_indexes_exist(self):
        """测试索引是否已创建"""
        indexes = [
            'idx_memories_user', 'idx_memories_valid_at', 'idx_memories_source',
            'idx_relations_source', 'idx_relations_target', 'idx_relations_type'
        ]
        
        for index in indexes:
            cursor = self.db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                (index,)
            )
            self.assertIsNotNone(cursor.fetchone(), f"索引 {index} 不存在")
        
        print(f"✓ 所有 {len(indexes)} 个索引已创建")
    
    # ========================================================================
    # Memory CRUD 测试
    # ========================================================================
    
    def test_04_create_memory(self):
        """测试创建记忆"""
        memory = MemoryCreate(
            user_id="user_001",
            content="Caroline 有一个兄弟叫 Ben",
            content_hash="abc123",
            embedding=[0.1] * 1536,  # 1536维向量 (Titan Embeddings)
            valid_at=datetime(2023, 5, 7),
            valid_at_confidence=0.95,
            temporal_tags=json.dumps(["2023-05-07", "May 2023"]),
            source_type="chat",
            source_id="session_1",
            source_metadata=json.dumps({"platform": "telegram"})
        )
        
        memory_id = self.db.create_memory(memory)
        self.assertIsNotNone(memory_id)
        self.assertIsInstance(memory_id, str)
        
        print(f"✓ 记忆创建成功: {memory_id[:8]}...")
        return memory_id
    
    def test_05_get_memory(self):
        """测试获取记忆"""
        memory_id = self.test_04_create_memory()
        
        retrieved = self.db.get_memory(memory_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, memory_id)
        self.assertEqual(retrieved.user_id, "user_001")
        self.assertEqual(retrieved.content, "Caroline 有一个兄弟叫 Ben")
        # 注意：access_count 返回的是获取前的值
        self.assertEqual(retrieved.access_count, 0)
        
        print(f"✓ 记忆获取成功")
    
    def test_06_update_memory(self):
        """测试更新记忆"""
        memory_id = self.test_04_create_memory()
        
        success = self.db.update_memory(
            memory_id,
            {"content": "Caroline 有一个兄弟叫 Benjamin", "version": 2}
        )
        self.assertTrue(success)
        
        retrieved = self.db.get_memory(memory_id)
        self.assertEqual(retrieved.content, "Caroline 有一个兄弟叫 Benjamin")
        self.assertEqual(retrieved.version, 2)
        
        print(f"✓ 记忆更新成功")
    
    def test_07_delete_memory(self):
        """测试删除记忆"""
        memory_id = self.test_04_create_memory()
        
        success = self.db.delete_memory(memory_id)
        self.assertTrue(success)
        
        retrieved = self.db.get_memory(memory_id)
        self.assertIsNone(retrieved)
        
        print(f"✓ 记忆删除成功")
    
    def test_08_list_memories(self):
        """测试列出记忆"""
        # 创建多个记忆
        for i in range(5):
            memory = MemoryCreate(
                user_id="user_002",
                content=f"测试记忆 {i}",
                embedding=[0.1] * 1536
            )
            self.db.create_memory(memory)
        
        memories = self.db.list_memories(user_id="user_002", limit=10)
        self.assertEqual(len(memories), 5)
        
        print(f"✓ 记忆列表查询成功: {len(memories)} 条")
    
    # ========================================================================
    # 向量搜索测试
    # ========================================================================
    
    def test_09_vector_search(self):
        """测试向量相似度搜索"""
        # 创建多个带向量的记忆
        base_embedding = [0.1] * 1536
        
        for i in range(10):
            # 略微不同的向量
            embedding = [0.1 + i * 0.01] * 1536
            memory = MemoryCreate(
                user_id="user_003",
                content=f"向量测试记忆 {i}",
                embedding=embedding
            )
            self.db.create_memory(memory)
        
        # 搜索
        query_embedding = [0.1] * 1536
        results = self.db.vector_search(query_embedding, top_k=5, user_id="user_003")
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        
        if results:
            self.assertIn('memory', results[0])
            self.assertIn('distance', results[0])
        
        print(f"✓ 向量搜索成功: 返回 {len(results)} 条结果")
    
    # ========================================================================
    # 全文检索测试
    # ========================================================================
    
    def test_10_fts_search(self):
        """测试全文检索"""
        # 创建带文本的记忆
        texts = [
            "Caroline 有一个兄弟叫 Ben",
            "Ben 是一名医生",
            "Caroline 喜欢 Python 编程",
            "Python 是一门强大的语言"
        ]
        
        for text in texts:
            memory = MemoryCreate(
                user_id="user_004",
                content=text
            )
            self.db.create_memory(memory)
        
        # 给 FTS 一点索引时间
        self.db._commit()
        
        # 搜索
        results = self.db.fts_search("Caroline", top_k=5)
        
        self.assertIsInstance(results, list)
        
        # 应该找到包含 Caroline 的记录
        caroline_results = [r for r in results if "Caroline" in r['memory'].content]
        self.assertGreaterEqual(len(caroline_results), 1)
        
        print(f"✓ 全文检索成功: 找到 {len(caroline_results)} 条包含 Caroline 的结果")
    
    # ========================================================================
    # Entity CRUD 测试
    # ========================================================================
    
    def test_11_create_entity(self):
        """测试创建实体"""
        entity = EntityCreate(
            name="Caroline",
            type="person",
            aliases=json.dumps(["Caro", "Cara"]),
            first_seen=datetime(2023, 5, 7)
        )
        
        entity_id = self.db.create_entity(entity)
        self.assertIsNotNone(entity_id)
        
        print(f"✓ 实体创建成功: {entity_id[:8]}...")
        return entity_id
    
    def test_12_get_entity(self):
        """测试获取实体"""
        entity_id = self.test_11_create_entity()
        
        retrieved = self.db.get_entity(entity_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Caroline")
        self.assertEqual(retrieved.type, "person")
        
        print(f"✓ 实体获取成功")
    
    def test_13_get_entity_by_name(self):
        """测试按名称获取实体"""
        self.test_11_create_entity()
        
        retrieved = self.db.get_entity_by_name("Caroline")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Caroline")
        
        print(f"✓ 按名称获取实体成功")
    
    # ========================================================================
    # Relation CRUD 测试
    # ========================================================================
    
    def test_14_create_relation(self):
        """测试创建关系"""
        # 先创建两个实体
        entity1 = EntityCreate(name="Caroline", type="person")
        entity2 = EntityCreate(name="Microsoft", type="organization")
        
        entity1_id = self.db.create_entity(entity1)
        entity2_id = self.db.create_entity(entity2)
        
        # 创建关系
        relation = RelationCreate(
            source_entity=entity1_id,
            target_entity=entity2_id,
            relation_type="WORKS_AT",
            valid_from=datetime(2020, 1, 1),
            valid_until=datetime(2023, 12, 31),
            evidence_memory_ids=json.dumps(["mem_001", "mem_002"]),
            confidence=0.95
        )
        
        relation_id = self.db.create_relation(relation)
        self.assertIsNotNone(relation_id)
        
        print(f"✓ 关系创建成功: {relation_id[:8]}...")
        return relation_id
    
    def test_15_get_relations_by_entity(self):
        """测试获取实体的关系"""
        # 创建实体和关系
        entity1 = EntityCreate(name="Caroline", type="person")
        entity2 = EntityCreate(name="Ben", type="person")
        
        entity1_id = self.db.create_entity(entity1)
        entity2_id = self.db.create_entity(entity2)
        
        relation = RelationCreate(
            source_entity=entity1_id,
            target_entity=entity2_id,
            relation_type="SIBLING_OF"
        )
        
        self.db.create_relation(relation)
        
        # 获取关系
        relations = self.db.get_relations_by_entity(entity1_id)
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].relation_type, "SIBLING_OF")
        
        print(f"✓ 实体关系查询成功: {len(relations)} 条关系")
    
    # ========================================================================
    # Raw Content CRUD 测试
    # ========================================================================
    
    def test_16_create_raw_content(self):
        """测试创建原始内容"""
        content = RawContentCreate(
            user_id="user_005",
            content_type="conversation",
            raw_text="Session 1: Caroline said she has a brother.",
            metadata=json.dumps({"session_id": "session_1", "platform": "telegram"}),
            occurred_at=datetime(2023, 5, 7, 14, 30)
        )
        
        content_id = self.db.create_raw_content(content)
        self.assertIsNotNone(content_id)
        
        print(f"✓ 原始内容创建成功: {content_id[:8]}...")
        return content_id
    
    def test_17_update_raw_content(self):
        """测试更新原始内容"""
        content_id = self.test_16_create_raw_content()
        
        success = self.db.update_raw_content(
            content_id,
            {"processed": True, "extracted_memory_ids": json.dumps(["mem_001"])}
        )
        self.assertTrue(success)
        
        retrieved = self.db.get_raw_content(content_id)
        self.assertTrue(retrieved.processed)
        
        print(f"✓ 原始内容更新成功")
    
    # ========================================================================
    # Entity-Memory 关联测试
    # ========================================================================
    
    def test_18_associate_entity_memory(self):
        """测试实体-记忆关联"""
        # 创建记忆和实体
        memory = MemoryCreate(
            user_id="user_006",
            content="Caroline works at Microsoft"
        )
        memory_id = self.db.create_memory(memory)
        
        entity = EntityCreate(name="Caroline", type="person")
        entity_id = self.db.create_entity(entity)
        
        # 关联
        success = self.db.associate_entity_memory(entity_id, memory_id)
        self.assertTrue(success)
        
        # 查询实体的记忆
        memories = self.db.get_entity_memories(entity_id)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].id, memory_id)
        
        # 查询记忆的实体
        entities = self.db.get_memory_entities(memory_id)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].id, entity_id)
        
        print(f"✓ 实体-记忆关联成功")
    
    # ========================================================================
    # 统计信息测试
    # ========================================================================
    
    def test_19_get_stats(self):
        """测试统计信息"""
        # 创建一些数据
        self.db.create_memory(MemoryCreate(user_id="user_007", content="test"))
        self.db.create_entity(EntityCreate(name="TestEntity"))
        
        stats = self.db.get_stats()
        
        self.assertIn('memory_count', stats)
        self.assertIn('entity_count', stats)
        self.assertIn('relation_count', stats)
        self.assertIn('raw_content_count', stats)
        self.assertIn('vector_count', stats)
        
        print(f"✓ 统计信息获取成功: {stats}")
    
    # ========================================================================
    # 工具函数测试
    # ========================================================================
    
    def test_20_embedding_serialization(self):
        """测试向量序列化"""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # 序列化
        blob = serialize_embedding(embedding)
        self.assertIsInstance(blob, bytes)
        
        # 反序列化
        restored = deserialize_embedding(blob)
        self.assertEqual(len(restored), len(embedding))
        
        for i, val in enumerate(embedding):
            self.assertAlmostEqual(restored[i], val, places=5)
        
        print(f"✓ 向量序列化/反序列化成功")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Mimir Memory V2 Database Layer Tests")
    print("=" * 60)
    print()
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMimirDatabase)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    if result.wasSuccessful():
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
