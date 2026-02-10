"""
Test to reproduce datetime string bug in temporal queries

This test demonstrates the bug when datetime values from database 
are ISO strings instead of datetime objects.
"""
import os
import sys
import json
import uuid
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径（确保 app.mimir_v2 可导入）
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # 指向 backend 目录

from app.mimir_v2.temporal_kg import TemporalKnowledgeGraph
from app.mimir_v2.database import MimirDatabase, init_database


class TestDatetimeStringBug(unittest.TestCase):
    """Test to reproduce the datetime string comparison bug"""

    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_bug.db")
        init_database(self.db_path)
        self.db = MimirDatabase(self.db_path)
        self.kg = TemporalKnowledgeGraph(self.db)
        self.user_id = "test_user"

    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bug_with_iso_string_datetime(self):
        """
        重现 Bug: 当 valid_from 是 ISO 字符串时，时序查询会失败

        错误信息: TypeError: '<' not supported between instances of 'str' and 'datetime.datetime'
        """
        # 创建测试实体 ID
        alice_id = str(uuid.uuid4())
        event_id = str(uuid.uuid4())

        # 添加节点
        self.kg.graph.add_node(alice_id, name="Alice", type="person", mention_count=1)
        self.kg.graph.add_node(event_id, name="Conference", type="event", mention_count=1)

        # 添加边，但使用 ISO 字符串格式的时间（模拟从数据库加载的情况）
        # 数据库通常以 ISO 格式存储 datetime
        iso_time = "2023-03-15T10:30:00"
        self.kg.graph.add_edge(alice_id, event_id,
                              relation_type="PARTICIPATED_IN",
                              valid_from=iso_time,  # ISO 字符串而不是 datetime 对象
                              confidence=1.0)

        # 尝试进行时序查询 - 这会触发 TypeError
        try:
            results = self.kg.query_temporal(alice_id, "before", datetime(2023, 4, 1))
            # 如果修复了 bug，应该能正常返回结果
            self.assertEqual(len(results), 1)
            print("✅ Bug fixed! query_temporal works with ISO string datetime")
        except TypeError as e:
            self.fail(f"❌ Bug reproduced! TypeError: {e}")

    def test_bug_query_between_with_iso_string(self):
        """
        重现 Bug: query_between 也会因 ISO 字符串而失败
        """
        alice_id = str(uuid.uuid4())
        event_id = str(uuid.uuid4())

        self.kg.graph.add_node(alice_id, name="Alice", type="person", mention_count=1)
        self.kg.graph.add_node(event_id, name="Conference", type="event", mention_count=1)

        # 使用 ISO 字符串格式的时间
        iso_time = "2023-03-15T10:30:00"
        self.kg.graph.add_edge(alice_id, event_id,
                              relation_type="PARTICIPATED_IN",
                              valid_from=iso_time,
                              confidence=1.0)

        # 尝试进行 between 查询
        try:
            results = self.kg.query_between(alice_id, datetime(2023, 2, 1), datetime(2023, 8, 1))
            # 如果修复了 bug，应该能正常返回结果
            self.assertEqual(len(results), 1)
            print("✅ Bug fixed! query_between works with ISO string datetime")
        except TypeError as e:
            self.fail(f"❌ Bug reproduced! TypeError: {e}")


class TestIntegrationWithRealDatabase(unittest.TestCase):
    """集成测试：从真实数据库加载数据并进行时序查询"""

    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_integration.db")
        init_database(self.db_path)
        self.db = MimirDatabase(self.db_path)
        self.kg = TemporalKnowledgeGraph(self.db)
        self.user_id = "integration_test_user"

        # 设置真实图谱数据（模拟从数据库加载 ISO 字符串 datetime）
        self._setup_test_data_with_iso_strings()

    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _setup_test_data_with_iso_strings(self):
        """
        设置测试数据，模拟从数据库加载 ISO 字符串 datetime 的情况

        数据库通常以 ISO 格式 (YYYY-MM-DDTHH:MM:SS) 存储 datetime，
        当加载到 Python 时，它们成为字符串而不是 datetime 对象。
        """
        # 创建测试实体 ID
        self.alice_id = str(uuid.uuid4())
        self.bob_id = str(uuid.uuid4())
        self.carol_id = str(uuid.uuid4())
        self.event1_id = str(uuid.uuid4())
        self.event2_id = str(uuid.uuid4())
        self.event3_id = str(uuid.uuid4())

        # 添加节点
        self.kg.graph.add_node(self.alice_id, name="Alice", type="person", mention_count=1)
        self.kg.graph.add_node(self.bob_id, name="Bob", type="person", mention_count=1)
        self.kg.graph.add_node(self.carol_id, name="Carol", type="person", mention_count=1)
        self.kg.graph.add_node(self.event1_id, name="Conference", type="event", mention_count=1)
        self.kg.graph.add_node(self.event2_id, name="Meeting", type="event", mention_count=1)
        self.kg.graph.add_node(self.event3_id, name="Workshop", type="event", mention_count=1)

        # 添加边，使用 ISO 字符串格式的时间（模拟从数据库加载的情况）
        # 各种 ISO 格式变体
        self.kg.graph.add_edge(self.alice_id, self.event1_id,
                              relation_type="PARTICIPATED_IN",
                              valid_from="2023-03-15T10:30:00",  # ISO 格式
                              confidence=1.0)
        self.kg.graph.add_edge(self.alice_id, self.event2_id,
                              relation_type="PARTICIPATED_IN",
                              valid_from="2023-01-10T14:00:00",
                              confidence=1.0)
        self.kg.graph.add_edge(self.alice_id, self.event3_id,
                              relation_type="PARTICIPATED_IN",
                              valid_from="2023-06-20T09:00:00",
                              confidence=1.0)
        self.kg.graph.add_edge(self.alice_id, self.bob_id,
                              relation_type="FRIEND_OF",
                              valid_from="2022-01-01T00:00:00",
                              confidence=1.0)
        self.kg.graph.add_edge(self.alice_id, self.carol_id,
                              relation_type="FRIEND_OF",
                              valid_from="2022-06-01T00:00:00",
                              confidence=1.0)

    def test_temporal_queries_with_iso_strings(self):
        """
        集成测试：使用 ISO 字符串时间进行时序查询

        验证所有时序查询方法在处理 ISO 字符串 datetime 时能正常工作。
        """
        # 测试 query_temporal - before
        results_before = self.kg.query_temporal(self.alice_id, "before", datetime(2023, 4, 1))
        # 4 events: Meeting(1/10), Conference(3/15), Bob friend(2022), Carol friend(2022)
        self.assertEqual(len(results_before), 4, "Should find 4 events before April 1")

        # 测试 query_temporal - after
        results_after = self.kg.query_temporal(self.alice_id, "after", datetime(2023, 3, 1))
        # 2 events: Conference(3/15) 和 Workshop(6/20)
        self.assertEqual(len(results_after), 2, "Should find 2 events after March 1")

        # 测试 query_between
        results_between = self.kg.query_between(
            self.alice_id,
            datetime(2023, 2, 1),
            datetime(2023, 8, 1)
        )
        self.assertEqual(len(results_between), 2, "Should find 2 events between Feb and Aug")

        # 测试 get_entity_timeline
        timeline = self.kg.get_entity_timeline(self.alice_id)
        self.assertEqual(len(timeline), 5, "Timeline should have 5 events")

        # 验证时间排序正确
        times = [e['time'] for e in timeline if e['time']]
        self.assertEqual(times, sorted(times), "Events should be sorted by time")

        # 验证返回的时间是 datetime 对象，不是字符串
        for event in timeline:
            if event['time']:
                self.assertIsInstance(event['time'], datetime,
                    f"Event time should be datetime object, got {type(event['time'])}")

        print("✅ Integration test passed! Temporal queries work with ISO string datetime")

    def test_various_iso_formats(self):
        """测试各种 ISO 格式的时间字符串"""
        test_id = str(uuid.uuid4())
        event_id = str(uuid.uuid4())

        self.kg.graph.add_node(test_id, name="TestPerson", type="person", mention_count=1)
        self.kg.graph.add_node(event_id, name="TestEvent", type="event", mention_count=1)

        # 测试不同 ISO 格式
        formats_to_test = [
            "2023-03-15T10:30:00",      # 标准 ISO
            "2023-03-15T10:30:00.000",  # 带毫秒
            "2023-03-15T10:30:00Z",     # UTC
            "2023-03-15 10:30:00",      # 空格分隔
            "2023-03-15",               # 仅日期
        ]

        for i, fmt in enumerate(formats_to_test):
            event_id = str(uuid.uuid4())
            self.kg.graph.add_node(event_id, name=f"Event{i}", type="event", mention_count=1)
            self.kg.graph.add_edge(test_id, event_id,
                                  relation_type="ATTENDED",
                                  valid_from=fmt,
                                  confidence=1.0)

        # 查询应该成功
        results = self.kg.query_temporal(test_id, "before", datetime(2023, 12, 31))
        self.assertEqual(len(results), len(formats_to_test),
            f"Should find all {len(formats_to_test)} events with different ISO formats")

        print(f"✅ Successfully parsed {len(formats_to_test)} different ISO datetime formats")


if __name__ == '__main__':
    unittest.main(verbosity=2)
