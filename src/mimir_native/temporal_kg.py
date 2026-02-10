"""
Mimir Memory V2 - Temporal Knowledge Graph

时序知识图谱 - NetworkX 实现
用于时序推理、多跳关联、实体关系追踪
"""

import json
import uuid
import networkx as nx
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)

# 导入数据库模型
from .database import MimirDatabase, Entity, Relation, EntityCreate, RelationCreate, Memory


class TemporalKnowledgeGraph:
    """时序知识图谱 - NetworkX 实现"""

    def __init__(self, db: MimirDatabase):
        """
        初始化时序知识图谱

        Args:
            db: MimirDatabase 实例
        """
        self.db = db
        self.graph = nx.DiGraph()  # 有向图支持时序
        self._entity_cache: Dict[str, Entity] = {}  # 实体缓存
        self._user_id: Optional[str] = None

    def build_from_memories(self, user_id: str) -> None:
        """
        从数据库加载构建图谱

        Args:
            user_id: 用户 ID
        """
        self._user_id = user_id
        self.graph.clear()
        self._entity_cache.clear()

        # 从数据库查询所有实体（通过 memories 的关联）
        cursor = self.db._execute(
            """SELECT DISTINCT e.* FROM entities e
               JOIN entity_memories em ON e.id = em.entity_id
               JOIN memories m ON em.memory_id = m.id
               WHERE m.user_id = ?""",
            (user_id,)
        )

        for row in cursor.fetchall():
            entity = self.db._row_to_entity(row)
            self.graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.type,
                aliases=entity.aliases,
                first_seen=entity.first_seen,
                last_seen=entity.last_seen,
                mention_count=entity.mention_count
            )
            self._entity_cache[entity.id] = entity

        # 加载所有关系
        # 查询涉及当前用户实体的所有关系
        cursor = self.db._execute(
            """SELECT DISTINCT r.* FROM relations r
               WHERE r.source_entity IN (
                   SELECT DISTINCT e.id FROM entities e
                   JOIN entity_memories em ON e.id = em.entity_id
                   JOIN memories m ON em.memory_id = m.id
                   WHERE m.user_id = ?
               ) OR r.target_entity IN (
                   SELECT DISTINCT e.id FROM entities e
                   JOIN entity_memories em ON e.id = em.entity_id
                   JOIN memories m ON em.memory_id = m.id
                   WHERE m.user_id = ?
               )""",
            (user_id, user_id)
        )

        for row in cursor.fetchall():
            relation = self.db._row_to_relation(row)
            # 只有当源实体和目标实体都存在时才添加边
            if relation.source_entity in self.graph and relation.target_entity in self.graph:
                self.graph.add_edge(
                    relation.source_entity,
                    relation.target_entity,
                    relation_type=relation.relation_type,
                    valid_from=relation.valid_from,
                    valid_until=relation.valid_until,
                    evidence=relation.evidence_memory_ids,
                    confidence=relation.confidence,
                    relation_id=relation.id
                )

        logger.info(f"图谱构建完成: {len(self.graph.nodes)} 实体, {len(self.graph.edges)} 关系")

    def add_fact(self, memory: Memory, entities: List[Entity],
                 relations: List[Relation]) -> None:
        """
        添加事实到图谱

        Args:
            memory: 记忆对象
            entities: 实体列表
            relations: 关系列表
        """
        # 添加/更新实体节点
        for entity in entities:
            if entity.id not in self.graph:
                self.graph.add_node(
                    entity.id,
                    name=entity.name,
                    type=entity.type,
                    aliases=entity.aliases,
                    first_seen=entity.first_seen or memory.valid_at,
                    last_seen=entity.last_seen or memory.valid_at,
                    mention_count=entity.mention_count
                )
                self._entity_cache[entity.id] = entity
            else:
                # 更新实体时间范围
                self._update_entity_timeline(entity, memory.valid_at)

        # 添加关系边
        for rel in relations:
            self.graph.add_edge(
                rel.source_entity,
                rel.target_entity,
                relation_type=rel.relation_type,
                valid_from=rel.valid_from or memory.valid_at,
                valid_until=rel.valid_until,
                evidence=rel.evidence_memory_ids or json.dumps([memory.id]),
                confidence=rel.confidence,
                relation_id=rel.id
            )

    def _update_entity_timeline(self, entity: Entity, memory_time: Optional[datetime]) -> None:
        """更新实体的时间范围"""
        if not memory_time:
            return

        from dateutil import parser as date_parser

        # 确保 memory_time 是 datetime 对象
        if isinstance(memory_time, str):
            memory_time = date_parser.parse(memory_time)

        node_data = self.graph.nodes[entity.id]

        first_seen = node_data.get('first_seen')
        last_seen = node_data.get('last_seen')

        # 确保从数据库/图读取的也是 datetime
        if isinstance(first_seen, str):
            first_seen = date_parser.parse(first_seen)
        if isinstance(last_seen, str):
            last_seen = date_parser.parse(last_seen)

        if first_seen is None or memory_time < first_seen:
            self.graph.nodes[entity.id]['first_seen'] = memory_time

        if last_seen is None or memory_time > last_seen:
            self.graph.nodes[entity.id]['last_seen'] = memory_time

    def _parse_datetime(self, value):
        """将 ISO 字符串或 datetime 统一转换为 datetime 对象"""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            # 尝试多种格式解析
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            # 尝试 ISO format (with timezone handling)
            try:
                # Handle 'Z' suffix for UTC
                if value.endswith('Z'):
                    value = value[:-1] + '+00:00'
                result = datetime.fromisoformat(value)
                # 如果结果带时区信息，转换为本地时间（无时区）以便比较
                if result.tzinfo is not None:
                    result = result.replace(tzinfo=None)
                return result
            except (ValueError, TypeError):
                pass
        return value  # 如果无法解析，原样返回

    # ========================================================================
    # 时序查询
    # ========================================================================

    def query_temporal(self, entity_id: str, query_type: str,
                       reference_time: datetime = None) -> List[Dict]:
        """
        时序查询

        Args:
            entity_id: 实体 ID
            query_type: 查询类型
                - "before": 某时间点之前发生的事
                - "after": 某时间点之后
                - "at": 特定时间点
                - "between": 时间段内 (需要 start_time, end_time 参数)
            reference_time: 参考时间

        Returns:
            List[Dict]: 事件列表，按时间排序
        """
        if entity_id not in self.graph:
            return []

        # 获取实体相关的所有事件
        events = []

        # 遍历出边
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            event = self._extract_event_data(target, data)
            if event:
                events.append(event)

        # 解析 reference_time（支持字符串或 datetime）
        ref_time = self._parse_datetime(reference_time) if reference_time else None

        # 时序过滤
        if query_type == 'before' and ref_time:
            events = [e for e in events
                     if e.get('time') and self._parse_datetime(e['time']) < ref_time]
        elif query_type == 'after' and ref_time:
            events = [e for e in events
                     if e.get('time') and self._parse_datetime(e['time']) > ref_time]
        elif query_type == 'at' and ref_time:
            # 查找特定时间点的事件（允许 1 天误差）
            from datetime import timedelta
            events = [e for e in events
                     if e.get('time') and abs((self._parse_datetime(e['time']) - ref_time).total_seconds()) < 86400]

        # 按时间排序（确保时间都被解析为 datetime）
        events.sort(key=lambda x: self._parse_datetime(x.get('time')) or datetime.min)
        return events

    def query_between(self, entity_id: str,
                      start_time: datetime, end_time: datetime) -> List[Dict]:
        """
        查询时间段内的事件

        Args:
            entity_id: 实体 ID
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[Dict]: 事件列表
        """
        if entity_id not in self.graph:
            return []

        # 解析时间参数（支持字符串或 datetime）
        start = self._parse_datetime(start_time)
        end = self._parse_datetime(end_time)

        events = []
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            event_time = self._parse_datetime(data.get('valid_from'))
            if event_time and start <= event_time <= end:
                event = self._extract_event_data(target, data)
                if event:
                    events.append(event)

        events.sort(key=lambda x: self._parse_datetime(x.get('time')) or datetime.min)
        return events

    def _extract_event_data(self, target_entity_id: str, edge_data: Dict) -> Optional[Dict]:
        """从边数据中提取事件信息"""
        if target_entity_id not in self.graph:
            return None

        target_node = self.graph.nodes[target_entity_id]
        relation_type = edge_data.get('relation_type', 'UNKNOWN')

        # 判断是否是事件类型的关系
        event_relations = ['PARTICIPATED_IN', 'HAPPENED_AT', 'OCCURRED_ON',
                          'ATTENDED', 'VISITED', 'WORKED_ON', 'CREATED',
                          'BORN_ON', 'DIED_ON', 'FOUNDED', 'ACQUIRED']

        # 解析时间字段（支持字符串或 datetime）
        raw_time = edge_data.get('valid_from')
        parsed_time = self._parse_datetime(raw_time)

        return {
            'entity_id': target_entity_id,
            'entity_name': target_node.get('name'),
            'entity_type': target_node.get('type'),
            'time': parsed_time,
            'relation': relation_type,
            'evidence': edge_data.get('evidence'),
            'confidence': edge_data.get('confidence', 1.0),
            'is_event': relation_type in event_relations
        }

    # ========================================================================
    # 多跳查询
    # ========================================================================

    def multi_hop_query(self, start_entity: str, end_entity: str,
                        max_hops: int = 3) -> List[List[Dict]]:
        """
        多跳路径查找

        例: "Caroline" → "adoption agencies" (通过 WORKED_ON 关系)

        Args:
            start_entity: 起始实体 ID
            end_entity: 目标实体 ID
            max_hops: 最大跳数

        Returns:
            List[List[Dict]]: 所有可能的路径列表，每条路径是边信息的列表
        """
        if start_entity not in self.graph or end_entity not in self.graph:
            return []

        try:
            # 使用 NetworkX 查找所有简单路径
            paths = list(nx.all_simple_paths(
                self.graph, start_entity, end_entity, cutoff=max_hops
            ))

            result = []
            for path in paths:
                path_info = []
                for i in range(len(path) - 1):
                    edge_data = self.graph.edges[path[i], path[i+1]]

                    # 获取节点名称
                    from_name = self.graph.nodes[path[i]].get('name', path[i])
                    to_name = self.graph.nodes[path[i+1]].get('name', path[i+1])

                    path_info.append({
                        'from_id': path[i],
                        'from_name': from_name,
                        'to_id': path[i+1],
                        'to_name': to_name,
                        'relation': edge_data.get('relation_type'),
                        'time': edge_data.get('valid_from'),
                        'evidence': edge_data.get('evidence'),
                        'confidence': edge_data.get('confidence', 1.0)
                    })
                result.append(path_info)

            return result
        except nx.NetworkXNoPath:
            return []

    def find_related_entities(self, entity_id: str,
                              relation_type: str = None,
                              max_depth: int = 2) -> Dict[str, List[Dict]]:
        """
        查找相关实体（按关系类型分组）

        Args:
            entity_id: 实体 ID
            relation_type: 可选，按特定关系类型过滤
            max_depth: 最大搜索深度

        Returns:
            Dict[str, List[Dict]]: 按关系类型分组的相关实体
        """
        if entity_id not in self.graph:
            return {}

        related: Dict[str, List[Dict]] = {}

        # BFS 遍历
        visited: Set[str] = {entity_id}
        queue: List[Tuple[str, int]] = [(entity_id, 0)]

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue

                edge_data = self.graph.edges[current, neighbor]
                rel_type = edge_data.get('relation_type', 'UNKNOWN')

                if relation_type and rel_type != relation_type:
                    continue

                neighbor_node = self.graph.nodes[neighbor]
                entity_info = {
                    'id': neighbor,
                    'name': neighbor_node.get('name'),
                    'type': neighbor_node.get('type'),
                    'depth': depth + 1,
                    'time': edge_data.get('valid_from')
                }

                if rel_type not in related:
                    related[rel_type] = []
                related[rel_type].append(entity_info)

                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

        return related

    def get_entity_timeline(self, entity_id: str) -> List[Dict]:
        """
        获取实体的时间线

        Args:
            entity_id: 实体 ID

        Returns:
            List[Dict]: 按时间排序的事件列表
        """
        if entity_id not in self.graph:
            return []

        events = []

        # 遍历所有出边
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            event = self._extract_event_data(target, data)
            if event:
                events.append(event)

        # 按时间排序
        events.sort(key=lambda x: x.get('time') or datetime.min)
        return events

    # ========================================================================
    # 实体提取与关系抽取（使用 Bedrock LLM）
    # ========================================================================

    def extract_entities_and_relations(self, text: str,
                                       llm_client) -> Tuple[List[Entity], List[Relation]]:
        """
        使用 Bedrock LLM 从文本中提取实体和关系

        Args:
            text: 输入文本
            llm_client: LLM 客户端（需要有 invoke_mistral 方法）

        Returns:
            Tuple[List[Entity], List[Relation]]: 提取的实体和关系
        """
        prompt = f"""从以下文本中提取实体和关系。

文本: {text}

要求:
1. 识别所有实体（人、组织、地点、事件、概念）
2. 识别实体之间的关系
3. 标注时间信息（如果有）

输出 JSON 格式:
{{
  "entities": [
    {{"name": "实体名", "type": "person|organization|location|event|concept", "aliases": ["别名"]}}
  ],
  "relations": [
    {{"source": "实体1", "target": "实体2", "relation_type": "WORKS_AT|LOCATED_IN|FRIEND_OF|PARTICIPATED_IN|...", "time": "2023-05-07"}}
  ]
}}

如果没有关系，relations 为空数组。
只输出 JSON，不要其他文字说明。"""

        try:
            # 使用 Mistral 模型（非 Claude）
            response = llm_client.invoke_mistral(prompt, max_tokens=2000)
            data = self._parse_llm_response(response)

            # 创建 Entity 对象
            entities = []
            for e_data in data.get('entities', []):
                # 检查是否已存在同名实体
                existing = self.db.get_entity_by_name(e_data['name'])
                if existing:
                    entities.append(existing)
                    continue

                entity_create = EntityCreate(
                    name=e_data['name'],
                    type=e_data.get('type'),
                    aliases=json.dumps(e_data.get('aliases', [])),
                    first_seen=datetime.now(),
                    last_seen=datetime.now()
                )
                # 创建实体并获取 ID
                entity_id = self.db.create_entity(entity_create)
                entity = Entity(
                    id=entity_id,
                    name=entity_create.name,
                    type=entity_create.type,
                    aliases=entity_create.aliases,
                    first_seen=entity_create.first_seen,
                    last_seen=entity_create.last_seen,
                    mention_count=1
                )
                entities.append(entity)

            # 创建 Relation 对象
            entity_name_to_id = {e.name: e.id for e in entities}
            relations = []
            for r_data in data.get('relations', []):
                source_name = r_data['source']
                target_name = r_data['target']

                if source_name in entity_name_to_id and target_name in entity_name_to_id:
                    # 解析时间
                    valid_from = None
                    if r_data.get('time'):
                        try:
                            valid_from = datetime.fromisoformat(r_data['time'].replace('Z', '+00:00'))
                        except ValueError:
                            pass

                    relation_create = RelationCreate(
                        source_entity=entity_name_to_id[source_name],
                        target_entity=entity_name_to_id[target_name],
                        relation_type=r_data['relation_type'],
                        valid_from=valid_from,
                        valid_until=None,
                        evidence_memory_ids=json.dumps([]),
                        confidence=1.0
                    )
                    relation_id = self.db.create_relation(relation_create)
                    relation = Relation(
                        id=relation_id,
                        source_entity=relation_create.source_entity,
                        target_entity=relation_create.target_entity,
                        relation_type=relation_create.relation_type,
                        valid_from=relation_create.valid_from,
                        valid_until=relation_create.valid_until,
                        evidence_memory_ids=relation_create.evidence_memory_ids,
                        confidence=relation_create.confidence,
                        created_at=datetime.now()
                    )
                    relations.append(relation)

            return entities, relations

        except Exception as e:
            logger.error(f"实体关系提取失败: {e}")
            return [], []

    def _parse_llm_response(self, response: str) -> Dict:
        """解析 LLM 返回的 JSON"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试从 markdown 代码块中提取
            try:
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0]
                    return json.loads(json_str.strip())
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0]
                    return json.loads(json_str.strip())
            except Exception:
                pass

            # 尝试提取 {} 包裹的内容
            try:
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    return json.loads(response[start:end+1])
            except Exception:
                pass

            logger.error(f"无法解析 LLM 响应: {response[:200]}")
            return {'entities': [], 'relations': []}

    # ========================================================================
    # 持久化
    # ========================================================================

    def save_to_db(self, user_id: str) -> Dict[str, int]:
        """
        将图谱持久化到数据库

        Args:
            user_id: 用户 ID

        Returns:
            Dict: 统计信息
        """
        entity_count = 0
        relation_count = 0

        # 保存所有实体
        for node_id, node_data in self.graph.nodes(data=True):
            # 检查实体是否已存在
            existing = self.db.get_entity(node_id)
            if existing:
                # 更新实体
                self.db.update_entity(node_id, {
                    'last_seen': node_data.get('last_seen'),
                    'mention_count': node_data.get('mention_count', 1)
                })
            else:
                # 新实体，创建
                entity_create = EntityCreate(
                    name=node_data.get('name', 'Unknown'),
                    type=node_data.get('type'),
                    aliases=json.dumps(node_data.get('aliases', [])),
                    first_seen=node_data.get('first_seen'),
                    last_seen=node_data.get('last_seen')
                )
                self.db.create_entity(entity_create)
                entity_count += 1

        # 保存所有关系
        for source, target, edge_data in self.graph.edges(data=True):
            relation_id = edge_data.get('relation_id')
            if relation_id:
                existing = self.db.get_relation(relation_id)
                if existing:
                    continue  # 已存在，跳过

            relation_create = RelationCreate(
                source_entity=source,
                target_entity=target,
                relation_type=edge_data.get('relation_type', 'UNKNOWN'),
                valid_from=edge_data.get('valid_from'),
                valid_until=edge_data.get('valid_until'),
                evidence_memory_ids=edge_data.get('evidence'),
                confidence=edge_data.get('confidence', 1.0)
            )
            self.db.create_relation(relation_create)
            relation_count += 1

        logger.info(f"图谱持久化完成: {entity_count} 实体, {relation_count} 关系")
        return {'entities': entity_count, 'relations': relation_count}

    # ========================================================================
    # 统计与分析
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        return {
            'node_count': len(self.graph.nodes),
            'edge_count': len(self.graph.edges),
            'entity_types': self._count_entity_types(),
            'relation_types': self._count_relation_types(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph) if len(self.graph.nodes) > 0 else False
        }

    def _count_entity_types(self) -> Dict[str, int]:
        """统计实体类型分布"""
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            entity_type = data.get('type', 'unknown')
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts

    def _count_relation_types(self) -> Dict[str, int]:
        """统计关系类型分布"""
        type_counts = {}
        for _, _, data in self.graph.edges(data=True):
            relation_type = data.get('relation_type', 'unknown')
            type_counts[relation_type] = type_counts.get(relation_type, 0) + 1
        return type_counts

    def find_central_entities(self, top_k: int = 10) -> List[Dict]:
        """
        找到图谱中的中心实体（按度中心性）

        Args:
            top_k: 返回前 K 个

        Returns:
            List[Dict]: 中心实体列表
        """
        if len(self.graph.nodes) == 0:
            return []

        # 计算度中心性
        centrality = nx.degree_centrality(self.graph)

        # 排序并取 top_k
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result = []
        for node_id, score in sorted_nodes:
            node_data = self.graph.nodes[node_id]
            result.append({
                'id': node_id,
                'name': node_data.get('name'),
                'type': node_data.get('type'),
                'centrality_score': score,
                'degree': self.graph.degree(node_id)
            })

        return result

    def export_to_dict(self) -> Dict:
        """导出图谱为字典格式"""
        return {
            'nodes': [
                {'id': node, **data}
                for node, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    **data
                }
                for source, target, data in self.graph.edges(data=True)
            ]
        }

    def import_from_dict(self, data: Dict, user_id: str) -> None:
        """从字典导入图谱"""
        self._user_id = user_id
        self.graph.clear()

        # 导入节点
        for node_data in data.get('nodes', []):
            node_id = node_data.pop('id')
            self.graph.add_node(node_id, **node_data)

        # 导入边
        for edge_data in data.get('edges', []):
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            self.graph.add_edge(source, target, **edge_data)
