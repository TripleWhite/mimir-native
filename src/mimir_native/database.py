"""
Mimir Memory V2 Database Layer
SQLite + sqlite-vec implementation
"""

import uuid
import json
import struct
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import threading

# 优先使用 apsw 以支持扩展加载
try:
    import apsw as sqlite3
    USE_APSW = True
except ImportError:
    import sqlite3
    USE_APSW = False

import sqlite_vec


# ============================================================================
# sqlite-vec 兼容性处理
# ============================================================================

def load_sqlite_vec(conn):
    """
    加载 sqlite-vec 扩展
    """
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except AttributeError:
        # 某些 Python 构建不支持扩展加载
        # 使用 sqlite-vec 的替代加载方式
        import sys
        print("Warning: sqlite3 extension loading not available", file=sys.stderr)
        # 尝试通过 sqlite_vec 的自动加载
        try:
            sqlite_vec.load(conn)
        except Exception as e:
            print(f"Warning: Could not load sqlite-vec: {e}", file=sys.stderr)
    return conn


def adapt_datetime(dt):
    """将 datetime 转换为 SQLite 可存储的格式"""
    if dt is None:
        return None
    return dt.isoformat()


def convert_datetime(s):
    """将 SQLite 存储的字符串转换回 datetime"""
    if s is None:
        return None
    if isinstance(s, datetime):
        return s
    try:
        # 处理 ISO 格式字符串
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except:
        return None


def adapt_params(params):
    """适配参数，转换 datetime 为 ISO 格式"""
    return tuple(adapt_datetime(p) if isinstance(p, datetime) else p for p in params)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class MemoryCreate:
    """用于创建记忆的输入模型"""
    user_id: str
    content: str
    content_hash: Optional[str] = None
    embedding: Optional[List[float]] = None
    valid_at: Optional[datetime] = None
    valid_at_confidence: float = 1.0
    temporal_tags: Optional[str] = None  # JSON string
    source_type: Optional[str] = None  # chat | document | image | audio
    source_id: Optional[str] = None
    source_metadata: Optional[str] = None  # JSON string


@dataclass
class Memory:
    """完整的记忆模型"""
    id: str
    user_id: str
    content: str
    content_hash: Optional[str]
    embedding: Optional[bytes]  # 序列化的向量
    valid_at: Optional[datetime]
    valid_at_confidence: float
    temporal_tags: Optional[str]
    source_type: Optional[str]
    source_id: Optional[str]
    source_metadata: Optional[str]
    created_at: datetime
    updated_at: datetime
    access_count: int
    last_accessed: Optional[datetime]
    version: int
    superseded_by: Optional[str]
    fts_docid: Optional[int]


@dataclass
class EntityCreate:
    """用于创建实体的输入模型"""
    name: str
    type: Optional[str] = None  # person | organization | location | event | concept
    aliases: Optional[str] = None  # JSON string ["alias1", "alias2"]
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


@dataclass
class Entity:
    """完整的实体模型"""
    id: str
    name: str
    type: Optional[str]
    aliases: Optional[str]
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    mention_count: int


@dataclass
class RelationCreate:
    """用于创建关系的输入模型"""
    source_entity: str  # entity.id
    target_entity: str  # entity.id
    relation_type: str
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    evidence_memory_ids: Optional[str] = None  # JSON string ["memory_id1", ...]
    confidence: float = 1.0


@dataclass
class Relation:
    """完整的关系模型"""
    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    valid_from: Optional[datetime]
    valid_until: Optional[datetime]
    evidence_memory_ids: Optional[str]
    confidence: float
    created_at: datetime


@dataclass
class RawContentCreate:
    """用于创建原始内容的输入模型"""
    user_id: str
    content_type: str  # conversation | document | image_description | audio_transcript
    raw_text: Optional[str] = None
    raw_file_path: Optional[str] = None
    metadata: Optional[str] = None  # JSON string
    occurred_at: Optional[datetime] = None


@dataclass
class RawContent:
    """完整的原始内容模型"""
    id: str
    user_id: str
    content_type: str
    raw_text: Optional[str]
    raw_file_path: Optional[str]
    metadata: Optional[str]
    occurred_at: Optional[datetime]
    ingested_at: datetime
    processed: bool
    extracted_memory_ids: Optional[str]


# ============================================================================
# Vector Serialization Helpers
# ============================================================================

def serialize_embedding(embedding: List[float]) -> bytes:
    """将向量序列化为二进制 blob"""
    if embedding is None:
        return None
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(blob: bytes) -> List[float]:
    """从二进制 blob 反序列化向量"""
    if blob is None:
        return None
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


# ============================================================================
# Main Database Class
# ============================================================================

class MimirDatabase:
    """
    Mimir Memory V2 数据库管理类

    功能：
    - SQLite 连接管理（线程安全）
    - 完整 Schema 创建
    - sqlite-vec 向量扩展集成
    - 基础 CRUD 操作
    """

    SCHEMA_SQL = """
    -- 核心表1: 原子化记忆 (Atomic Memories)
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT,

        -- 向量化
        embedding BLOB,

        -- 时序信息
        valid_at DATETIME,
        valid_at_confidence REAL DEFAULT 1.0,
        temporal_tags TEXT,

        -- 来源追踪
        source_type TEXT,
        source_id TEXT,
        source_metadata TEXT,

        -- 状态管理
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        access_count INTEGER DEFAULT 0,
        last_accessed DATETIME,

        -- 冲突解决
        version INTEGER DEFAULT 1,
        superseded_by TEXT,

        -- FTS5 虚拟表关联
        fts_docid INTEGER
    );

    -- 全文检索虚拟表
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        content,
        content_rowid=rowid,
        content=memories
    );

    -- 核心表2: 实体 (Entities)
    CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT,
        aliases TEXT,
        first_seen DATETIME,
        last_seen DATETIME,
        mention_count INTEGER DEFAULT 0
    );

    -- 核心表3: 关系 (Relations) - 时序知识图谱
    CREATE TABLE IF NOT EXISTS relations (
        id TEXT PRIMARY KEY,
        source_entity TEXT NOT NULL,
        target_entity TEXT NOT NULL,
        relation_type TEXT NOT NULL,

        -- 时序属性
        valid_from DATETIME,
        valid_until DATETIME,

        -- 证据
        evidence_memory_ids TEXT,
        confidence REAL DEFAULT 1.0,

        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (source_entity) REFERENCES entities(id),
        FOREIGN KEY (target_entity) REFERENCES entities(id)
    );

    -- 核心表4: 会话/文档原始内容 (Raw Content)
    CREATE TABLE IF NOT EXISTS raw_contents (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        content_type TEXT,

        -- 原始内容 (可能很大，可选存文件系统)
        raw_text TEXT,
        raw_file_path TEXT,

        -- 元数据
        metadata TEXT,

        -- 时序 (关键!)
        occurred_at DATETIME,
        ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        -- 处理状态
        processed BOOLEAN DEFAULT FALSE,
        extracted_memory_ids TEXT
    );

    -- 核心表5: 实体-记忆关联 (Many-to-Many)
    CREATE TABLE IF NOT EXISTS entity_memories (
        entity_id TEXT,
        memory_id TEXT,
        mention_count INTEGER DEFAULT 1,
        PRIMARY KEY (entity_id, memory_id),
        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    );

    -- 索引优化
    CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
    CREATE INDEX IF NOT EXISTS idx_memories_valid_at ON memories(valid_at);
    CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_type, source_id);
    CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity);
    CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity);
    CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
    CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
    CREATE INDEX IF NOT EXISTS idx_raw_contents_user ON raw_contents(user_id);
    CREATE INDEX IF NOT EXISTS idx_raw_contents_occurred ON raw_contents(occurred_at);

    -- FTS5 同步触发器
    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
    END;

    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    END;

    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
        INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
    END;
    """

    def __init__(self, db_path: str = "mimir_v2.db"):
        """
        初始化数据库连接

        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = db_path
        self._local = threading.local()
        self._vec_available = False  # 初始化向量可用标志
        self._init_connection()

    def _init_connection(self):
        """初始化数据库连接（线程本地）"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            if USE_APSW:
                # apsw 连接方式
                self._local.conn = sqlite3.Connection(self.db_path)
                # 启用扩展加载
                self._local.conn.enableloadextension(True)
                # apsw 使用 rowtrace 代替 row_factory
                self._local.conn.setrowtrace(self._apsw_row_factory)
            else:
                # 标准 sqlite3 连接方式
                self._local.conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
                )
                self._local.conn.row_factory = sqlite3.Row

            # 启用外键约束
            self._local.conn.execute("PRAGMA foreign_keys = ON")
            # 尝试加载 sqlite-vec 扩展
            try:
                load_sqlite_vec(self._local.conn)
            except Exception as e:
                print(f"Warning: Could not load sqlite-vec: {e}")

    def _apsw_row_factory(self, cursor, row):
        """apsw 行工厂：将行转换为字典"""
        return {
            col[0]: row[i]
            for i, col in enumerate(cursor.getdescription())
        }

    @property
    def conn(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接"""
        self._init_connection()
        return self._local.conn

    def init_schema(self):
        """
        初始化数据库 Schema
        创建所有表、索引和触发器
        """
        if USE_APSW:
            self.conn.cursor().execute(self.SCHEMA_SQL)
        else:
            self.conn.executescript(self.SCHEMA_SQL)

        # 尝试创建 sqlite-vec 向量虚拟表
        self._init_vector_table()

    def _init_vector_table(self):
        """初始化向量虚拟表（如果 sqlite-vec 可用）"""
        try:
            # 检查表是否已存在
            cursor = self._execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_memories'"
            )
            if cursor.fetchone() is None:
                # 创建向量虚拟表，使用 1536 维（匹配 Titan embedding）
                self._execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                        memory_id TEXT PRIMARY KEY,
                        embedding FLOAT[1536]
                    )
                """)
                self._commit()
                self._vec_available = True
        except Exception as e:
            # sqlite-vec 不可用，使用纯 Python 向量搜索
            print(f"Warning: sqlite-vec not available, using Python fallback: {e}")
            self._vec_available = False

    def _commit(self):
        """提交事务（兼容 apsw 和 sqlite3）"""
        if not USE_APSW:
            self.conn.commit()

    def _execute(self, sql, parameters=None):
        """执行 SQL（兼容 apsw 和 sqlite3）"""
        # 适配 datetime 参数
        if parameters:
            parameters = adapt_params(parameters)

        if USE_APSW:
            cursor = self.conn.cursor()
            if parameters:
                return cursor.execute(sql, parameters)
            return cursor.execute(sql)
        else:
            if parameters:
                return self.conn.execute(sql, parameters)
            return self.conn.execute(sql)

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ========================================================================
    # Memory CRUD Operations
    # ========================================================================

    def _vec_insert(self, memory_id: str, embedding: List[float]):
        """插入向量到 vec_memories 表"""
        # 添加调试断言
        if embedding and len(embedding) != 1536:
            import traceback
            import sys

            print(f"\n{'='*80}")
            print(f"ERROR: 向量维度错误！")
            print(f"  期望: 1536 维")
            print(f"  实际: {len(embedding)} 维")
            print(f"  memory_id: {memory_id}")
            print(f"\n调用栈:")
            traceback.print_stack(file=sys.stdout)
            print(f"{'='*80}\n")

            # 可选：抛出异常停止执行
            raise ValueError(f"Embedding dimension mismatch: expected 1536, got {len(embedding)}")

        # 插入向量表（如果 sqlite-vec 可用）
        if self._vec_available and embedding:
            try:
                self._execute("""
                    INSERT INTO vec_memories (memory_id, embedding) VALUES (?, ?)
                """, (memory_id, serialize_embedding(embedding)))
            except Exception as e:
                print(f"Warning: Could not insert into vec_memories: {e}")

    def create_memory(self, memory: MemoryCreate) -> str:
        """
        创建新的记忆

        Args:
            memory: MemoryCreate 对象

        Returns:
            新创建的记忆 ID
        """
        memory_id = str(uuid.uuid4())
        now = datetime.now()

        # 在插入向量前检查
        if memory.embedding:
            if len(memory.embedding) != 1536:
                import traceback
                import sys

                print(f"\n{'='*80}")
                print(f"ERROR: create_memory 收到错误维度向量！")
                print(f"  期望: 1536 维")
                print(f"  实际: {len(memory.embedding)} 维")
                content_preview = memory.content[:100] if memory.content else "<empty>"
                print(f"  内容: {content_preview}...")
                print(f"\n调用栈:")
                traceback.print_stack(file=sys.stdout)
                print(f"{'='*80}\n")

                # 可选：截断或拒绝
                if len(memory.embedding) == 6144:
                    # 如果收到 6144，可能是 4 个 1536 被拼接了
                    # 尝试只取前 1536 个作为临时修复
                    print("[WARN] 收到 6144 维向量，尝试截断为 1536 维")
                    memory.embedding = memory.embedding[:1536]

        # 序列化 embedding
        embedding_blob = serialize_embedding(memory.embedding)

        self._execute("""
            INSERT INTO memories (
                id, user_id, content, content_hash, embedding,
                valid_at, valid_at_confidence, temporal_tags,
                source_type, source_id, source_metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, memory.user_id, memory.content, memory.content_hash,
            embedding_blob, memory.valid_at, memory.valid_at_confidence,
            memory.temporal_tags, memory.source_type, memory.source_id,
            memory.source_metadata, now, now
        ))

        # 插入向量表（通过 _vec_insert）
        if self._vec_available and memory.embedding:
            self._vec_insert(memory_id, memory.embedding)

        self._commit()
        return memory_id

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        根据 ID 获取记忆

        Args:
            memory_id: 记忆 ID

        Returns:
            Memory 对象或 None
        """
        cursor = self._execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        # 更新访问统计
        self._execute("""
            UPDATE memories
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
        """, (datetime.now(), memory_id))
        self._commit()

        return self._row_to_memory(row)

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新记忆

        Args:
            memory_id: 记忆 ID
            updates: 要更新的字段字典

        Returns:
            是否成功更新
        """
        allowed_fields = {
            'content', 'content_hash', 'embedding', 'valid_at',
            'valid_at_confidence', 'temporal_tags', 'source_type',
            'source_id', 'source_metadata', 'version', 'superseded_by'
        }

        # 过滤允许的字段
        update_fields = {k: v for k, v in updates.items() if k in allowed_fields}

        if not update_fields:
            return False

        # 处理 embedding 序列化
        if 'embedding' in update_fields and update_fields['embedding'] is not None:
            update_fields['embedding'] = serialize_embedding(update_fields['embedding'])

        # 添加 updated_at
        update_fields['updated_at'] = datetime.now()

        # 构建更新语句
        set_clause = ', '.join(f"{k} = ?" for k in update_fields.keys())
        values = list(update_fields.values()) + [memory_id]

        cursor = self._execute(
            f"UPDATE memories SET {set_clause} WHERE id = ?",
            values
        )

        # 更新向量表（如果 sqlite-vec 可用）
        if self._vec_available and 'embedding' in updates:
            try:
                self._execute(
                    "DELETE FROM vec_memories WHERE memory_id = ?",
                    (memory_id,)
                )
                if updates['embedding']:
                    self._execute(
                        "INSERT INTO vec_memories (memory_id, embedding) VALUES (?, ?)",
                        (memory_id, serialize_embedding(updates['embedding']))
                    )
            except Exception as e:
                print(f"Warning: Could not update vec_memories: {e}")

        self._commit()
        # apsw 不支持 rowcount，使用 changes()
        if USE_APSW:
            return self.conn.changes() > 0
        return cursor.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆 ID

        Returns:
            是否成功删除
        """
        # 删除向量表中的记录（如果 sqlite-vec 可用）
        if self._vec_available:
            try:
                self._execute(
                    "DELETE FROM vec_memories WHERE memory_id = ?",
                    (memory_id,)
                )
            except Exception as e:
                print(f"Warning: Could not delete from vec_memories: {e}")

        # 删除实体关联
        self._execute(
            "DELETE FROM entity_memories WHERE memory_id = ?",
            (memory_id,)
        )

        # 删除记忆
        cursor = self._execute(
            "DELETE FROM memories WHERE id = ?",
            (memory_id,)
        )

        self._commit()
        # apsw 不支持 rowcount，使用 changes()
        if USE_APSW:
            return self.conn.changes() > 0
        return cursor.rowcount > 0

    def list_memories(self, user_id: Optional[str] = None,
                      limit: int = 100, offset: int = 0) -> List[Memory]:
        """
        列出记忆

        Args:
            user_id: 可选的用户 ID 过滤
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            Memory 对象列表
        """
        if user_id:
            cursor = self._execute(
                """SELECT * FROM memories
                   WHERE user_id = ?
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (user_id, limit, offset)
            )
        else:
            cursor = self._execute(
                """SELECT * FROM memories
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset)
            )

        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def _row_to_memory(self, row: Dict) -> Memory:
        """将数据库行转换为 Memory 对象"""
        return Memory(
            id=row['id'],
            user_id=row['user_id'],
            content=row['content'],
            content_hash=row['content_hash'],
            embedding=row['embedding'],
            valid_at=convert_datetime(row['valid_at']),
            valid_at_confidence=row['valid_at_confidence'],
            temporal_tags=row['temporal_tags'],
            source_type=row['source_type'],
            source_id=row['source_id'],
            source_metadata=row['source_metadata'],
            created_at=convert_datetime(row['created_at']),
            updated_at=convert_datetime(row['updated_at']),
            access_count=row['access_count'],
            last_accessed=convert_datetime(row['last_accessed']),
            version=row['version'],
            superseded_by=row['superseded_by'],
            fts_docid=row['fts_docid']
        )

    # ========================================================================
    # Vector Search Operations
    # ========================================================================

    def vector_search(self, embedding: List[float], top_k: int = 10,
                      user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        向量相似度搜索

        Args:
            embedding: 查询向量
            top_k: 返回结果数量
            user_id: 可选的用户 ID 过滤

        Returns:
            包含 memory 和 distance 的字典列表
        """
        if self._vec_available:
            return self._vector_search_sqlite_vec(embedding, top_k, user_id)
        else:
            return self._vector_search_python(embedding, top_k, user_id)

    def _vector_search_sqlite_vec(self, embedding: List[float], top_k: int = 10,
                                   user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """使用 sqlite-vec 进行向量搜索"""
        embedding_blob = serialize_embedding(embedding)

        if user_id:
            cursor = self._execute("""
                SELECT vm.memory_id, vec_distance_L2(vm.embedding, ?) as distance
                FROM vec_memories vm
                JOIN memories m ON vm.memory_id = m.id
                WHERE m.user_id = ?
                ORDER BY distance
                LIMIT ?
            """, (embedding_blob, user_id, top_k))
        else:
            cursor = self._execute("""
                SELECT memory_id, vec_distance_L2(embedding, ?) as distance
                FROM vec_memories
                ORDER BY distance
                LIMIT ?
            """, (embedding_blob, top_k))

        results = []
        for row in cursor.fetchall():
            memory = self.get_memory(row['memory_id'])
            if memory:
                results.append({
                    'memory': memory,
                    'distance': row['distance'],
                    'memory_id': row['memory_id']
                })

        return results

    def _vector_search_python(self, embedding: List[float], top_k: int = 10,
                               user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """使用纯 Python 计算余弦相似度（sqlite-vec 不可用时 fallback）"""
        import numpy as np

        query_vec = np.array(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        # 获取候选记忆
        if user_id:
            memories = self.list_memories(user_id=user_id, limit=1000)
        else:
            memories = self.list_memories(limit=1000)

        # 计算相似度
        scored = []
        for memory in memories:
            if memory.embedding is None:
                continue

            mem_vec = deserialize_embedding(memory.embedding)
            if mem_vec is None or len(mem_vec) != len(embedding):
                continue

            mem_vec = np.array(mem_vec, dtype=np.float32)
            mem_norm = np.linalg.norm(mem_vec)

            if mem_norm == 0 or query_norm == 0:
                continue

            # 余弦相似度 (1 - cosine_distance)
            cosine_sim = np.dot(query_vec, mem_vec) / (query_norm * mem_norm)
            distance = 1 - cosine_sim  # 转换为距离

            scored.append({
                'memory': memory,
                'distance': float(distance),
                'memory_id': memory.id
            })

        # 排序并返回 top_k
        scored.sort(key=lambda x: x['distance'])
        return scored[:top_k]

    def fts_search(self, query: str, top_k: int = 10,
                   user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        全文检索

        Args:
            query: 搜索查询
            top_k: 返回结果数量
            user_id: 可选的用户 ID 过滤

        Returns:
            包含 memory 和 rank 的字典列表
        """
        if user_id:
            cursor = self._execute("""
                SELECT m.*, fts.rank
                FROM memories_fts fts
                JOIN memories m ON fts.rowid = m.rowid
                WHERE memories_fts MATCH ? AND m.user_id = ?
                ORDER BY fts.rank
                LIMIT ?
            """, (query, user_id, top_k))
        else:
            cursor = self._execute("""
                SELECT m.*, fts.rank
                FROM memories_fts fts
                JOIN memories m ON fts.rowid = m.rowid
                WHERE memories_fts MATCH ?
                ORDER BY fts.rank
                LIMIT ?
            """, (query, top_k))

        results = []
        for row in cursor.fetchall():
            memory = self._row_to_memory(row)
            results.append({
                'memory': memory,
                'rank': row['rank'],
                'memory_id': memory.id
            })

            # 更新访问统计
            self._execute("""
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """, (datetime.now(), memory.id))

        self._commit()
        return results

    # ========================================================================
    # Entity CRUD Operations
    # ========================================================================

    def create_entity(self, entity: EntityCreate) -> str:
        """
        创建新实体

        Args:
            entity: EntityCreate 对象

        Returns:
            新创建的实体 ID
        """
        entity_id = str(uuid.uuid4())
        now = datetime.now()

        first_seen = entity.first_seen or now
        last_seen = entity.last_seen or now

        self._execute("""
            INSERT INTO entities (id, name, type, aliases, first_seen, last_seen, mention_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (entity_id, entity.name, entity.type, entity.aliases,
              first_seen, last_seen))

        self._commit()
        return entity_id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """根据 ID 获取实体"""
        cursor = self._execute(
            "SELECT * FROM entities WHERE id = ?",
            (entity_id,)
        )
        row = cursor.fetchone()
        return self._row_to_entity(row) if row else None

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """根据名称获取实体"""
        cursor = self._execute(
            "SELECT * FROM entities WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()
        return self._row_to_entity(row) if row else None

    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """更新实体"""
        allowed_fields = {'name', 'type', 'aliases', 'last_seen', 'mention_count'}
        update_fields = {k: v for k, v in updates.items() if k in allowed_fields}

        if not update_fields:
            return False

        set_clause = ', '.join(f"{k} = ?" for k in update_fields.keys())
        values = list(update_fields.values()) + [entity_id]

        cursor = self._execute(
            f"UPDATE entities SET {set_clause} WHERE id = ?",
            values
        )
        self._commit()
        # apsw 不支持 rowcount，使用 changes()
        if USE_APSW:
            return self.conn.changes() > 0
        return cursor.rowcount > 0

    def _row_to_entity(self, row: Dict) -> Entity:
        """将数据库行转换为 Entity 对象"""
        return Entity(
            id=row['id'],
            name=row['name'],
            type=row['type'],
            aliases=row['aliases'],
            first_seen=row['first_seen'],
            last_seen=row['last_seen'],
            mention_count=row['mention_count']
        )

    # ========================================================================
    # Relation CRUD Operations
    # ========================================================================

    def create_relation(self, relation: RelationCreate) -> str:
        """
        创建新关系

        Args:
            relation: RelationCreate 对象

        Returns:
            新创建的关系 ID
        """
        relation_id = str(uuid.uuid4())

        self._execute("""
            INSERT INTO relations (
                id, source_entity, target_entity, relation_type,
                valid_from, valid_until, evidence_memory_ids, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            relation_id, relation.source_entity, relation.target_entity,
            relation.relation_type, relation.valid_from, relation.valid_until,
            relation.evidence_memory_ids, relation.confidence
        ))

        self._commit()
        return relation_id

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """根据 ID 获取关系"""
        cursor = self._execute(
            "SELECT * FROM relations WHERE id = ?",
            (relation_id,)
        )
        row = cursor.fetchone()
        return self._row_to_relation(row) if row else None

    def get_relations_by_entity(self, entity_id: str,
                                 as_source: bool = True,
                                 as_target: bool = True) -> List[Relation]:
        """
        获取实体相关的所有关系

        Args:
            entity_id: 实体 ID
            as_source: 是否包含作为 source 的关系
            as_target: 是否包含作为 target 的关系

        Returns:
            Relation 对象列表
        """
        relations = []

        if as_source:
            cursor = self._execute(
                "SELECT * FROM relations WHERE source_entity = ?",
                (entity_id,)
            )
            relations.extend([self._row_to_relation(row) for row in cursor.fetchall()])

        if as_target:
            cursor = self._execute(
                "SELECT * FROM relations WHERE target_entity = ?",
                (entity_id,)
            )
            relations.extend([self._row_to_relation(row) for row in cursor.fetchall()])

        return relations

    def _row_to_relation(self, row: Dict) -> Relation:
        """将数据库行转换为 Relation 对象"""
        return Relation(
            id=row['id'],
            source_entity=row['source_entity'],
            target_entity=row['target_entity'],
            relation_type=row['relation_type'],
            valid_from=row['valid_from'],
            valid_until=row['valid_until'],
            evidence_memory_ids=row['evidence_memory_ids'],
            confidence=row['confidence'],
            created_at=row['created_at']
        )

    # ========================================================================
    # Raw Content CRUD Operations
    # ========================================================================

    def create_raw_content(self, content: RawContentCreate) -> str:
        """
        创建新的原始内容

        Args:
            content: RawContentCreate 对象

        Returns:
            新创建的原始内容 ID
        """
        content_id = str(uuid.uuid4())

        self._execute("""
            INSERT INTO raw_contents (
                id, user_id, content_type, raw_text, raw_file_path,
                metadata, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            content_id, content.user_id, content.content_type,
            content.raw_text, content.raw_file_path, content.metadata,
            content.occurred_at
        ))

        self._commit()
        return content_id

    def get_raw_content(self, content_id: str) -> Optional[RawContent]:
        """根据 ID 获取原始内容"""
        cursor = self._execute(
            "SELECT * FROM raw_contents WHERE id = ?",
            (content_id,)
        )
        row = cursor.fetchone()
        return self._row_to_raw_content(row) if row else None

    def update_raw_content(self, content_id: str, updates: Dict[str, Any]) -> bool:
        """更新原始内容"""
        allowed_fields = {'raw_text', 'raw_file_path', 'metadata',
                         'occurred_at', 'processed', 'extracted_memory_ids'}
        update_fields = {k: v for k, v in updates.items() if k in allowed_fields}

        if not update_fields:
            return False

        set_clause = ', '.join(f"{k} = ?" for k in update_fields.keys())
        values = list(update_fields.values()) + [content_id]

        cursor = self._execute(
            f"UPDATE raw_contents SET {set_clause} WHERE id = ?",
            values
        )
        self._commit()
        # apsw 不支持 rowcount，使用 changes()
        if USE_APSW:
            return self.conn.changes() > 0
        return cursor.rowcount > 0

    def _row_to_raw_content(self, row: Dict) -> RawContent:
        """将数据库行转换为 RawContent 对象"""
        return RawContent(
            id=row['id'],
            user_id=row['user_id'],
            content_type=row['content_type'],
            raw_text=row['raw_text'],
            raw_file_path=row['raw_file_path'],
            metadata=row['metadata'],
            occurred_at=row['occurred_at'],
            ingested_at=row['ingested_at'],
            processed=row['processed'],
            extracted_memory_ids=row['extracted_memory_ids']
        )

    # ========================================================================
    # Entity-Memory Association Operations
    # ========================================================================

    def associate_entity_memory(self, entity_id: str, memory_id: str) -> bool:
        """
        关联实体和记忆

        Args:
            entity_id: 实体 ID
            memory_id: 记忆 ID

        Returns:
            是否成功创建关联
        """
        try:
            self._execute("""
                INSERT INTO entity_memories (entity_id, memory_id, mention_count)
                VALUES (?, ?, 1)
                ON CONFLICT(entity_id, memory_id) DO UPDATE SET
                    mention_count = mention_count + 1
            """, (entity_id, memory_id))
            self._commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_entity_memories(self, entity_id: str) -> List[Memory]:
        """
        获取实体关联的所有记忆

        Args:
            entity_id: 实体 ID

        Returns:
            Memory 对象列表
        """
        cursor = self._execute("""
            SELECT m.* FROM memories m
            JOIN entity_memories em ON m.id = em.memory_id
            WHERE em.entity_id = ?
            ORDER BY em.mention_count DESC, m.created_at DESC
        """, (entity_id,))

        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def get_memory_entities(self, memory_id: str) -> List[Entity]:
        """
        获取记忆关联的所有实体

        Args:
            memory_id: 记忆 ID

        Returns:
            Entity 对象列表
        """
        cursor = self._execute("""
            SELECT e.* FROM entities e
            JOIN entity_memories em ON e.id = em.entity_id
            WHERE em.memory_id = ?
            ORDER BY em.mention_count DESC
        """, (memory_id,))

        return [self._row_to_entity(row) for row in cursor.fetchall()]

    # ========================================================================
    # Statistics & Maintenance
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats = {}

        def get_count_result(cursor):
            """兼容 apsw（字典）和 sqlite3（元组）的结果获取"""
            row = cursor.fetchone()
            if isinstance(row, dict):
                return row['COUNT(*)']
            return row[0]

        # 记忆数量
        cursor = self._execute("SELECT COUNT(*) FROM memories")
        stats['memory_count'] = get_count_result(cursor)

        # 实体数量
        cursor = self._execute("SELECT COUNT(*) FROM entities")
        stats['entity_count'] = get_count_result(cursor)

        # 关系数量
        cursor = self._execute("SELECT COUNT(*) FROM relations")
        stats['relation_count'] = get_count_result(cursor)

        # 原始内容数量
        cursor = self._execute("SELECT COUNT(*) FROM raw_contents")
        stats['raw_content_count'] = get_count_result(cursor)

        # 向量数量
        cursor = self._execute("SELECT COUNT(*) FROM vec_memories")
        stats['vector_count'] = get_count_result(cursor)

        return stats

    def vacuum(self):
        """优化数据库（VACUUM）"""
        self._execute("VACUUM")

    def analyze(self):
        """更新统计信息（ANALYZE）"""
        self._execute("ANALYZE")


# ============================================================================
# Database Initialization Script
# ============================================================================

def init_database(db_path: str = "mimir_v2.db") -> MimirDatabase:
    """
    初始化数据库

    Args:
        db_path: 数据库文件路径

    Returns:
        初始化后的 MimirDatabase 实例
    """
    db = MimirDatabase(db_path)
    db.init_schema()
    print(f"Database initialized at: {db_path}")
    print(f"Schema version: v2.0")
    return db


if __name__ == "__main__":
    # 作为脚本运行时初始化数据库
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "mimir_v2.db"
    init_database(db_path)
