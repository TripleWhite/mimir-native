"""
Mimir Memory V2 - SQLite + sqlite-vec implementation

数据库层，包含完整的 Schema 和 CRUD 操作。
Memory Agent 层，负责事实提取、去重、冲突解决。
Temporal KG 层，负责时序知识图谱查询和推理。

Main Entry Point:
    from app.mimir_v2 import MimirMemory
    
    mimir = MimirMemory(db_path="mimir.db")
    memories = mimir.add_content(content="...", content_type="conversation")
"""

import logging
import os
from typing import List, Optional, Dict, Any

from mimir_native.database import (
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
from mimir_native.models import (
    Fact,
    ConflictResolutionResult,
    DeduplicationResult,
)
from mimir_native.llm_client import (
    BedrockClient,
    BedrockConfig,
    create_llm_client,
)
from mimir_native.memory_agent import (
    MemoryAgent,
)
from mimir_native.temporal_kg import (
    TemporalKnowledgeGraph,
)
from mimir_native.retrieval.hybrid_retriever import HybridRetriever
from mimir_native.preprocessors import MultimodalPreprocessor

# Import evaluation module
from . import evaluation
from .evaluation import LoCoMoEvaluator

__version__ = '2.0.0'

__all__ = [
    # Main Class
    'MimirMemory',
    # Database
    'MimirDatabase',
    'MemoryCreate',
    'Memory',
    'EntityCreate',
    'Entity',
    'RelationCreate',
    'Relation',
    'RawContentCreate',
    'RawContent',
    'init_database',
    'serialize_embedding',
    'deserialize_embedding',
    # Models
    'Fact',
    'ConflictResolutionResult',
    'DeduplicationResult',
    # LLM Client
    'BedrockClient',
    'BedrockConfig',
    'create_llm_client',
    # Memory Agent
    'MemoryAgent',
    # Temporal KG
    'TemporalKnowledgeGraph',
    # Preprocessors
    'MultimodalPreprocessor',
    # Retrieval
    'HybridRetriever',
    # Evaluation
    'evaluation',
    'LoCoMoEvaluator',
]


class MimirMemory:
    """
    Mimir Memory V2 统一入口
    
    整合所有组件提供统一接口：
    - 数据库 (MimirDatabase)
    - 预处理器 (MultimodalPreprocessor)
    - 记忆智能体 (MemoryAgent)
    - 时序知识图谱 (TemporalKnowledgeGraph)
    - 混合检索器 (HybridRetriever)
    """
    
    def __init__(
        self,
        db_path: str = "mimir.db",
        llm_client=None,
        auto_init: bool = True,
        log_level: str = "INFO",
        enable_audio: bool = False
    ):
        """
        初始化 Mimir Memory V2
        
        Args:
            db_path: SQLite 数据库路径
            llm_client: LLM 客户端（可选，默认创建 BedrockClient）
            auto_init: 是否自动初始化数据库
            log_level: 日志级别
            enable_audio: 是否启用音频处理（需要 OpenAI API Key，默认关闭）
        """
        # 设置日志级别
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Initializing Mimir Memory V2 (db: {db_path})")
        
        # 初始化各组件
        self.db = MimirDatabase(db_path)
        self.db.init_schema()  # 初始化数据库 schema
        self.llm = llm_client or create_llm_client()
        
        # 初始化预处理器（根据 enable_audio 决定是否启用音频）
        if enable_audio:
            self.preprocessor = MultimodalPreprocessor()
        else:
            # 手动初始化，跳过音频处理器
            from mimir_native.preprocessors import (
                DocumentProcessor, ImageProcessor, ConversationProcessor,
                BedrockClient as PreprocBedrockClient, BedrockConfig
            )
            try:
                config = BedrockConfig(region=os.getenv('AWS_REGION', 'us-east-1'))
                bedrock_client = PreprocBedrockClient(config)
            except Exception as e:
                logger.warning(f"Bedrock client for preprocessor init failed: {e}")
                bedrock_client = None
            
            self.preprocessor = MultimodalPreprocessor.__new__(MultimodalPreprocessor)
            self.preprocessor.processors = {
                'document': DocumentProcessor(),
                'image': ImageProcessor(bedrock_client=bedrock_client),
                'conversation': ConversationProcessor(),
            }
            self.preprocessor._type_aliases = {
                'pdf': 'document', 'docx': 'document', 'txt': 'document', 'text': 'document',
                'img': 'image', 'picture': 'image', 'photo': 'image',
                'chat': 'conversation', 'dialogue': 'conversation', 'message': 'conversation',
            }
            self.preprocessor._bedrock_client = bedrock_client
        
        self.memory_agent = MemoryAgent(self.db, self.llm)
        self.kg = TemporalKnowledgeGraph(self.db)
        
        # 初始化检索器（需要数据库和图谱）
        self.retriever = HybridRetriever(
            db=self.db,
            kg=self.kg,
            llm_client=self.llm
        )
        
        if auto_init:
            self._init_default_user()
        
        logger.info("Mimir Memory V2 initialized successfully")
    
    def _init_default_user(self):
        """初始化默认用户"""
        # 创建默认用户（如果用户系统可用）
        pass
    
    def add_content(
        self,
        content,
        content_type: str,
        metadata: dict = None,
        user_id: str = "default"
    ) -> List[Memory]:
        """
        添加内容到记忆系统
        
        完整流程：
        1. 预处理（提取文本、日期）
        2. 提取事实
        3. 存储到数据库
        4. 更新知识图谱
        
        Args:
            content: 输入内容（文本、文件路径、对话等）
            content_type: 内容类型（text, document, image, audio, conversation）
            metadata: 可选元数据字典
            user_id: 用户 ID
        
        Returns:
            List[Memory]: 创建/更新的记忆列表
        """
        if metadata is None:
            metadata = {}
        
        metadata['user_id'] = user_id
        
        logger = logging.getLogger(__name__)
        logger.debug(f"Adding content of type: {content_type}")
        
        # 1. 预处理
        raw_content = self.preprocessor.process(
            content, content_type, metadata
        )
        
        # 2. 提取事实并存储
        memories = self.memory_agent.process_raw_content(raw_content)
        
        # 3. 更新图谱
        for memory in memories:
            entities, relations = self.kg.extract_entities_and_relations(
                memory.content, self.llm
            )
            self.kg.add_fact(memory, entities, relations)
        
        logger.info(f"Added {len(memories)} memories from {content_type} content")
        return memories
    
    def query(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 10,
        filters: Dict = None
    ) -> List[Any]:
        """
        查询记忆
        
        Args:
            query: 查询文本
            user_id: 用户 ID
            top_k: 返回结果数
            filters: 可选过滤条件
        
        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        return self.retriever.retrieve(
            query=query,
            user_id=user_id,
            top_k=top_k,
            filters=filters
        )
    
    def query_with_explanation(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        带解释的查询
        
        Args:
            query: 查询文本
            user_id: 用户 ID
            top_k: 返回结果数
        
        Returns:
            包含结果和解释的字典
        """
        return self.retriever.retrieve_with_explanation(
            query=query,
            user_id=user_id,
            top_k=top_k
        )
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """获取单个记忆"""
        return self.db.get_memory(memory_id)
    
    def update_memory(self, memory_id: str, updates: Dict) -> bool:
        """更新记忆"""
        return self.db.update_memory(memory_id, updates)
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        return self.db.delete_memory(memory_id)
    
    def get_user_memories(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Memory]:
        """获取用户的所有记忆"""
        return self.db.list_memories(user_id=user_id, limit=limit, offset=offset)
    
    def get_stats(self, user_id: str = "default") -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            'memory_count': self.db.count_memories(user_id=user_id),
            'entity_count': self.kg.graph.number_of_nodes() if self.kg.graph else 0,
            'relation_count': self.kg.graph.number_of_edges() if self.kg.graph else 0,
        }
    
    def build_kg_from_memories(self, user_id: str = "default"):
        """从数据库重建知识图谱"""
        import logging
        logger = logging.getLogger(__name__)
        self.kg.build_from_memories(user_id)
        logger.info(f"Knowledge graph rebuilt for user: {user_id}")
    
    def close(self):
        """关闭所有资源"""
        self.db.close()
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Mimir Memory V2 closed")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
        return False
