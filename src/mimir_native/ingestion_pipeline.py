"""
Mimir Ingestion Pipeline V3

核心改进：
1. 使用 ContentProcessor 统一处理所有内容
2. 确保每条消息绑定正确的 session_date
3. 强制时序标准化在存储前完成
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class MimirIngestionPipeline:
    """
    Mimir 数据摄入管道 V3
    
    处理流程：
    输入 (多模态内容) 
        → 模态识别 
        → ContentProcessor 提取 
        → 时序标准化 
        → 向量化 
        → 存储
    """
    
    def __init__(self, mimir_db, llm_client, embedder):
        self.db = mimir_db
        self.llm = llm_client
        self.embedder = embedder
        
        # 初始化 ContentProcessor
        from mimir_native.content_processor import ContentProcessor
        self.processor = ContentProcessor(llm_client)
    
    def ingest_conversation(
        self,
        messages: List[Dict],
        session_date: str,
        source_type: str = 'conversation',
        user_id: str = 'default'
    ) -> Dict[str, Any]:
        """
        摄入对话内容
        
        Args:
            messages: [{speaker, text, timestamp}, ...]
            session_date: 对话发生的日期（如 "8 May 2023"）
            source_type: 来源类型
            user_id: 用户ID
        
        Returns:
            {'ingested': 10, 'memories_created': 15}
        """
        logger.info(f"摄入对话: {len(messages)} 条消息, session_date={session_date}")
        
        # 1. 使用 ContentProcessor 处理
        processed_memories = self.processor.process_conversation(
            messages=messages,
            session_date=session_date,
            source_type=source_type
        )
        
        # 2. 存储到数据库
        memories_created = 0
        for memory in processed_memories:
            try:
                self._store_memory(memory, user_id)
                memories_created += 1
            except Exception as e:
                logger.warning(f"存储记忆失败: {e}")
        
        logger.info(f"摄入完成: {len(messages)} 条消息 → {memories_created} 条记忆")
        
        return {
            'ingested': len(messages),
            'memories_created': memories_created
        }
    
    def _store_memory(self, memory: Dict, user_id: str):
        """存储单个记忆"""
        from mimir_native.database import MemoryCreate
        
        content = memory['content']
        content_hash = hashlib.md5(content.lower().strip().encode()).hexdigest()
        
        # 获取 embedding
        embedding = self.embedder.embed(content)
        
        # 构建 metadata
        metadata = {
            'source_type': memory['source_type'],
            'source_id': memory.get('source_id', ''),
            'speaker': memory.get('speaker', ''),
            'session_date': memory.get('session_date', ''),
            'entities': memory.get('entities', []),
            'topics': memory.get('topics', []),
            'content_type': memory.get('content_type', 'fact'),
            'temporal_info': memory.get('temporal_info', {}),
            'created_at': memory.get('created_at', datetime.now().isoformat())
        }
        
        memory_create = MemoryCreate(
            user_id=user_id,
            content=content,
            content_hash=content_hash,
            embedding=embedding,
            source_type=memory['source_type'],
            source_metadata=json.dumps(metadata)
        )
        
        memory_id = self.db.create_memory(memory_create)
        return memory_id
    
    def ingest_locomo_conversation(
        self,
        conversation: Dict,
        user_id: str = 'default'
    ) -> Dict[str, Any]:
        """
        专门处理 LoCoMo 格式的对话
        
        Args:
            conversation: LoCoMo 格式的对话数据
            user_id: 用户ID
        
        Returns:
            摄入统计
        """
        total_messages = 0
        total_memories = 0
        
        # 遍历所有 session
        for key, value in conversation.items():
            # 跳过日期字段
            if '_date_time' in key:
                continue
            
            # 只处理消息列表
            if not isinstance(value, list):
                continue
            
            # 获取对应的 session_date
            date_key = f"{key}_date_time"
            session_date = conversation.get(date_key)
            
            if not session_date:
                logger.warning(f"Session {key} 没有日期信息，跳过")
                continue
            
            # 转换消息格式
            messages = []
            for msg in value:
                if isinstance(msg, dict) and 'text' in msg:
                    messages.append({
                        'speaker': msg.get('speaker', ''),
                        'text': msg.get('text', ''),
                        'timestamp': msg.get('timestamp', '')
                    })
            
            if messages:
                result = self.ingest_conversation(
                    messages=messages,
                    session_date=session_date,
                    source_type='locomo_conversation',
                    user_id=user_id
                )
                total_messages += result['ingested']
                total_memories += result['memories_created']
        
        return {
            'total_messages': total_messages,
            'total_memories': total_memories
        }


# 使用示例
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from mimir_native import MimirMemory
    from mimir_native.llm_client import BedrockClient
    
    # 初始化
    mimir = MimirMemory(db_path=':memory:')
    llm = BedrockClient()
    
    pipeline = MimirIngestionPipeline(
        mimir_db=mimir.memory_agent.db,
        llm_client=llm,
        embedder=llm
    )
    
    # 测试数据
    messages = [
        {'speaker': 'Caroline', 'text': 'I visited the LGBTQ support group yesterday.'},
        {'speaker': 'Friend', 'text': 'How was it?'},
        {'speaker': 'Caroline', 'text': 'It was great. I am a transgender woman.'},
    ]
    
    result = pipeline.ingest_conversation(
        messages=messages,
        session_date='8 May 2023',
        user_id='test'
    )
    
    print(f"摄入完成: {result}")
    
    # 验证存储的记忆
    cursor = mimir.memory_agent.db._execute(
        "SELECT content FROM memories WHERE user_id = ?",
        ('test',)
    )
    rows = cursor.fetchall()
    
    print("\n存储的记忆:")
    for row in rows:
        print(f"  - {row['content'][:80]}...")
