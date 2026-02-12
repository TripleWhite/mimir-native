"""Memory Entry - Data structures for memory storage"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np
import hashlib


class ContentType(Enum):
    """Supported content types"""
    TEXT = "text"
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    CODE = "code"
    IMAGE_DESCRIPTION = "image_description"
    AUDIO_TRANSCRIPT = "audio_transcript"
    VIDEO_SUMMARY = "video_summary"
    STRUCTURED_DATA = "structured_data"


class MemoryTier(Enum):
    """Memory storage tiers"""
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass
class MemoryMetadata:
    """Metadata for memory entries"""
    timestamp: datetime
    source: str
    content_type: ContentType
    tier: MemoryTier = MemoryTier.SHORT_TERM
    tags: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'content_type': self.content_type.value,
            'tier': self.tier.value,
            'tags': self.tags,
            'relations': self.relations,
            'session_id': self.session_id,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryMetadata':
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            content_type=ContentType(data['content_type']),
            tier=MemoryTier(data['tier']),
            tags=data.get('tags', []),
            relations=data.get('relations', []),
            session_id=data.get('session_id'),
            importance=data.get('importance', 0.5),
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None
        )


@dataclass 
class MemoryEntry:
    """A single memory entry"""
    id: str
    content: str
    content_type: ContentType
    metadata: MemoryMetadata
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'content_type': self.content_type.value,
            'metadata': self.metadata.to_dict(),
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        embedding = np.array(data['embedding']) if data.get('embedding') else None
        return cls(
            id=data['id'],
            content=data['content'],
            content_type=ContentType(data['content_type']),
            metadata=MemoryMetadata.from_dict(data['metadata']),
            embedding=embedding
        )
    
    def update_access(self):
        """Update access statistics"""
        self.metadata.access_count += 1
        self.metadata.last_accessed = datetime.now()
