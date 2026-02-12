"""
Core Memory Storage - Unified interface for hierarchical memory management
"""

from enum import Enum, auto
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
import numpy as np
from pathlib import Path


class MemoryTier(Enum):
    """Memory storage tiers"""
    WORKING = "working"      # Active context (current session)
    SHORT_TERM = "short_term"  # Recent history (last few sessions)
    LONG_TERM = "long_term"    # Persistent storage (all history)


class ContentType(Enum):
    """Supported content types"""
    TEXT = "text"
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    CODE = "code"
    IMAGE_DESCRIPTION = "image_description"
    AUDIO_TRANSCRIPT = "audio_transcript"
    VIDEO_SUMMARY = "video_summary"


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
    importance: float = 0.5  # 0-1, for retention priority
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
    """
    A single memory entry.
    
    Supports multiple content types and rich metadata.
    """
    id: str
    content: str
    content_type: ContentType
    metadata: MemoryMetadata
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from content hash
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


class UnifiedMemoryStore:
    """
    Unified memory storage with hierarchical tiers.
    
    Features:
    - Multi-tier storage (working, short-term, long-term)
    - Multi-content-type support
    - Automatic importance-based retention
    - Embedding support for semantic search
    - Persistence to disk
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_working: int = 50,
        max_short_term: int = 500,
        max_long_term: int = 10000
    ):
        """
        Initialize memory store.
        
        Args:
            storage_dir: Directory for persistent storage
            max_working: Max entries in working memory
            max_short_term: Max entries in short-term memory
            max_long_term: Max entries in long-term memory
        """
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.max_working = max_working
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        
        # Memory tiers
        self.working_memory: Dict[str, MemoryEntry] = {}
        self.short_term_memory: Dict[str, MemoryEntry] = {}
        self.long_term_memory: Dict[str, MemoryEntry] = {}
        
        # Index for fast lookup
        self.tag_index: Dict[str, List[str]] = {}
        self.session_index: Dict[str, List[str]] = {}
        
        # Load from disk if available
        if self.storage_dir:
            self._load_from_disk()
    
    def add(
        self,
        content: str,
        content_type: ContentType,
        source: str,
        tier: MemoryTier = MemoryTier.SHORT_TERM,
        tags: Optional[List[str]] = None,
        relations: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        importance: float = 0.5,
        embedding: Optional[np.ndarray] = None
    ) -> MemoryEntry:
        """
        Add a memory entry.
        
        Args:
            content: The content to store
            content_type: Type of content
            source: Source identifier
            tier: Storage tier
            tags: Optional tags for categorization
            relations: Optional related memory IDs
            session_id: Optional session identifier
            importance: Importance score (0-1)
            embedding: Optional vector embedding
            
        Returns:
            The created MemoryEntry
        """
        metadata = MemoryMetadata(
            timestamp=datetime.now(),
            source=source,
            content_type=content_type,
            tier=tier,
            tags=tags or [],
            relations=relations or [],
            session_id=session_id,
            importance=importance
        )
        
        entry = MemoryEntry(
            id="",
            content=content,
            content_type=content_type,
            metadata=metadata,
            embedding=embedding
        )
        
        # Store in appropriate tier
        if tier == MemoryTier.WORKING:
            self._add_to_working(entry)
        elif tier == MemoryTier.SHORT_TERM:
            self._add_to_short_term(entry)
        else:
            self._add_to_long_term(entry)
        
        # Update indexes
        self._update_indexes(entry)
        
        return entry
    
    def _add_to_working(self, entry: MemoryEntry):
        """Add to working memory with eviction"""
        self.working_memory[entry.id] = entry
        
        # Evict if over capacity
        if len(self.working_memory) > self.max_working:
            self._evict_working()
    
    def _add_to_short_term(self, entry: MemoryEntry):
        """Add to short-term memory with eviction"""
        self.short_term_memory[entry.id] = entry
        
        if len(self.short_term_memory) > self.max_short_term:
            self._evict_short_term()
    
    def _add_to_long_term(self, entry: MemoryEntry):
        """Add to long-term memory with eviction"""
        self.long_term_memory[entry.id] = entry
        
        if len(self.long_term_memory) > self.max_long_term:
            self._evict_long_term()
    
    def _evict_working(self):
        """Evict least important entries from working memory"""
        entries = sorted(
            self.working_memory.values(),
            key=lambda e: (e.metadata.importance, e.metadata.last_accessed or datetime.min),
            reverse=False
        )
        
        # Evict bottom 20%
        to_evict = int(len(entries) * 0.2)
        for entry in entries[:to_evict]:
            # Promote to short-term before evicting
            entry.metadata.tier = MemoryTier.SHORT_TERM
            self.short_term_memory[entry.id] = entry
            del self.working_memory[entry.id]
    
    def _evict_short_term(self):
        """Evict to long-term memory"""
        entries = sorted(
            self.short_term_memory.values(),
            key=lambda e: e.metadata.importance,
            reverse=False
        )
        
        to_evict = int(len(entries) * 0.1)
        for entry in entries[:to_evict]:
            entry.metadata.tier = MemoryTier.LONG_TERM
            self.long_term_memory[entry.id] = entry
            del self.short_term_memory[entry.id]
    
    def _evict_long_term(self):
        """Delete least important long-term entries"""
        entries = sorted(
            self.long_term_memory.values(),
            key=lambda e: (e.metadata.importance, e.metadata.access_count),
            reverse=False
        )
        
        # Delete bottom 10%
        to_evict = int(len(entries) * 0.1)
        for entry in entries[:to_evict]:
            del self.long_term_memory[entry.id]
    
    def _update_indexes(self, entry: MemoryEntry):
        """Update lookup indexes"""
        # Tag index
        for tag in entry.metadata.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            if entry.id not in self.tag_index[tag]:
                self.tag_index[tag].append(entry.id)
        
        # Session index
        if entry.metadata.session_id:
            sid = entry.metadata.session_id
            if sid not in self.session_index:
                self.session_index[sid] = []
            if entry.id not in self.session_index[sid]:
                self.session_index[sid].append(entry.id)
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific entry by ID"""
        # Check all tiers
        for tier in [self.working_memory, self.short_term_memory, self.long_term_memory]:
            if entry_id in tier:
                entry = tier[entry_id]
                entry.update_access()
                return entry
        return None
    
    def search(
        self,
        query: str = "",
        tags: Optional[List[str]] = None,
        content_type: Optional[ContentType] = None,
        session_id: Optional[str] = None,
        tier: Optional[MemoryTier] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Search memory entries.
        
        Args:
            query: Text query (searches content)
            tags: Filter by tags
            content_type: Filter by content type
            session_id: Filter by session
            tier: Filter by memory tier
            limit: Max results
            
        Returns:
            Matching entries
        """
        results = []
        
        # Determine which tiers to search
        tiers = []
        if tier == MemoryTier.WORKING:
            tiers = [self.working_memory]
        elif tier == MemoryTier.SHORT_TERM:
            tiers = [self.short_term_memory]
        elif tier == MemoryTier.LONG_TERM:
            tiers = [self.long_term_memory]
        else:
            tiers = [self.working_memory, self.short_term_memory, self.long_term_memory]
        
        # Search each tier
        for tier_dict in tiers:
            for entry in tier_dict.values():
                # Filter by content type
                if content_type and entry.content_type != content_type:
                    continue
                
                # Filter by session
                if session_id and entry.metadata.session_id != session_id:
                    continue
                
                # Filter by tags
                if tags and not any(tag in entry.metadata.tags for tag in tags):
                    continue
                
                # Text search
                if query and query.lower() not in entry.content.lower():
                    continue
                
                results.append(entry)
        
        # Sort by importance and recency
        results.sort(
            key=lambda e: (e.metadata.importance, e.metadata.timestamp),
            reverse=True
        )
        
        return results[:limit]
    
    def get_all_embeddings(self) -> Tuple[List[str], List[np.ndarray], List[Dict[str, Any]]]:
        """
        Get all entries with embeddings for semantic search.
        
        Returns:
            (contents, embeddings, metadata) tuples
        """
        contents = []
        embeddings = []
        metadata = []
        
        for tier in [self.working_memory, self.short_term_memory, self.long_term_memory]:
            for entry in tier.values():
                if entry.embedding is not None:
                    contents.append(entry.content)
                    embeddings.append(entry.embedding)
                    metadata.append({
                        'id': entry.id,
                        'type': entry.content_type.value,
                        'source': entry.metadata.source,
                        'session': entry.metadata.session_id,
                        'timestamp': entry.metadata.timestamp.isoformat()
                    })
        
        return contents, embeddings, metadata
    
    def save(self):
        """Persist to disk"""
        if not self.storage_dir:
            return
        
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each tier
        for tier_name, tier_dict in [
            ('working', self.working_memory),
            ('short_term', self.short_term_memory),
            ('long_term', self.long_term_memory)
        ]:
            data = {
                'entries': [e.to_dict() for e in tier_dict.values()]
            }
            
            file_path = self.storage_dir / f'{tier_name}_memory.json'
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def _load_from_disk(self):
        """Load from disk"""
        if not self.storage_dir:
            return
        
        for tier_name, tier_dict in [
            ('working', self.working_memory),
            ('short_term', self.short_term_memory),
            ('long_term', self.long_term_memory)
        ]:
            file_path = self.storage_dir / f'{tier_name}_memory.json'
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    for entry_data in data.get('entries', []):
                        entry = MemoryEntry.from_dict(entry_data)
                        tier_dict[entry.id] = entry
                        self._update_indexes(entry)
                except Exception as e:
                    print(f"Warning: Failed to load {tier_name} memory: {e}")
    
    def clear(self, tier: Optional[MemoryTier] = None):
        """Clear memory (optionally by tier)"""
        if tier == MemoryTier.WORKING or tier is None:
            self.working_memory.clear()
        if tier == MemoryTier.SHORT_TERM or tier is None:
            self.short_term_memory.clear()
        if tier == MemoryTier.LONG_TERM or tier is None:
            self.long_term_memory.clear()
        
        if tier is None:
            self.tag_index.clear()
            self.session_index.clear()
    
    def stats(self) -> Dict[str, int]:
        """Get memory statistics"""
        return {
            'working': len(self.working_memory),
            'short_term': len(self.short_term_memory),
            'long_term': len(self.long_term_memory),
            'total': len(self.working_memory) + len(self.short_term_memory) + len(self.long_term_memory),
            'tags': len(self.tag_index),
            'sessions': len(self.session_index)
        }
