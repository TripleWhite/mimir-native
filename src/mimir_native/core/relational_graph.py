"""
Relational Memory Graph - Supermemory SOTA Architecture

Tracks relationships between memories:
- UPDATES: State mutation (contradictions/corrections)
- EXTENDS: Refinement (add details)
- DERIVES: Inference (combine multiple memories)

Based on: https://supermemory.ai/research
"""

from enum import Enum, auto
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


class RelationType(Enum):
    """Types of relationships between memories"""
    UPDATES = "updates"      # State mutation: new info contradicts old
    EXTENDS = "extends"      # Refinement: adds details without contradiction
    DERIVES = "derives"      # Inference: combines multiple memories
    RELATED = "related"      # General association


@dataclass
class MemoryNode:
    """
    A node in the memory graph.
    
    Contains the memory content plus metadata for versioning.
    """
    id: str
    content: str
    created_at: datetime
    version: int = 1
    previous_version: Optional[str] = None
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Relation:
    """
    A relationship between two memories.
    """
    source_id: str      # Earlier memory
    target_id: str      # Later memory  
    relation_type: RelationType
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RelationalMemoryGraph:
    """
    Graph-based memory system with versioning and relations.
    
    Key features from Supermemory:
    1. Version history: Track how facts evolve over time
    2. Relation tracking: Understand how memories connect
    3. Knowledge chains: Follow the evolution of facts
    """
    
    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[Relation] = []
        
        # Indexes for fast lookup
        self.version_history: Dict[str, List[str]] = {}  # original_id -> [version_ids]
        self.outgoing: Dict[str, List[str]] = {}  # node_id -> [target_ids]
        self.incoming: Dict[str, List[str]] = {}  # node_id -> [source_ids]
        
    def add_memory(self, content: str, memory_id: Optional[str] = None) -> MemoryNode:
        """
        Add a new memory to the graph.
        
        Args:
            content: The memory content
            memory_id: Optional ID (generated if not provided)
            
        Returns:
            The created MemoryNode
        """
        # Generate ID from content if not provided
        if not memory_id:
            memory_id = hashlib.md5(content.encode()).hexdigest()[:16]
        
        # Create node
        node = MemoryNode(
            id=memory_id,
            content=content,
            created_at=datetime.now()
        )
        
        self.nodes[memory_id] = node
        
        # Initialize version history
        if memory_id not in self.version_history:
            self.version_history[memory_id] = [memory_id]
        
        # Check for relations with existing memories
        self._detect_relations(node)
        
        return node
    
    def update_memory(
        self,
        old_memory_id: str,
        new_content: str
    ) -> MemoryNode:
        """
        Update an existing memory, creating a version chain.
        
        This creates an UPDATES relationship.
        
        Args:
            old_memory_id: ID of memory to update
            new_content: New content
            
        Returns:
            The new MemoryNode
        """
        if old_memory_id not in self.nodes:
            raise ValueError(f"Memory {old_memory_id} not found")
        
        old_node = self.nodes[old_memory_id]
        
        # Create new version
        new_id = f"{old_memory_id}_v{old_node.version + 1}"
        new_node = MemoryNode(
            id=new_id,
            content=new_content,
            created_at=datetime.now(),
            version=old_node.version + 1,
            previous_version=old_memory_id
        )
        
        self.nodes[new_id] = new_node
        
        # Add UPDATES relation
        self.add_relation(old_memory_id, new_id, RelationType.UPDATES)
        
        # Update version history
        original_id = self._get_original_id(old_memory_id)
        self.version_history[original_id].append(new_id)
        
        return new_node
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a relation between two memories.
        
        Args:
            source_id: Earlier/source memory
            target_id: Later/target memory
            relation_type: Type of relation
            confidence: Confidence score (0-1)
            metadata: Optional metadata
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Source or target memory not found")
        
        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.edges.append(relation)
        
        # Update indexes
        if source_id not in self.outgoing:
            self.outgoing[source_id] = []
        self.outgoing[source_id].append(target_id)
        
        if target_id not in self.incoming:
            self.incoming[target_id] = []
        self.incoming[target_id].append(source_id)
    
    def _detect_relations(self, new_node: MemoryNode):
        """
        Automatically detect relations between new memory and existing ones.
        """
        for existing_id, existing_node in self.nodes.items():
            if existing_id == new_node.id:
                continue
            
            # Check for contradiction (UPDATES)
            if self._is_contradiction(existing_node.content, new_node.content):
                self.add_relation(
                    existing_id, new_node.id, 
                    RelationType.UPDATES,
                    confidence=0.9
                )
                continue
            
            # Check for extension (EXTENDS)
            if self._is_extension(existing_node.content, new_node.content):
                self.add_relation(
                    existing_id, new_node.id,
                    RelationType.EXTENDS,
                    confidence=0.8
                )
                continue
            
            # Check for derivation (DERIVES)
            if self._is_derivation(existing_node.content, new_node.content):
                self.add_relation(
                    existing_id, new_node.id,
                    RelationType.DERIVES,
                    confidence=0.7
                )
    
    def _is_contradiction(self, old: str, new: str) -> bool:
        """
        Detect if new content contradicts old content.
        
        Examples:
        - "favorite color is Blue" → "favorite color is now Green"
        - "lives in NY" → "moved to LA"
        """
        old_lower = old.lower()
        new_lower = new.lower()
        
        # Same subject, different attribute
        # Extract key patterns
        contradiction_patterns = [
            (r'(favorite\s+\w+)\s+is\s+(\w+)', r'\1\s+is\s+(now\s+)?(?!\2)(\w+)'),
            (r'(lives?\s+in)\s+(\w+)', r'(moved\s+to|now\s+in)\s+(?!\2)(\w+)'),
            (r'(works?\s+(at|for))\s+(\w+)', r'(now\s+works?\s+\2|switched\s+to)\s+(?!\3)(\w+)'),
        ]
        
        for old_pattern, new_pattern in contradiction_patterns:
            old_match = re.search(old_pattern, old_lower)
            if old_match:
                new_match = re.search(new_pattern, new_lower)
                if new_match:
                    return True
        
        # Direct negation
        negation_words = ['not', 'no longer', 'changed', 'updated', 'now']
        if any(word in new_lower for word in negation_words):
            # Check for overlapping content
            old_words = set(old_lower.split())
            new_words = set(new_lower.split())
            if len(old_words & new_words) > len(old_words) * 0.5:
                return True
        
        return False
    
    def _is_extension(self, old: str, new: str) -> bool:
        """
        Detect if new content extends old content with details.
        
        Examples:
        - "works at Company" → "works at Company as a Manager"
        - "has a cat" → "has a cat named Whiskers"
        """
        old_lower = old.lower()
        new_lower = new.lower()
        
        # New contains old as substring
        if old_lower in new_lower and len(new) > len(old):
            return True
        
        # Same subject, more details
        old_words = set(old_lower.split())
        new_words = set(new_lower.split())
        
        # High overlap but new has more words
        overlap = len(old_words & new_words)
        if overlap >= len(old_words) * 0.6 and len(new_words) > len(old_words):
            return True
        
        return False
    
    def _is_derivation(self, existing: str, new: str) -> bool:
        """
        Detect if new content derives from existing.
        
        This is inference from combining information.
        """
        # For now, simple heuristic
        # In practice, this would use LLM to detect logical inference
        existing_words = set(existing.lower().split())
        new_words = set(new.lower().split())
        
        # Some overlap but distinct content
        overlap = len(existing_words & new_words)
        if 0.3 < overlap / max(len(existing_words), 1) < 0.7:
            return True
        
        return False
    
    def get_version_history(self, memory_id: str) -> List[MemoryNode]:
        """Get all versions of a memory"""
        original_id = self._get_original_id(memory_id)
        version_ids = self.version_history.get(original_id, [memory_id])
        return [self.nodes[mid] for mid in version_ids if mid in self.nodes]
    
    def get_latest_version(self, memory_id: str) -> Optional[MemoryNode]:
        """Get the latest version of a memory"""
        history = self.get_version_history(memory_id)
        if history:
            return max(history, key=lambda n: n.version)
        return None
    
    def get_related_memories(
        self,
        memory_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[Tuple[MemoryNode, RelationType]]:
        """
        Get memories related to a given memory.
        
        Args:
            memory_id: The memory to find relations for
            relation_type: Optional filter by relation type
            
        Returns:
            List of (memory_node, relation_type) tuples
        """
        related = []
        
        for edge in self.edges:
            if edge.source_id == memory_id:
                if relation_type is None or edge.relation_type == relation_type:
                    if edge.target_id in self.nodes:
                        related.append((self.nodes[edge.target_id], edge.relation_type))
            
            elif edge.target_id == memory_id:
                # Reverse relation
                if relation_type is None or edge.relation_type == relation_type:
                    if edge.source_id in self.nodes:
                        related.append((self.nodes[edge.source_id], edge.relation_type))
        
        return related
    
    def get_knowledge_chain(self, memory_id: str) -> List[MemoryNode]:
        """
        Get the full knowledge chain starting from a memory.
        
        Follows UPDATES relations to show evolution.
        """
        chain = []
        current_id = memory_id
        
        # Go backwards to find original
        while current_id in self.nodes:
            node = self.nodes[current_id]
            chain.append(node)
            
            if node.previous_version:
                current_id = node.previous_version
            else:
                break
        
        # Reverse to get chronological order
        chain.reverse()
        
        # Go forwards to find updates
        current_id = memory_id
        while current_id in self.outgoing:
            next_ids = [
                tid for tid in self.outgoing[current_id]
                if any(e.target_id == tid and e.relation_type == RelationType.UPDATES 
                      for e in self.edges)
            ]
            if next_ids:
                current_id = next_ids[0]  # Take first update
                if current_id in self.nodes:
                    chain.append(self.nodes[current_id])
            else:
                break
        
        return chain
    
    def query(
        self,
        content_filter: Optional[str] = None,
        relation_type: Optional[RelationType] = None,
        limit: int = 10
    ) -> List[MemoryNode]:
        """
        Query the memory graph.
        """
        results = []
        
        for node in self.nodes.values():
            # Filter by content
            if content_filter and content_filter.lower() not in node.content.lower():
                continue
            
            # Filter by relation type (must have at least one such relation)
            if relation_type:
                has_relation = any(
                    (e.source_id == node.id or e.target_id == node.id) and 
                    e.relation_type == relation_type
                    for e in self.edges
                )
                if not has_relation:
                    continue
            
            results.append(node)
        
        # Sort by creation time (most recent first)
        results.sort(key=lambda n: n.created_at, reverse=True)
        
        return results[:limit]
    
    def _get_original_id(self, memory_id: str) -> str:
        """Get the original ID in a version chain"""
        # Remove version suffix if present
        if '_v' in memory_id:
            return memory_id.split('_v')[0]
        return memory_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'nodes': {
                nid: {
                    'id': n.id,
                    'content': n.content,
                    'created_at': n.created_at.isoformat(),
                    'version': n.version,
                    'previous_version': n.previous_version
                }
                for nid, n in self.nodes.items()
            },
            'edges': [
                {
                    'source_id': e.source_id,
                    'target_id': e.target_id,
                    'relation_type': e.relation_type.value,
                    'confidence': e.confidence,
                    'created_at': e.created_at.isoformat(),
                    'metadata': e.metadata
                }
                for e in self.edges
            ]
        }
    
    def stats(self) -> Dict[str, int]:
        """Get graph statistics"""
        return {
            'total_memories': len(self.nodes),
            'total_relations': len(self.edges),
            'version_chains': len(self.version_history),
            'updates': sum(1 for e in self.edges if e.relation_type == RelationType.UPDATES),
            'extends': sum(1 for e in self.edges if e.relation_type == RelationType.EXTENDS),
            'derives': sum(1 for e in self.edges if e.relation_type == RelationType.DERIVES),
        }


import re
